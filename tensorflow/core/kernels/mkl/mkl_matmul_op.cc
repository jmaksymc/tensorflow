/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

// This file uses MKL CBLAS xGEMM for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) operations.
// We currently register this kernel only for MKL supported data
// types (float, double, complex64, complex128). The macro INTEL_MKL is defined
// by the build system only when MKL is chosen as an option at configure stage
// and when it is undefined at build time, this file becomes an empty
// compilation unit

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "tensor_adapter.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/lib/core/errors.h"

using dnnl::engine;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <class T> struct DnnlDataType;

template <> struct DnnlDataType<float> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::f32;
};

template <> struct DnnlDataType<bfloat16> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::bf16;
};

template <> struct DnnlDataType<int8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::s8;
};

template <> struct DnnlDataType<uint8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::u8;
};

class BuildAttrs {
public:
    static constexpr float noScale = 1.f;

    BuildAttrs& Scale(float scale) {
        if (scale != noScale) {
            attr_.set_output_scales(0, {scale});
            empty = false;
        }
        return *this;
    }

    BuildAttrs& Scale(int mask, std::vector<float> scale) {
        attr_.set_output_scales(mask, scale);
        empty = false;
        return *this;
    }

    BuildAttrs& Eltwise(dnnl::algorithm algo, float alpha = 0, float beta = 0, float scale = 1.f) {
        post_ops_.append_eltwise(scale, algo, alpha, beta);
        empty = false;
        return *this;
    }

    BuildAttrs& Sum(float scale = 1.f) {
        post_ops_.append_sum(scale);
        empty = false;
        return *this;
    }

    BuildAttrs& Binary(dnnl::algorithm algo, dnnl::memory::desc memory_desc) {
        post_ops_.append_binary(algo, memory_desc);
        empty = false;
        return *this;
    }

    bool Empty(){
        return empty;
    }

    operator dnnl::primitive_attr() const { return MakeAttr_(); }

private:
    dnnl::primitive_attr MakeAttr_() const {
        auto result = attr_;
        result.set_post_ops(post_ops_);
        return result;
    }

    dnnl::primitive_attr attr_;
    dnnl::post_ops post_ops_;
    bool empty = true;
};

class DataSource {
public:
    DataSource(const dnnl::memory& mem = {}, BuildAttrs attr = {})
        : mem_{mem} 
        , attr_{attr} {}

    DataSource(const DataSource& other) = default;
    DataSource(DataSource&& other) = default;
    DataSource& operator=(const DataSource& other) = default;
    DataSource& operator=(DataSource&& other) = default;
    virtual ~DataSource() = default;

    virtual dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) {
        if (!mem_) {
             return mem_;
        }

        if (attr_.Empty() && mem_.get_engine() == stm.get_engine() && mem_.get_desc() == md) {
            return mem_;
        }
        dnnl::memory result{md, stm.get_engine()};

        // No need to check for nullptr, implicitly convert to dnnl::primitive_attr
        dnnl::reorder rdr{mem_, result, attr_};
        rdr.execute(stm, mem_, result);
        return result;
    }

private:
    dnnl::memory mem_;
    BuildAttrs attr_;
};

class CachedDataSource : public DataSource {
public:
    using DataSource::DataSource;

    dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) override {
        if (!cached_mem_ || cached_mem_.get_engine() != stm.get_engine() || cached_mem_.get_desc() != md) {
            cached_mem_ = DataSource::GetData(stm, md);
        }
        return cached_mem_;
    }

private:
    dnnl::memory cached_mem_;
};

class InnerProduct {
public:
  InnerProduct(const engine& eng,
      const memory::desc& src_md, const memory::desc& weights_md,
      const memory::desc& bias_md, const memory::desc& dst_md,
      const primitive_attr& attr)
      : prim_{dnnl::inner_product_forward::primitive_desc{dnnl::inner_product_forward::desc{dnnl::prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md}, attr, eng}} {}

  void Compute(dnnl::stream& stm, DataSource& src, DataSource& weights, DataSource& bias, dnnl::memory& dst_memory) {
    const auto prim_desc = PrimDesc();
    assert(prim_desc.dst_desc() == dst_memory.get_desc());

    auto src_memory = src.GetData(stm, prim_desc.src_desc());
    auto weights_memory = weights.GetData(stm, prim_desc.weights_desc());
    auto bias_memory = bias.GetData(stm, prim_desc.bias_desc());

    prim_.execute(stm, {
      { DNNL_ARG_SRC, src_memory },
      { DNNL_ARG_WEIGHTS, weights_memory },
      { DNNL_ARG_BIAS, bias_memory },
      { DNNL_ARG_DST, dst_memory } });
  }

  inner_product_forward::primitive_desc PrimDesc() const {
    auto c_desc = prim_.get_primitive_desc();
    return inner_product_forward::primitive_desc{const_cast<dnnl_primitive_desc_t>(c_desc)};
  }

  const dnnl::primitive& Prim() const{
    return prim_;
  }
private:
  primitive prim_;
};

struct InnerProductDims {
    InnerProductDims(int batch, int m, int n, int k)
        : src_tz{batch * m, k}
        , weights_tz{n, k}
        , bias_tz{n}
        , dst_tz{batch * m, n} {}

    dnnl::memory::dims src_tz;
    dnnl::memory::dims weights_tz;
    dnnl::memory::dims bias_tz;
    dnnl::memory::dims dst_tz;
};

inline InnerProduct MakeInnerProduct(const dnnl::engine& eng, int batch, int m, int n, int k, 
                        dnnl::memory::data_type src_dt, dnnl::memory::data_type weights_dt,
                        dnnl::memory::data_type bias_dt, dnnl::memory::data_type dst_dt,
                        const dnnl::primitive_attr& attr = {}) {

    const InnerProductDims dims{batch, m, n, k};

    // plain memory format can be defined using empty strides argument
    dnnl::memory::dims plain{};
    using fmt = dnnl::memory::format_tag;

    const auto src_md     = dnnl::memory::desc(dims.src_tz , src_dt, plain);
    const auto weights_md = dnnl::memory::desc(dims.weights_tz, weights_dt, fmt::any);
    const auto bias_md    = dnnl::memory::desc(dims.bias_tz, bias_dt, plain);
    const auto dst_md     = dnnl::memory::desc(dims.dst_tz, dst_dt, plain);

    return InnerProduct{eng, src_md, weights_md, bias_md, dst_md, attr};
}

template <typename T_src, typename T_wei, typename T_bias, typename T_dst>
InnerProduct MakeInnerProduct(const dnnl::engine& eng, int batch, int m, int n, int k, const dnnl::primitive_attr& attr = {}) {
    const auto src_dt = DnnlDataType<T_src>::value;
    const auto weights_dt = DnnlDataType<T_wei>::value;
    const auto bias_dt = DnnlDataType<T_bias>::value;
    const auto dst_dt = DnnlDataType<T_dst>::value;
    return MakeInnerProduct(eng, batch, m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
}

template <typename Device, typename T, bool native_format = false>
class MklMatMulOp : public MklDnnMatMulOpBase<T, T> {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx) 
      : MklDnnMatMulOpBase<T, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    if (AreWeightsFrozen()) {
      this->is_weight_const_ = true;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    // const Tensor& bias_tensor = ctx->input(this->kInputIndexBias);


    // MklDnnShape src_mkl_shape;
    // MklDnnShape weight_mkl_shape;
    // std::cout << "CHECKPOINT 1" << std::endl;
    // GetMklShape(ctx, this->kInputIndexSrc, &src_mkl_shape, native_format);
    // GetMklShape(ctx, this->kInputIndexWeight, &weight_mkl_shape, native_format);
    // OP_REQUIRES(ctx, !weight_mkl_shape.IsMklTensor(),
    //             errors::InvalidArgument("Weight should not be in MKL Layout."));


    // Get shapes of input tensors
    auto src_tf_shape = src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    // const int dim_pair[] = {1, transpose_b_ ? 1: 0};
    // const int channel = weight_tf_shape.dim_size(1 - dim_pair[1]);
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    const int channel = weight_tf_shape.dim_size(1 - dim_pair[0].second);

    Tensor bias_tensor(DT_FLOAT, TensorShape({channel}));
    auto bias_flat = bias_tensor.flat<float>();
    for (int i = 0; i < bias_tensor.NumElements(); i++) {
      bias_flat(i) = 0;
    }

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                errors::InvalidArgument("In[1] is not a matrix"));
    for (int i = 0; i < bias_tensor.dims() - 1; i++){
      OP_REQUIRES(
        ctx, bias_tensor.dim_size(i) == 1,
        errors::InvalidArgument("For bias_dims > 1, all except the "
                                "last dimension (channel) must be 1, got: ",
                                bias_tensor.shape().DebugString()));
    }

    const int m = src_tf_shape.dim_size(1 - dim_pair[0].first);
    const int k = src_tf_shape.dim_size(dim_pair[0].first);

    OP_REQUIRES(
        ctx, k == weight_tf_shape.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
            ", In[1]: ", weight_tf_shape.DebugString()));
    OP_REQUIRES(ctx, bias_tensor.dim_size(bias_tensor.dims() - 1) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel size: ",
                    bias_tensor.shape().DebugString(), " vs. ", channel));

    auto eng = engine(engine::kind::cpu, 0);
    auto stm = stream(eng);

    TensorAdapter tensor_adapter;
    auto src = tensor_adapter.AsDnnlMemory(src_tensor, eng);

    dnnl::primitive_attr inner_product_attr;

    auto innerProd = MakeInnerProduct<T, T, T, T>(eng, batch_, m, channel, k, inner_product_attr);

    auto reshapeTo2D = [](const dnnl::memory& mem) {
            auto mem_desc = mem.get_desc();
            auto mem_dims = mem_desc.dims();
            assert(mem_dims.size() == 3);
            dnnl::memory::dims new_dims{mem_dims[0] * mem_dims[1], mem_dims[2]};
            return dnnl::memory{mem_desc.reshape(new_dims), mem.get_engine(), mem.get_data_handle()};
        };

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    TensorShape output_tf_shape({m, channel});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_tf_shape, &dst_tensor));

    memory::dims src_dims = memory::dims({m, k});
    memory::dims weight_dims = memory::dims({channel, k});
    memory::dims bias_dims = memory::dims({channel});
    memory::dims dst_dims = memory::dims({m, channel});

    dnnl::memory::dims plain{};
    using fmt = dnnl::memory::format_tag;

    const auto src_md = memory::desc(src_dims, memory::data_type::f32, plain);
    const auto weights_md = memory::desc(weight_dims, memory::data_type::f32, fmt::any);
    const auto bias_md = memory::desc(bias_dims, memory::data_type::f32, plain);
    const auto dst_md = memory::desc(dst_dims, memory::data_type::f32, plain);

    auto src_mem = dnnl::memory(src_md, eng);
    auto weights_mem = dnnl::memory(weights_md, eng);
    auto bias_mem = dnnl::memory(bias_md, eng);
    auto dst_mem = dnnl::memory(dst_md, eng);

    auto reshapedSrc = reshapeTo2D(src);
    auto reshapedOutput = reshapeTo2D(dst_mem);

    mainMatMul_->Compute(stm, reshapedSrc, weights_mem, bias_mem, reshapedOutput);

    // // m - batch
    // // n - channel
    // // k - k
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  std::unique_ptr<InnerProduct> mainMatMul_;
  int batch_ = 1;
  // --------------------------------------------------------------------------
  //
  // @brief Matrix-Matrix Multiplication with FP32 tensors, a, b, c using CBLAS
  // interface. c = op(a) * op(b)
  //
  // @param transa  Specifies the form of op(a) used in MatMul. If transa is
  // true, then op(a) = a^T, otherwise op(a) = a
  //
  // @param transb  Specifies the form of op(b) used in MatMul. If transb is
  // true, then op(b) = b^T, otherwise op(b) = b
  //
  // @param m       Specifies the number of rows of the matrix op(a) and of the
  // matrix c. The value of m must be at least zero.
  //
  // @param n       Specifies the number of columns of the matrix op(b) and the
  // number of columns of the matrix c. The value of n must be at least zero.
  //
  // @param k       Specifies the number of columns of the matrix op(a) and the
  // number of rows of the matrix op(b)
  //
  // @param a       Address of matrix a
  //
  // @param lda     Leading dimension of 'a' matrix. This is set at calling site
  // depending on transa parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // lda = max(1,k) when transa is false, otherwise lda = max(1,m)
  //
  // @param b       Address of matrix b
  //
  // @param ldb     Leading dimension of 'b' matrix. This is set at calling site
  // depending on transb parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // ldb = max(1,n) when transb is false, otherwise ldb = max(1,k)
  //
  // @param c       Address of matrix c
  //
  // @param ldc     Leading dimension of 'c' matrix. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows, max(1,n)
  //
  // --------------------------------------------------------------------------
};

#define REGISTER_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                \
      Name("_MklMatMul")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T")                         \
          .Label(mkl_op_registry::kMklNameChangeOpLabel), \
      MklMatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

// TODO(intel-tf): Consider template specialization when adding/removing
// additional types
TF_CALL_float(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
}  // namespace tensorflow
#endif  // INTEL_MKL