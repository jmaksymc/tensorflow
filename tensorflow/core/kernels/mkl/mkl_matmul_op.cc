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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

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

    const int batch = src_tf_shape.dim_size(1 - dim_pair[0].first);
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

    memory::dims src_dims = memory::dims({batch, k});
    memory::dims weight_dims = memory::dims({channel, k});
    memory::dims bias_dims = memory::dims({channel});
    memory::dims dst_dims = memory::dims({batch, channel});
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b_ ? memory::format_tag::oi : memory::format_tag::io;

    MklDnnMatMulFwdParams matmul_params(
      src_dims, weight_dims, bias_dims, dst_dims, src_format,
      (this->is_weight_const_) ? memory::format_tag::any : weight_format,
      memory::format_tag::nc, this->is_weight_const_);
    // ExtendMklDnnMatMulFwdParams(ctx, matmul_params);

    MklDnnMatMulFwdPrimitive<T, T, T, T, T>* matmul_prim =
      MklDnnMatMulFwdPrimitiveFactory<T, T, T, T, T>::Get(matmul_params, 0);

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> matmul_pd =
      matmul_prim->GetPrimitiveDesc();
    
    // MklDnnShape output_mkl_shape;
    // output_mkl_shape.SetMklTensor(false);

    TensorShape output_tf_shape({batch, channel});
    // std::cout << "CHECKPOINT 2" << std::endl;
    // AllocateOutputSetMklShape(ctx, 0, &dst_tensor, output_tf_shape,
    //                           output_mkl_shape, native_format);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_tf_shape, &dst_tensor));

    // std::cout << "CHECKPOINT 3" << std::endl;

    if (batch == 0 || channel == 0) {
      return;
    }

    try {
      T* src_data = const_cast<T*>(src_tensor.flat<T>().data());
      T* weight_data = const_cast<T*>(weight_tensor.flat<T>().data());
      T* bias_data = const_cast<T*>(bias_tensor.flat<T>().data());
      T* dst_data = const_cast<T*>(dst_tensor->flat<T>().data());

      // MklDnnData<T> src_mkl(&(this->cpu_engine_));
      // MklDnnData<T> weight_mkl(&(this->cpu_engine_));

      auto src_md = memory::desc(src_dims, MklDnnType<T>(), src_format);
      auto weight_md = memory::desc(weight_dims, MklDnnType<T>(), weight_format);
      
      // if (src_md != matmul_pd->src_desc()) {
      //   src_mkl.SetUsrMem(src_md, src_data);
      //   src_mkl.CheckReorderToOpMem(matmul_pd.get()->src_desc(),
      //                               this->cpu_engine_, ctx);
      //   src_data = reinterpret_cast<T*>(src_mkl.GetOpMem().get_data_handle());
      // }

      // Get cached data when weight is const.
      // const memory::desc weight_md = 
      //   memory::desc(weight_dims, MklDnnType<T>(), weight_format);
      // if (weight_md != matmul_pd->weights_desc()) {
      //   T* cached_weight_data = nullptr;

      //   if (this->is_weight_const_) {
      //     if (this->IsWeightCacheEmpty(ctx)) {
      //       this->CacheWeight(ctx, matmul_pd, cached_weight_data,
      //       weight_tensor, weight_mkl, weight_md);
      //     }
      //     cached_weight_data =
      //       this->GetCachedWeight(ctx, matmul_pd->weights_desc());
      //   }

      //   if (cached_weight_data != nullptr) {
      //     weight_data = cached_weight_data;
      //   } else {
      //     weight_mkl.SetUsrMem(weight_md, weight_data);
      //     weight_mkl.CheckReorderToOpMem(matmul_pd.get()->weights_desc(),
      //                                    this->cpu_engine_, ctx);
      //     weight_data =
      //       reinterpret_cast<T*>(weight_mkl.GetOpMem().get_data_handle());
      //   }
      // }
      std::shared_ptr<stream> cpu_stream;
      auto st = ExecuteSingleThreadedGemm(batch, channel, k, sizeof(T));
      MklDnnThreadPool eigen_tp(ctx, st ? 1 : -1);
      cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

      UserScratchPad<unsigned char> scratch_pad;
      scratch_pad.AllocateSPTensor(matmul_prim, ctx);

      // Execute matmul op
      matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                           scratch_pad.Get(), cpu_stream);
    } catch(dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }       
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
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