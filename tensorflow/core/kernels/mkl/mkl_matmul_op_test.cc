/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include <functional>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow{
namespace {
namespace MKLMatmulTestDefs{
    typedef std::tuple<
    DataType,
    std::vector<long long int>, // m
    std::vector<long long int>, // k
    std::vector<long long int>, // n
    std::vector<bool>,          // transA
    std::vector<bool>          // transB
    > MklMatMulTestParams;
    std::vector<DataType> dataTypes {
        DataType::DT_FLOAT
        // DataType::DT_BFLOAT16
    };
    std::vector<std::vector<long long int>> M = {{1}, {8}, {16}, {128}};
    std::vector<std::vector<long long int>> K = {{512}};
    std::vector<std::vector<long long int>> N = {{512}};
    std::vector<std::vector<bool>> TRANSA = {{true}, {false}};
    std::vector<std::vector<bool>> TRANSB = {{true}, {false}};
} // namespace MKLMatmulTestDefs

using namespace MKLMatmulTestDefs;

class MklMatMulTestBase:
  public ::testing::WithParamInterface<MKLMatmulTestDefs::MklMatMulTestParams>,
  public OpsTestBase {
private:
    DataType input_type;
    std::vector<long long int> vec_m;
    std::vector<long long int> vec_k;
    std::vector<long long int> vec_n;
    std::vector<bool> vec_transa;
    std::vector<bool> vec_transb;

    Tensor input_0;
    Tensor input_1;

    bool in_transA;
    bool in_transB;

    Tensor out_mkl;
    Tensor out_default;

    static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
    }

    void runDefault() {
        auto root = tensorflow::Scope::NewRootScope();
        auto input_op_0 =
            ops::Const(root.WithOpName("input_0"), Input::Initializer(input_0));
        auto input_op_1 =
            ops::Const(root.WithOpName("input_1"), Input::Initializer(input_1));
        auto attr =
            ops::MatMul::TransposeA(in_transA).TransposeB(in_transB);
        Output next_op = ops::MatMul(root.WithOpName("matmul"), input_op_0, input_op_1, attr);
        string last_op = "matmul";
        RunAndFetch(root, last_op, &out_default);
    };

    void runMkl() {
        TF_EXPECT_OK(
            NodeDefBuilder("mkl_matmul_op", "_MklMatMul")
                .Input(FakeInput(input_type))
                .Input(FakeInput(input_type))
                .Attr("transpose_a", in_transA)
                .Attr("transpose_b", in_transB)
                .Attr("_kernel", "MklNameChangeOp")
                .Finalize(node_def()));
        TF_EXPECT_OK(InitOp());
        switch(input_type) {
            case DT_FLOAT:
                AddInputFromArray<float>(input_0.shape(), input_0.flat<float>());
                AddInputFromArray<float>(input_1.shape(), input_1.flat<float>());
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
        TF_EXPECT_OK(RunOpKernel());
        out_mkl = *GetOutput(0);
    }  
public:
    static std::string getTestCaseName(::testing::TestParamInfo<MklMatMulTestParams> obj) {
        DataType input_type;
        std::vector<long long int> vec_m;
        std::vector<long long int> vec_k;
        std::vector<long long int> vec_n;
        std::vector<bool> vec_transa;
        std::vector<bool> vec_transb;

        std::tie(input_type, vec_m, vec_k, vec_n, vec_transa, vec_transb) = obj.param;
        std::ostringstream result;

        result << "MklMatMul_type_";
        switch(input_type) {
            case DT_FLOAT:
                result << "FLOAT";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_sizes_0";
        vec_transa[0] ? result << "_" << vec_k[0] << "_" << vec_m[0] << "_TransA_true" : result << "_" << vec_m[0] << "_" << vec_k[0] << "_TransA_false";
        result << "_sizes_1";
        vec_transb[0] ? result << "_" << vec_n[0] << "_" << vec_k[0] << "_TransB_true" : result << "_" << vec_k[0] << "_" << vec_n[0] << "_TransB_false";

        return result.str();
    }

    void SetUp() {
        std::tie(input_type, vec_m, vec_k, vec_n, vec_transa, vec_transb) = this->GetParam();
        input_0 = Tensor(input_type, vec_transa[0] ? TensorShape({vec_k[0], vec_m[0]}) : TensorShape({vec_m[0], vec_k[0]}));
        input_1 = Tensor(input_type, vec_transb[0] ? TensorShape({vec_n[0], vec_k[0]}) : TensorShape({vec_k[0], vec_n[0]}));
        switch(input_type) {
            case DT_FLOAT:
                input_0.flat<float>() = input_0.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>();
                input_1.flat<float>() = input_1.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>();
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }

        in_transA = vec_transa[0];
        in_transB = vec_transb[0];
    }

    void Run() {
        runDefault();
        runMkl();
    }

    void Validate() {
        ASSERT_EQ(out_default.dtype(), out_mkl.dtype());
        ASSERT_EQ(out_default.shape(), out_mkl.shape());
        test::ExpectClose(out_default, out_mkl, 1e-6);
    }
};

TEST_P(MklMatMulTestBase, CompareWithRefs) {
    SetUp();
    Run();
    Validate();
};

INSTANTIATE_TEST_CASE_P(MklMatMul, MklMatMulTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(M),
        ::testing::ValuesIn(K),
        ::testing::ValuesIn(N),
        ::testing::ValuesIn(TRANSA),
        ::testing::ValuesIn(TRANSB)),
    MklMatMulTestBase::getTestCaseName);
} // namespace
} // namespace tensorflow