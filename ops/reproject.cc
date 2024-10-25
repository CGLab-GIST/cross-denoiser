/* 
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "cuda_runtime.h"

using namespace tensorflow;

// Some debug macros
#define CUDA_CHECK(val)                                                                                       \
    {                                                                                                         \
        if (val != cudaSuccess)                                                                               \
        {                                                                                                     \
            fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
            exit(1);                                                                                          \
        }                                                                                                     \
    }

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Reproject")
    .Attr("T: list(type)")
    .Input("mvec: float")
    .Input("input_list: T")
    .Input("linear_z: float")
    .Input("prev_linear_z: float")
    .Input("normal: float")
    .Input("prev_normal: float")
    .Input("pn_fwidth: float")
    .Input("opacity: float")
    .Input("prev_opacity: float")
    .Output("success: bool")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                    // Set success
                    auto B = c->Dim(c->input(0), 0);
                    auto H = c->Dim(c->input(0), 1);
                    auto W = c->Dim(c->input(0), 2);
                    c->set_output(0, c->MakeShape({B, H, W, 1}));

                    // Set shapes for each element in the output list
                    // c->set_output(1, c->input(1));

                    // Set shapes for each element in the input_list
                    int num_outputs = c->num_outputs() - 1; // Subtract success

                    // Set shapes for each element in the input_list
                    for (int i = 0; i < num_outputs; ++i)
                    {
                        c->set_output(i + 1,            // Offset by 1 to account for the 'success' output
                                      c->input(i + 1)); // Offset by 1 to account for the 'mvec' input
                    }
                    return Status(); //
                } //
    );

void ReprojectFunc(
    const GPUDevice &_dev,
    const float *mvec,
    const int *dims,
    const float **input_list,
    const float *linearZ,
    const float *prevLinearZ,
    const float *normal,
    const float *prevNormal,
    const float *pnFwidth,
    const float *opacity,
    const float *prevOpacity,
    float **output,
    bool *success,
    const int batches, const int num_dims, const int height, const int width);

class ReprojectOp : public OpKernel
{
public:
    explicit ReprojectOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext *context) override
    {
        // Get input tensors (from REGISTER_OP)
        const Tensor *mvec, *linearZ, *prevLinearZ, *normal, *prevNormal, *pnFwidth, *opacity, *prevOpacity;
        OP_REQUIRES_OK(context, context->input("mvec", &mvec));
        OP_REQUIRES_OK(context, context->input("linear_z", &linearZ));
        OP_REQUIRES_OK(context, context->input("prev_linear_z", &prevLinearZ));
        OP_REQUIRES_OK(context, context->input("normal", &normal));
        OP_REQUIRES_OK(context, context->input("prev_normal", &prevNormal));
        OP_REQUIRES_OK(context, context->input("pn_fwidth", &pnFwidth));
        OP_REQUIRES_OK(context, context->input("opacity", &opacity));
        OP_REQUIRES_OK(context, context->input("prev_opacity", &prevOpacity));

        // Get list of tensors
        OpInputList input_list;
        OP_REQUIRES_OK(context, context->input_list("input_list", &input_list));
        int num_dims = input_list.size();
        std::vector<const float *> input_list_ptr(num_dims);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            input_list_ptr[i] = input_list[i].flat<float>().data();
            dims[i] = input_list[i].shape().dim_size(3);
        }

        // Shape of input
        TensorShape input_shape = mvec->shape();
        int B = input_shape.dim_size(0);
        int H = input_shape.dim_size(1);
        int W = input_shape.dim_size(2);

        // Output success
        Tensor *success_img = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, H, W, 1}, &success_img));
        auto success_mat = success_img->tensor<bool, 4>();

        // Output list
        OpOutputList output_list(context, 1, num_dims + 1);
        std::vector<Tensor *> out_imgs(num_dims, NULL);
        OP_REQUIRES_OK(context, context->output_list("output", &output_list));
        std::vector<float *> out_mats(num_dims);
        for (int i = 0; i < num_dims; ++i)
        {
            OP_REQUIRES_OK(context, output_list.allocate(i, TensorShape{B, H, W, dims[i]}, &out_imgs[i]));
            out_mats[i] = out_imgs[i]->tensor<float, 4>().data();
        }

        if (!initialized)
        {
            // printf("Initializing ReprojectOp\n");
            // Allocate memory for dims in GPU
            CUDA_CHECK(cudaMalloc((void **)&dims_gpu_ptr, sizeof(int) * num_dims));
            // Allocate memory for pointer of input/output list in GPU
            CUDA_CHECK(cudaMalloc((void **)&input_gpu_ptr, sizeof(float *) * num_dims));
            CUDA_CHECK(cudaMalloc((void **)&out_gpu_ptr, sizeof(float *) * num_dims));
            initialized = true;
        }

        CUDA_CHECK(cudaMemcpy(dims_gpu_ptr, dims.data(), sizeof(int) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input_gpu_ptr, input_list_ptr.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out_gpu_ptr, out_mats.data(), sizeof(float *) * num_dims, cudaMemcpyHostToDevice));

        ReprojectFunc(context->eigen_device<GPUDevice>(),
                      mvec->flat<float>().data(),
                      dims_gpu_ptr,
                      input_gpu_ptr,
                      linearZ->flat<float>().data(),
                      prevLinearZ->flat<float>().data(),
                      normal->flat<float>().data(),
                      prevNormal->flat<float>().data(),
                      pnFwidth->flat<float>().data(),
                      opacity->flat<float>().data(),
                      prevOpacity->flat<float>().data(),
                      out_gpu_ptr,
                      success_mat.data(),
                      B, num_dims, H, W);

        // CUDA_CHECK(cudaFree(out_gpu_ptr));
    }

private:
    static bool initialized;
    static int *dims_gpu_ptr;
    static const float **input_gpu_ptr;
    static float **out_gpu_ptr;
};
bool ReprojectOp::initialized = false;
int *ReprojectOp::dims_gpu_ptr = nullptr;
const float **ReprojectOp::input_gpu_ptr = nullptr;
float **ReprojectOp::out_gpu_ptr = nullptr;

REGISTER_KERNEL_BUILDER(
    Name("Reproject")
        .Device(DEVICE_GPU),
    ReprojectOp);
