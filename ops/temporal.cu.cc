/* 
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "tensorflow/core/framework/shape_inference.h"

#define __CUDACC__
#ifdef __CUDACC__

// CUDA specific code
#include "cuda_utils.cuh"

#define EPSILON 1e-4f
#define FLT_MAX 10000000.f
#define ZERO4 make_float4(0.f, 0.f, 0.f, 0.f)

template <typename Func>
__device__ void fo2(int cx, int cy, int width, int height, int halfWinSize, int stepSize, Func func)
{
    int sx = cx - halfWinSize * stepSize;
    int sy = cy - halfWinSize * stepSize;
    int ex = cx + halfWinSize * stepSize;
    int ey = cy + halfWinSize * stepSize;

    int kernelIdx = 0;
    for (int iy = sy; iy <= ey; iy += stepSize)
    {
        for (int ix = sx; ix <= ex; ix += stepSize)
        {
            if (ix < 0 || ix >= width || iy < 0 || iy >= height)
            {
                kernelIdx++;
                continue;
            }
            int idx = iy * width + ix;
            func(ix, iy, idx, kernelIdx);
            kernelIdx++;
        }
    }
}

__global__ void TemporalScreening(
    float *_output,
    float *_output1,
    float *_output2,
    float *_success,
    const float *_current1,
    const float *_current2,
    const float *_prev,
    const float *_prev1,
    const float *_prev2,
    const float *_reprojSuccess,
    const int height, const int width, const int winCurr, const int winPrev)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;

    float4 current1 = get3(_current1, cIdx);
    float4 current2 = get3(_current2, cIdx);
    float4 prev1 = get3(_prev1, cIdx);
    float4 prev2 = get3(_prev2, cIdx);

    float4 current = (current1 + current2) * 0.5f;
    float4 prev = get3(_prev, cIdx);

    if (_reprojSuccess[cIdx] + 1e-4f < 1.f)
    {
        _output[3 * cIdx + 0] = current.x;
        _output[3 * cIdx + 1] = current.y;
        _output[3 * cIdx + 2] = current.z;

        _output1[3 * cIdx + 0] = current1.x;
        _output1[3 * cIdx + 1] = current1.y;
        _output1[3 * cIdx + 2] = current1.z;

        _output2[3 * cIdx + 0] = current2.x;
        _output2[3 * cIdx + 1] = current2.y;
        _output2[3 * cIdx + 2] = current2.z;

        _success[cIdx] = 0.f;
        return;
    }

    // Find min/max of the previous frame inside the window
    float4 minCurr = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f),
           minCurr1 = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f),
           minCurr2 = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f);
    float4 maxCurr = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f),
           maxCurr1 = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f),
           maxCurr2 = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f);
    fo2(cx, cy, width, height, winCurr / 2, 1 /* step size */,
        [&](const int ix, const int iy, const int idx, const int kernelIdx)
        {
            const float4 iCurr1 = get3(_current1, idx);
            const float4 iCurr2 = get3(_current2, idx);
            const float4 iCurr = (iCurr1 + iCurr2) * 0.5f;
            minCurr1 = fmin3f(minCurr1, iCurr1);
            maxCurr1 = fmax3f(maxCurr1, iCurr1);
            minCurr2 = fmin3f(minCurr2, iCurr2);
            maxCurr2 = fmax3f(maxCurr2, iCurr2);
            minCurr = fmin3f(minCurr, iCurr);
            maxCurr = fmax3f(maxCurr, iCurr);
        });

    float4 minPrev = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f),
           minPrev1 = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f),
           minPrev2 = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, 0.f);
    float4 maxPrev = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f),
           maxPrev1 = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f),
           maxPrev2 = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f);
    fo2(cx, cy, width, height, winPrev / 2, 1 /* step size */,
        [&](const int ix, const int iy, const int idx, const int kernelIdx)
        {
            if (_reprojSuccess[idx] < 1.f)
                return;

            const float4 iPrev = get3(_prev, idx);
            const float4 iPrev1 = get3(_prev1, idx);
            const float4 iPrev2 = get3(_prev2, idx);
            minPrev = fmin3f(minPrev, iPrev);
            maxPrev = fmax3f(maxPrev, iPrev);
            minPrev1 = fmin3f(minPrev1, iPrev1);
            maxPrev1 = fmax3f(maxPrev1, iPrev1);
            minPrev2 = fmin3f(minPrev2, iPrev2);
            maxPrev2 = fmax3f(maxPrev2, iPrev2);
        });

    // If the current pixel is outside the min/max range, then it is a moving pixel
    bool success, success1, success2;
    float4 output, output1, output2;

    // Check overalps two min/max
    if (AllLess3(minPrev, maxCurr) && AllLess3(minCurr, maxPrev))
    {
        output = prev;
        success = true;
    }
    else
    {
        output = current;
        success = false;
    }
    if (AllLess3(minPrev1, maxCurr1) && AllLess3(minCurr1, maxPrev1))
    {
        output1 = prev1;
        success1 = true;
    }
    else
    {
        output1 = current1;
        success1 = false;
    }

    if (AllLess3(minPrev2, maxCurr2) && AllLess3(minCurr2, maxPrev2))
    {
        output2 = prev2;
        success2 = true;
    }
    else
    {
        output2 = current2;
        success2 = false;
    }

    _output[3 * cIdx + 0] = output.x;
    _output[3 * cIdx + 1] = output.y;
    _output[3 * cIdx + 2] = output.z;

    _output1[3 * cIdx + 0] = output1.x;
    _output1[3 * cIdx + 1] = output1.y;
    _output1[3 * cIdx + 2] = output1.z;

    _output2[3 * cIdx + 0] = output2.x;
    _output2[3 * cIdx + 1] = output2.y;
    _output2[3 * cIdx + 2] = output2.z;

    _success[cIdx] = success ? 1.f : 0.f;
}
#endif

// C++ code
using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
class TemporalScreeningOp : public OpKernel
{
public:
    explicit TemporalScreeningOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("win_curr", &winCurr));
        OP_REQUIRES_OK(context, context->GetAttr("win_prev", &winPrev));
    }

    void Compute(OpKernelContext *context) override
    {
        auto current1 = context->input(0);
        auto current2 = context->input(1);
        auto prev = context->input(2);
        auto prev1 = context->input(3);
        auto prev2 = context->input(4);
        auto reprojSuccess = context->input(5);

        const GPUDevice &dev = context->eigen_device<GPUDevice>();

        // Get size
        const TensorShape &input_shape = current1.shape();
        const int nBatch = input_shape.dim_size(0);
        const int H = input_shape.dim_size(1);
        const int W = input_shape.dim_size(2);

        // Allocate output
        Tensor *output = nullptr, *output1 = nullptr, *output2 = nullptr;
        Tensor *success = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        OP_REQUIRES_OK(context, context->allocate_output(1, input_shape, &output1));
        OP_REQUIRES_OK(context, context->allocate_output(2, input_shape, &output2));
        OP_REQUIRES_OK(context, context->allocate_output(3, {nBatch, H, W, 1}, &success));

        const int blockDim = 8;
        dim3 threads(blockDim, blockDim);
        dim3 grid(iDivUp(W, blockDim), iDivUp(H, blockDim));

        TemporalScreening<<<grid, threads, 0, dev.stream()>>>(
            // Output
            output->flat<float>().data(),
            output1->flat<float>().data(),
            output2->flat<float>().data(),
            success->flat<float>().data(),
            // Input
            current1.flat<float>().data(),
            current2.flat<float>().data(),
            prev.flat<float>().data(),
            prev1.flat<float>().data(),
            prev2.flat<float>().data(),
            reprojSuccess.flat<float>().data(),
            H, W, winCurr, winPrev);
    }

private:
    int winCurr;
    int winPrev;
};

REGISTER_OP("TemporalScreening")
    .Input("current1: float")
    .Input("current2: float")
    .Input("prev: float")
    .Input("prev1: float")
    .Input("prev2: float")
    .Input("reproj_success: float")
    .Output("output: float")
    .Output("output1: float")
    .Output("output2: float")
    .Output("success: float")
    .Attr("win_curr: int >= 1")
    .Attr("win_prev: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(0));
    c->set_output(2, c->input(0));
    auto B = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    c->set_output(3, c->MakeShape({B, H, W, 1}));
    return Status(); });

REGISTER_KERNEL_BUILDER(Name("TemporalScreening").Device(DEVICE_GPU), TemporalScreeningOp);