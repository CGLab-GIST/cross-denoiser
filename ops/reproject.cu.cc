/* 
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#define GOOGLE_CUDA

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "cuda_runtime.h"
#include "vector_types.h"
#include "cuda_utils.cuh"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// This function is mostly taken from Falcor (SVGFReproject.slang)
__device__ bool isReprojValid(const int2 &imageDim, const int2 &coord,                                  //
                              const float &Z, const float &Zprev, const float &fwidthZ,                 //
                              const float3 &normal, const float3 &normalPrev, const float &fwidthNormal //
)
{
    // check whether reprojected pixel is inside of the screen
    if (any(coord < make_int2(1, 1)) || any(coord > (imageDim - make_int2(1, 1))))
        return false;

    // check if deviation of depths is acceptable
    if (abs(Zprev - Z) / (fwidthZ + 1e-2f) > 10.f)
        return false;

    // check normals for compatibility
    if (distance(normal, normalPrev) / (fwidthNormal + 1e-2) > 4.0f)
        return false;

    return true;
}

// This function is mostly taken from Falcor (SVGFReproject.slang)
__global__ void Reproject(
    bool *success,             // [1, H, W, 1]: Output success flag whether reprojection is successful
    float **output,            // [1, H, W, ?]: Output reprojected vectors
    const float *mvec,         // [1, H, W, 3]: Motion vector
    const int *dims,           // [?]: Dimensions for each input
    const float **input,       // [1, H, W, ?]: Input vectors
    const float *linearZ,      // [1, H, W, 3]: Depth and its screen space derivative
    const float *prevLinearZ,  // [1, H, W, 3]: Preivous depth and its screen space derivative
    const float *normal,       // [1, H, W, 3]: Normal
    const float *prevNormal,   // [1, H, W, 3]: Previous normal
    const float *pnFwidth,     // [1, H, W, 3]: Screen space derivative ofor position and normal
    const float *opacity,      // [1, H, W, 1]: Opacity
    const float *prev_opacity, // [1, H, W, 1]: Previous opacity
    const uint32_t batchOffset, const int width, const int height, const int num_dims)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;
    const int2 ipos = make_int2(cx, cy);
    const int2 imageDim = make_int2(width, height);

    for (int d = 0; d < num_dims; ++d)
    {
        int dim = dims[d];
        for (int c = 0; c < dim; ++c)
            output[d][(batchOffset + cIdx) * dim + c] = 0.f;
    }

    // Check opacity. Do not reproject if opacity of current pixel is 0.
    if (opacity[batchOffset + cIdx] == 0.f)
    {
        success[batchOffset + cIdx] = false;
        return;
    }

    const float2 posH = make_float2(cx, cy);
    const float2 motion = make_float2(mvec[(batchOffset + cIdx) * 3 + 0], mvec[(batchOffset + cIdx) * 3 + 1]);
    const float normalFwidth = pnFwidth[(batchOffset + cIdx) * 3 + 1];

    // +0.5 to account for texel center offset
    const int2 iposPrev = float2int(posH + motion * imageDim + make_float2(0.5f, 0.5f));

    const float2 depth = make_float2(linearZ[(batchOffset + cIdx) * 3 + 0], linearZ[(batchOffset + cIdx) * 3 + 1]);
    const float3 norm = make_float3(normal[(batchOffset + cIdx) * 3 + 0], normal[(batchOffset + cIdx) * 3 + 1], normal[(batchOffset + cIdx) * 3 + 2]);

    bool v[4];
    const float2 posPrev = floor(posH) + motion * imageDim;
    const int2 offset[4] = {make_int2(0, 0), make_int2(1, 0), make_int2(0, 1), make_int2(1, 1)};

    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        float2 depthPrev;
        float3 normalPrev;
        const int2 loc = float2int(posPrev) + offset[sampleIdx];
        const int locIdx = loc.y * imageDim.x + loc.x;
        if (loc.x >= imageDim.x || loc.x < 0 || loc.y >= imageDim.y || loc.y < 0 || prev_opacity[batchOffset + locIdx] == 0.f)
        {
            v[sampleIdx] = false;
        }
        else
        {
            depthPrev = make_float2(prevLinearZ[(batchOffset + locIdx) * 3 + 0], prevLinearZ[(batchOffset + locIdx) * 3 + 1]);
            normalPrev = make_float3(prevNormal[(batchOffset + locIdx) * 3 + 0], prevNormal[(batchOffset + locIdx) * 3 + 1], prevNormal[(batchOffset + locIdx) * 3 + 2]);
            v[sampleIdx] = isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth);
            valid = valid || v[sampleIdx];
        }
    }

    if (valid)
    {
        float sumw = 0;
        float x = fracf(posPrev.x);
        float y = fracf(posPrev.y);

        // bilinear weights
        const float w[4] = {(1 - x) * (1 - y),
                            x * (1 - y),
                            (1 - x) * y,
                            x * y};

        // perform the actual bilinear interpolation
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            const int2 loc = float2int(posPrev) + offset[sampleIdx];
            const int locIdx = loc.y * imageDim.x + loc.x;

            if (v[sampleIdx])
            {
                for (int d = 0; d < num_dims; ++d)
                {
                    int dim = dims[d];
                    for (int c = 0; c < dim; ++c)
                        output[d][(batchOffset + cIdx) * dim + c] += w[sampleIdx] * input[d][(batchOffset + locIdx) * dim + c];
                }
                sumw += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01f);
        if (valid)
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][(batchOffset + cIdx) * dim + c] /= sumw;
            }
        }
        else
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][(batchOffset + cIdx) * dim + c] = 0.f;
            }
        }
    }

    if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float nValid = 0.0;

        // this code performs a binary descision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                const int2 p = iposPrev + make_int2(xx, yy);
                const int pIdx = p.y * imageDim.x + p.x;
                if (p.x >= imageDim.x || p.x < 0 || p.y >= imageDim.y || p.y < 0 || prev_opacity[pIdx] == 0.f)
                {
                    // Outside window
                }
                else
                {
                    // Inside window
                    float2 depthPrev = make_float2(prevLinearZ[(batchOffset + pIdx) * 3 + 0], prevLinearZ[(batchOffset + pIdx) * 3 + 1]);
                    float3 normalPrev = make_float3(prevNormal[(batchOffset + pIdx) * 3 + 0], prevNormal[(batchOffset + pIdx) * 3 + 1], prevNormal[(batchOffset + pIdx) * 3 + 2]);

                    if (isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth))
                    {
                        for (int d = 0; d < num_dims; ++d)
                        {
                            int dim = dims[d];
                            for (size_t c = 0; c < dim; ++c)
                                output[d][(batchOffset + cIdx) * dim + c] += input[d][(batchOffset + pIdx) * dim + c];
                        }
                        nValid += 1.f;
                    }
                }
            }
        }

        if (nValid > 0)
        {
            valid = true;
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = dims[d];
                for (size_t c = 0; c < dim; ++c)
                    output[d][(batchOffset + cIdx) * dim + c] /= nValid;
            }
        }
    }

    if (!valid)
    {
        for (int d = 0; d < num_dims; ++d)
        {
            int dim = dims[d];
            for (size_t c = 0; c < dim; ++c)
                output[d][(batchOffset + cIdx) * dim + c] = 0.f;
        }
    }

    success[batchOffset + cIdx] = valid;
}

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
    const float *prev_opacity,
    float **output,
    bool *success,
    const int batches, const int num_dims, const int height, const int width)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < batches; ++b)
    {
        uint32_t batchOffset = b * height * width;
        Reproject<<<grid, threads, 0, _dev.stream()>>>(
            // output
            success, output,
            // input
            mvec, dims, input_list, linearZ, prevLinearZ, normal, prevNormal, pnFwidth, opacity, prev_opacity,
            // misc.
            batchOffset, width, height, num_dims
            //
        );
    }
}

#endif // GOOGLE_CUDA
