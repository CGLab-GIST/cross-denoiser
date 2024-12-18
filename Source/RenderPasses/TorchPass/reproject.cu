#include <torch/extension.h>
#include <thrust/device_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

#define FLT_MAX2 10000000.f

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
    TensorAccessor4Db success,           // [1, H, W, 1]: Output success flag whether reprojection is successful
    TensorAccessor4D *output,            // [1, H, W, ?]: Output reprojected vectors
    const TensorAccessor4D mvec,         // [1, H, W, 3]: Motion vector
    const TensorAccessor4D *input,       // [1, H, W, ?]: Input vectors
    const TensorAccessor4D linearZ,      // [1, H, W, 3]: Depth and its screen space derivative
    const TensorAccessor4D prevLinearZ,  // [1, H, W, 3]: Preivous depth and its screen space derivative
    const TensorAccessor4D normal,       // [1, H, W, 3]: Normal
    const TensorAccessor4D prevNormal,   // [1, H, W, 3]: Previous normal
    const TensorAccessor4D pnFwidth,     // [1, H, W, 3]: Screen space derivative for position and normal
    const TensorAccessor4D opacity,      // [1, H, W, 1]: Opacity
    const TensorAccessor4D prev_opacity, // [1, H, W, 1]: Previous opacity
    const uint32_t batch, const int num_dims, const int width, const int height)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int2 ipos = make_int2(cx, cy);
    const int2 imageDim = make_int2(width, height);

    for (int d = 0; d < num_dims; ++d)
    {
        int dim = input[d].size(1);
        for (int c = 0; c < dim; ++c)
            output[d][batch][c][cy][cx] = 0.f;
    }

    // Check opacity. Do not reproject if opacity of current pixel is 0.
    if (opacity[batch][0][cy][cx] == 0.f)
    {
        success[batch][0][cy][cx] = false;
        return;
    }

    const float2 posH = make_float2(cx, cy);
    const float2 motion = make_float2(mvec[batch][0][cy][cx], mvec[batch][1][cy][cx]);
    const float normalFwidth = pnFwidth[batch][1][cy][cx];

    // +0.5 to account for texel center offset
    const int2 iposPrev = float2int(posH + motion * imageDim + make_float2(0.5f, 0.5f));

    const float2 depth = make_float2(linearZ[batch][0][cy][cx], linearZ[batch][1][cy][cx]);
    const float3 norm = make_float3(normal[batch][0][cy][cx], normal[batch][1][cy][cx], normal[batch][2][cy][cx]);

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
        if (loc.x >= imageDim.x || loc.x < 0 || loc.y >= imageDim.y || loc.y < 0 || prev_opacity[batch][0][loc.y][loc.x] == 0.f)
        {
            v[sampleIdx] = false;
        }
        else
        {
            depthPrev = make_float2(prevLinearZ[batch][0][loc.y][loc.x], prevLinearZ[batch][1][loc.y][loc.x]);
            normalPrev = make_float3(prevNormal[batch][0][loc.y][loc.x], prevNormal[batch][1][loc.y][loc.x], prevNormal[batch][2][loc.y][loc.x]);
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

            if (v[sampleIdx])
            {
                for (int d = 0; d < num_dims; ++d)
                {
                    int dim = input[d].size(1);
                    for (int c = 0; c < dim; ++c)
                        output[d][batch][c][cy][cx] += w[sampleIdx] * input[d][batch][c][loc.y][loc.x];
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
                int dim = input[d].size(1);
                for (size_t c = 0; c < dim; ++c)
                    output[d][batch][c][cy][cx] /= sumw;
            }
        }
        else
        {
            for (int d = 0; d < num_dims; ++d)
            {
                int dim = input[d].size(1);
                for (size_t c = 0; c < dim; ++c)
                    output[d][batch][c][cy][cx] = 0.f;
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
                if (p.x >= imageDim.x || p.x < 0 || p.y >= imageDim.y || p.y < 0 || prev_opacity[batch][0][p.y][p.x] == 0.f)
                {
                    // Outside window
                }
                else
                {
                    // Inside window
                    float2 depthPrev = make_float2(prevLinearZ[batch][0][p.y][p.x], prevLinearZ[batch][1][p.y][p.x]);
                    float3 normalPrev = make_float3(prevNormal[batch][0][p.y][p.x], prevNormal[batch][1][p.y][p.x], prevNormal[batch][2][p.y][p.x]);

                    if (isReprojValid(imageDim, iposPrev, depth.x, depthPrev.x, depth.y, norm, normalPrev, normalFwidth))
                    {
                        for (int d = 0; d < num_dims; ++d)
                        {
                            int dim = input[d].size(1);
                            for (size_t c = 0; c < dim; ++c)
                                output[d][batch][c][cy][cx] += input[d][batch][c][p.y][p.x];
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
                int dim = input[d].size(1);
                for (size_t c = 0; c < dim; ++c)
                    output[d][batch][c][cy][cx] /= nValid;
            }
        }
    }

    if (!valid)
    {
        for (int d = 0; d < num_dims; ++d)
        {
            int dim = input[d].size(1);
            for (size_t c = 0; c < dim; ++c)
                output[d][batch][c][cy][cx] = 0.f;
        }
    }

    success[batch][0][cy][cx] = valid;
}

std::vector<torch::Tensor> reproject_cuda_forward(
    const std::vector<torch::Tensor> _inputList, // [B, ?, H, W]
    const torch::Tensor _mvec,                   // [B, 3, H, W]
    const torch::Tensor _linearZ,                // [B, 3, H, W]
    const torch::Tensor _prevLinearZ,            // [B, 3, H, W]
    const torch::Tensor _normal,                 // [B, 3, H, W]
    const torch::Tensor _prevNormal,             // [B, 3, H, W]
    const torch::Tensor _pnFwidth,               // [B, 3, H, W]
    const torch::Tensor _opacity,                // [B, 1, H, W]
    const torch::Tensor _prevOpacity             // [B, 1, H, W]
)
{
    for (const auto &input : _inputList)
    {
        CHECK_INPUT(input);
    }
    CHECK_INPUT(_mvec);
    CHECK_INPUT(_linearZ);
    CHECK_INPUT(_prevLinearZ);
    CHECK_INPUT(_normal);
    CHECK_INPUT(_prevNormal);
    CHECK_INPUT(_pnFwidth);
    CHECK_INPUT(_opacity);
    CHECK_INPUT(_prevOpacity);

    const int nBatch = _mvec.size(0);
    const int height = _mvec.size(2);
    const int width = _mvec.size(3);
    const int numDims = _inputList.size();

    // Input pointers
    std::vector<TensorAccessor4D> inputListPtr;
    for (auto &input : _inputList)
        inputListPtr.push_back(input.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    auto mvecptr = _mvec.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto linearZptr = _linearZ.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prevLinearZptr = _prevLinearZ.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto normalptr = _normal.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prevNormalptr = _prevNormal.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto pnFwidthptr = _pnFwidth.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto opacityptr = _opacity.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prevOpacityptr = _prevOpacity.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Output tensors and pointers (4D tensors)
    auto outSuccess = torch::zeros({nBatch, 1, height, width}, torch::CUDA(torch::kBool));
    auto outSuccessptr = outSuccess.packed_accessor32<bool, 4, torch::RestrictPtrTraits>();
    std::vector<torch::Tensor> reprojList;
    std::vector<TensorAccessor4D> reprojListPtr;
    for (auto &input : _inputList)
    {
        auto out = torch::zeros_like(input);
        reprojList.push_back(out);
        reprojListPtr.push_back(out.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
    }

    // Device pointers
    thrust::device_vector<TensorAccessor4D> d_inputListPtr;
    thrust::device_vector<TensorAccessor4D> d_reprojListPtr;
    d_inputListPtr = inputListPtr;
    d_reprojListPtr = reprojListPtr;
    TensorAccessor4D *d_ptr = thrust::raw_pointer_cast(d_reprojListPtr.data());

    // Launch kernel
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < nBatch; ++b)
    {
        Reproject<<<grid, threads>>>(
            outSuccessptr,
            thrust::raw_pointer_cast(d_reprojListPtr.data()),
            mvecptr,
            thrust::raw_pointer_cast(d_inputListPtr.data()),
            linearZptr,
            prevLinearZptr,
            normalptr,
            prevNormalptr,
            pnFwidthptr,
            opacityptr,
            prevOpacityptr,
            b, numDims, width, height);
    }

    std::vector<torch::Tensor> outList;
    outList.push_back(outSuccess);
    for (auto &reproj : reprojList)
        outList.push_back(reproj);

    return outList;
}

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
    TensorAccessor4D _output,
    TensorAccessor4D _output1,
    TensorAccessor4D _output2,
    TensorAccessor4D _success,
    const TensorAccessor4D _current1,
    const TensorAccessor4D _current2,
    const TensorAccessor4D _prev,
    const TensorAccessor4D _prev1,
    const TensorAccessor4D _prev2,
    const TensorAccessor4D _reprojSuccess,
    const int batch, const int height, const int width, const int winCurr, const int winPrev)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    float4 current1 = make_float4(_current1[batch][0][cy][cx], _current1[batch][1][cy][cx], _current1[batch][2][cy][cx], 0.f);
    float4 current2 = make_float4(_current2[batch][0][cy][cx], _current2[batch][1][cy][cx], _current2[batch][2][cy][cx], 0.f);
    float4 prev1 = make_float4(_prev1[batch][0][cy][cx], _prev1[batch][1][cy][cx], _prev1[batch][2][cy][cx], 0.f);
    float4 prev2 = make_float4(_prev2[batch][0][cy][cx], _prev2[batch][1][cy][cx], _prev2[batch][2][cy][cx], 0.f);

    float4 current = (current1 + current2) * 0.5f;
    float4 prev = make_float4(_prev[batch][0][cy][cx], _prev[batch][1][cy][cx], _prev[batch][2][cy][cx], 0.f);

    if (_reprojSuccess[batch][0][cy][cx] < 1.f)
    {
        _output[batch][0][cy][cx] = current.x;
        _output[batch][1][cy][cx] = current.y;
        _output[batch][2][cy][cx] = current.z;

        _output1[batch][0][cy][cx] = current1.x;
        _output1[batch][1][cy][cx] = current1.y;
        _output1[batch][2][cy][cx] = current1.z;

        _output2[batch][0][cy][cx] = current2.x;
        _output2[batch][1][cy][cx] = current2.y;
        _output2[batch][2][cy][cx] = current2.z;

        _success[batch][0][cy][cx] = 0.f;
        return;
    }

    // Find min/max of the previous frame inside the window
    float4 minCurr = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f),
           minCurr1 = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f),
           minCurr2 = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f);
    float4 maxCurr = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f),
           maxCurr1 = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f),
           maxCurr2 = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f);
    fo2(cx, cy, width, height, winCurr / 2, 1 /* step size */,
        [&](const int ix, const int iy, const int idx, const int kernelIdx)
        {
            const float4 iCurr1 = make_float4(_current1[batch][0][iy][ix], _current1[batch][1][iy][ix], _current1[batch][2][iy][ix], 0.f);
            const float4 iCurr2 = make_float4(_current2[batch][0][iy][ix], _current2[batch][1][iy][ix], _current2[batch][2][iy][ix], 0.f);
            const float4 iCurr = (iCurr1 + iCurr2) * 0.5f;
            minCurr1 = fmin3f(minCurr1, iCurr1);
            maxCurr1 = fmax3f(maxCurr1, iCurr1);
            minCurr2 = fmin3f(minCurr2, iCurr2);
            maxCurr2 = fmax3f(maxCurr2, iCurr2);
            minCurr = fmin3f(minCurr, iCurr);
            maxCurr = fmax3f(maxCurr, iCurr);
        });

    float4 minPrev = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f),
           minPrev1 = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f),
           minPrev2 = make_float4(FLT_MAX2, FLT_MAX2, FLT_MAX2, 0.f);
    float4 maxPrev = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f),
           maxPrev1 = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f),
           maxPrev2 = make_float4(-FLT_MAX2, -FLT_MAX2, -FLT_MAX2, 0.f);
    fo2(cx, cy, width, height, winPrev / 2, 1 /* step size */,
        [&](const int ix, const int iy, const int idx, const int kernelIdx)
        {
            if (_reprojSuccess[batch][0][iy][ix] + 1e-4f < 1.f)
                return;

            const float4 iPrev = make_float4(_prev[batch][0][iy][ix], _prev[batch][1][iy][ix], _prev[batch][2][iy][ix], 0.f);
            const float4 iPrev1 = make_float4(_prev1[batch][0][iy][ix], _prev1[batch][1][iy][ix], _prev1[batch][2][iy][ix], 0.f);
            const float4 iPrev2 = make_float4(_prev2[batch][0][iy][ix], _prev2[batch][1][iy][ix], _prev2[batch][2][iy][ix], 0.f);
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

    _output[batch][0][cy][cx] = output.x;
    _output[batch][1][cy][cx] = output.y;
    _output[batch][2][cy][cx] = output.z;

    _output1[batch][0][cy][cx] = output1.x;
    _output1[batch][1][cy][cx] = output1.y;
    _output1[batch][2][cy][cx] = output1.z;

    _output2[batch][0][cy][cx] = output2.x;
    _output2[batch][1][cy][cx] = output2.y;
    _output2[batch][2][cy][cx] = output2.z;

    _success[batch][0][cy][cx] = success ? 1.f : 0.f;
}

std::vector<torch::Tensor> temporal_screening_cuda_forward(
    const torch::Tensor _current1,
    const torch::Tensor _current2,
    const torch::Tensor _prev,
    const torch::Tensor _prev1,
    const torch::Tensor _prev2,
    const torch::Tensor _reprojSuccess,
    const int winCurr, const int winPrev)
{
    CHECK_INPUT(_current1);
    CHECK_INPUT(_current2);
    CHECK_INPUT(_prev);
    CHECK_INPUT(_prev1);
    CHECK_INPUT(_prev2);
    CHECK_INPUT(_reprojSuccess);

    const int nBatch = _current1.size(0);
    const int height = _current1.size(2);
    const int width = _current1.size(3);

    // Input pointers
    auto current1ptr = _current1.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto current2ptr = _current2.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prevptr = _prev.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prev1ptr = _prev1.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto prev2ptr = _prev2.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto reprojSuccessptr = _reprojSuccess.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Output tensors and pointers
    torch::Tensor outSuccess = torch::zeros_like(_reprojSuccess);
    torch::Tensor out = torch::zeros_like(_current1);
    torch::Tensor out1 = torch::zeros_like(_current1);
    torch::Tensor out2 = torch::zeros_like(_current1);
    auto outSuccessptr = outSuccess.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outptr = out.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto out1ptr = out1.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto out2ptr = out2.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Launch kernel
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < nBatch; ++b)
    {
        TemporalScreening<<<grid, threads>>>(
            outptr,
            out1ptr,
            out2ptr,
            outSuccessptr,
            current1ptr,
            current2ptr,
            prevptr,
            prev1ptr,
            prev2ptr,
            reprojSuccessptr,
            b, height, width, winCurr, winPrev);
    }

    return {out, out1, out2, outSuccess};
}

__global__ void Erosion2D(
    TensorAccessor4D _output,
    const TensorAccessor4D _input,
    const int batch, const int height, const int width, const int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    float minVal = FLT_MAX2;
    fo2(cx, cy, width, height, winSize / 2, 1 /* step size */,
        [&](const int ix, const int iy, const int idx, const int kernelIdx)
        {
            const float iVal = _input[batch][0][iy][ix];
            minVal = fminf(minVal, iVal);
        });

    _output[batch][0][cy][cx] = minVal;
}

torch::Tensor erosion2d_cuda_forward(
    const torch::Tensor _input,
    const int winSize)
{
    CHECK_INPUT(_input);

    const int nBatch = _input.size(0);
    const int channels = _input.size(1);
    const int height = _input.size(2);
    const int width = _input.size(3);

    assert(channels == 1 && "Input tensor must have only 1 channel");

    auto inputptr = _input.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    auto out = torch::zeros_like(_input);
    auto outptr = out.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < nBatch; ++b)
    {
        Erosion2D<<<grid, threads>>>(
            outptr,
            inputptr,
            b, height, width, winSize);
    }

    return out;
}