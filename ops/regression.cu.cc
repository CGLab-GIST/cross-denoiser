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
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define COEFFI_DIM_PILOT 6
#define BAND_RANK 4

__global__ void SpatiotemporalFilterGradKernel(float *_outGradBand, const float *_gradAccImgA, const float *_gradAccImgB,
                                               const float *_gradWgtSumA, const float *_gradWgtSumB,
                                               const float4 *_gradBandA, const float4 *_gradBandB,
                                               int height, int width, int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;

    const float4 &cGradAccImgA = make_float4(_gradAccImgA[cIdx * 3 + 0], _gradAccImgA[cIdx * 3 + 1], _gradAccImgA[cIdx * 3 + 2], 0.f);
    const float4 &cGradAccImgB = make_float4(_gradAccImgB[cIdx * 3 + 0], _gradAccImgB[cIdx * 3 + 1], _gradAccImgB[cIdx * 3 + 2], 0.f);
    const float &cGradWgtSumA = _gradWgtSumA[cIdx];
    const float &cGradWgtSumB = _gradWgtSumB[cIdx];

    float4 cGradBandA[BAND_RANK], cGradBandB[BAND_RANK];
    for (int i = 0; i < BAND_RANK; ++i)
    {
        cGradBandA[i] = _gradBandA[cIdx * BAND_RANK + i];
        cGradBandB[i] = _gradBandB[cIdx * BAND_RANK + i];
    }

    float outGradBandA[BAND_RANK] = {
        0.f,
    };
    float outGradBandB[BAND_RANK] = {
        0.f,
    };
    for (int i = 0; i < BAND_RANK; ++i)
    {
        outGradBandA[i] = cGradAccImgA.x * cGradBandA[i].x +
                          cGradAccImgA.y * cGradBandA[i].y +
                          cGradAccImgA.z * cGradBandA[i].z +
                          cGradWgtSumA * cGradBandA[i].w;
        outGradBandB[i] = cGradAccImgB.x * cGradBandB[i].x +
                          cGradAccImgB.y * cGradBandB[i].y +
                          cGradAccImgB.z * cGradBandB[i].z +
                          cGradWgtSumB * cGradBandB[i].w;
    }

    _outGradBand[cIdx * (BAND_RANK + 1) + 0] = outGradBandA[0];
    _outGradBand[cIdx * (BAND_RANK + 1) + 1] = outGradBandB[0];
    _outGradBand[cIdx * (BAND_RANK + 1) + 2] = (outGradBandA[1] + outGradBandB[1]);
    _outGradBand[cIdx * (BAND_RANK + 1) + 3] = (outGradBandA[2] + outGradBandB[2]);
    _outGradBand[cIdx * (BAND_RANK + 1) + 4] = (outGradBandA[3] + outGradBandB[3]);
}

// Note: _gradBand for the backprop later
__global__ void SpatiotemporalFilterKernel(float *_outAccImgA, float *_outAccImgB, float *_outWgtSumA, float *_outWgtSumB,
                                           float4 *_gradBandA, float4 *_gradBandB,
                                           const float *_imgA, const float *_imgB,
                                           const float *_albedo, const float *_normal, const float *_band, int height, int width, int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;
    const int halfWinSize = winSize / 2;
    int sx = cx - halfWinSize;
    int ex = cx + halfWinSize;
    int sy = cy - halfWinSize;
    int ey = cy + halfWinSize;

    float cBand[BAND_RANK + 1];
    for (int i = 0; i < BAND_RANK + 1; ++i)
        cBand[i] = _band[cIdx * (BAND_RANK + 1) + i];

    float deriWgtA[BAND_RANK], deriWgtB[BAND_RANK];
    float4 gradBandA[BAND_RANK], gradBandB[BAND_RANK];
    for (int i = 0; i < BAND_RANK; ++i)
    {
        gradBandA[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        gradBandB[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    const float4 &cColA = make_float4(_imgA[cIdx * 3 + 0], _imgA[cIdx * 3 + 1], _imgA[cIdx * 3 + 2], 0.f);
    const float4 &cColB = make_float4(_imgB[cIdx * 3 + 0], _imgB[cIdx * 3 + 1], _imgB[cIdx * 3 + 2], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
    const float4 &cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);

    float4 accColA = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 accColB = make_float4(0.f, 0.f, 0.f, 0.f);
    float sumwA = 0.f;
    float sumwB = 0.f;

    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
            int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
            int idx = y * width + x;

            const float4 &iColA = make_float4(_imgA[idx * 3 + 0], _imgA[idx * 3 + 1], _imgA[idx * 3 + 2], 0.f);
            const float4 &iColB = make_float4(_imgB[idx * 3 + 0], _imgB[idx * 3 + 1], _imgB[idx * 3 + 2], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
            const float4 &iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);

            float dy = (float)(cy - iy) / halfWinSize;
            float dx = (float)(cx - ix) / halfWinSize;
            float pixDist = dx * dx + dy * dy;

            float distColA = norm2(cColA - iColA);
            float distColB = norm2(cColB - iColB);

            float distAlbedo = norm2(cAlbedo - iAlbedo);
            float distNorm = norm2(cNormal - iNormal);

            float distA = -cBand[0] * distColA - cBand[2] * distAlbedo - cBand[3] * distNorm - cBand[4] * pixDist;
            float distB = -cBand[1] * distColB - cBand[2] * distAlbedo - cBand[3] * distNorm - cBand[4] * pixDist;
            float wgtA = __expf(distA);
            float wgtB = __expf(distB);

            accColA += wgtA * iColA;
            accColB += wgtB * iColB;
            sumwA += wgtA;
            sumwB += wgtB;

            // for backprop later
            deriWgtA[0] = wgtA * (-distColA);
            deriWgtA[1] = wgtA * (-distAlbedo);
            deriWgtA[2] = wgtA * (-distNorm);
            deriWgtA[3] = wgtA * (-pixDist);
            deriWgtB[0] = wgtB * (-distColB);
            deriWgtB[1] = wgtB * (-distAlbedo);
            deriWgtB[2] = wgtB * (-distNorm);
            deriWgtB[3] = wgtB * (-pixDist);

            for (int i = 0; i < BAND_RANK; ++i)
            {
                gradBandA[i].x += deriWgtA[i] * iColA.x;
                gradBandA[i].y += deriWgtA[i] * iColA.y;
                gradBandA[i].z += deriWgtA[i] * iColA.z;
                gradBandA[i].w += deriWgtA[i];
                gradBandB[i].x += deriWgtB[i] * iColB.x;
                gradBandB[i].y += deriWgtB[i] * iColB.y;
                gradBandB[i].z += deriWgtB[i] * iColB.z;
                gradBandB[i].w += deriWgtB[i];
            }
        }
    }
    _outAccImgA[cIdx * 3 + 0] = accColA.x;
    _outAccImgA[cIdx * 3 + 1] = accColA.y;
    _outAccImgA[cIdx * 3 + 2] = accColA.z;
    _outWgtSumA[cIdx] = sumwA;

    _outAccImgB[cIdx * 3 + 0] = accColB.x;
    _outAccImgB[cIdx * 3 + 1] = accColB.y;
    _outAccImgB[cIdx * 3 + 2] = accColB.z;
    _outWgtSumB[cIdx] = sumwB;

    for (int i = 0; i < BAND_RANK; ++i)
    {
        _gradBandA[cIdx * BAND_RANK + i] = gradBandA[i];
        _gradBandB[cIdx * BAND_RANK + i] = gradBandB[i];
    }
}

__device__ float4 SQRT(const float4 &c)
{
    float4 ret;
    ret.x = sqrtf(max(1e-5f, c.x));
    ret.y = sqrtf(max(1e-5f, c.y));
    ret.z = sqrtf(max(1e-5f, c.z));
    return ret;
}

__device__ void cholesky_color(const float4 *A, float4 *L, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < (i + 1); ++j)
        {
            float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int k = 0; k < j; ++k)
                s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (i == j) ? SQRT(A[i * n + i] - s) : (make_float4(1.f, 1.f, 1.f, 1.f) / L[j * n + j] * (A[j * n + i] - s));
        }
    }
}

__device__ void forward_backward_solver(float4 *beta, const float4 *XtB, const float4 *L, const int n)
{
    float4 W[COEFFI_DIM_PILOT];

    // Forward substitution
    W[0] = XtB[0] / L[0 * n + 0];
    for (int i = 1; i < n; ++i)
    {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < i; ++k)
            s += L[i * n + k] * W[k];
        W[i] = (XtB[i] - s) / L[i * n + i];
    }
    // Backward substituation
    beta[n - 1] = W[n - 1] / L[(n - 1) * n + (n - 1)];
    for (int i = n - 2; i >= 0; --i)
    {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = n - 1; k > i; --k)
            s += L[k * n + i] * beta[k];
        beta[i] = (W[i] - s) / L[i * n + i];
    }
}

__global__ void EstInputVarianceKernel(const float *_rand, float *_outVar, int height, int width, int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int halfWinSize = winSize / 2;
    int sx = MAX(0, cx - halfWinSize);
    int ex = MIN(width - 1, cx + halfWinSize);
    int sy = MAX(0, cy - halfWinSize);
    int ey = MIN(height - 1, cy + halfWinSize);
    int cIdx = cy * width + cx;
    int numPixels = 0;
    float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idx = iy * width + ix;
            const float4 &iCol = make_float4(_rand[idx * 3 + 0], _rand[idx * 3 + 1], _rand[idx * 3 + 2], 0.f);
            if (idx != cIdx)
            {
                accCol += iCol;
                ++numPixels;
            }
        }
    }
    accCol = accCol / (float)numPixels;
    const float4 &cCol = make_float4(_rand[cIdx * 3 + 0], _rand[cIdx * 3 + 1], _rand[cIdx * 3 + 2], 0.f);
    _outVar[cy * width + cx] = norm2(accCol - cCol);
}

__device__ float regressionWgtDist(const float4 &cCol, const float4 &iCol, const float cVar, const float iVar)
{
    return (norm2(iCol - cCol)) / ((cVar + iVar) + 1e-4f);
}

__device__ float regressionWgt(const float4 &cCol, const float4 &iCol, const float cVar, const float iVar)
{
    float dist = regressionWgtDist(cCol, iCol, cVar, iVar);
    return __expf(-dist);
}

__global__ void BlockRecon(float *_outImg, const float4 *_beta, const float *_inImg, const float *_wgtImg,
                           const float *_varWgt, const float *_albedo, const float *_normal,
                           int height, int width, int winSize, int sparsity, int reconScale)
{

    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    const int widthRecon = width / reconScale;
    const int heightRecon = height / reconScale;

    if (cx >= widthRecon || cy >= heightRecon)
        return;

    const int cxOrig = cx * reconScale;
    const int cyOrig = cy * reconScale;

    const int P = COEFFI_DIM_PILOT;
    const int cIdxOrig = cyOrig * width + cxOrig;
    const int cIdxRecon = cy * widthRecon + cx;
    const int halfWinSize = winSize / 2;
    const int widthBeta = width / sparsity;

    int sx = max(0, iDivUp(cxOrig - halfWinSize, sparsity));
    int ex = min((width - 1) / sparsity, (cxOrig + halfWinSize) / sparsity);
    int sy = max(0, iDivUp(cyOrig - halfWinSize, sparsity));
    int ey = min((height - 1) / sparsity, (cyOrig + halfWinSize) / sparsity);

    const float4 &cCol = make_float4(_inImg[cIdxOrig * 3 + 0], _inImg[cIdxOrig * 3 + 1], _inImg[cIdxOrig * 3 + 2], 0.f);
    const float4 &cColWgt = make_float4(_wgtImg[cIdxOrig * 3 + 0], _wgtImg[cIdxOrig * 3 + 1], _wgtImg[cIdxOrig * 3 + 2], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[cIdxOrig * 3 + 0], _albedo[cIdxOrig * 3 + 1], _albedo[cIdxOrig * 3 + 2], 0.f);
    const float4 &cNormal = make_float4(_normal[cIdxOrig * 3 + 0], _normal[cIdxOrig * 3 + 1], _normal[cIdxOrig * 3 + 2], 0.f);
    const float &cVar = _varWgt[cIdxOrig];
    const float cStd = sqrtf(cVar);

    float4 delta[P];
    float4 outCol = make_float4(0.f, 0.f, 0.f, 0.f);
    float sumW = 0.f;
    delta[0] = make_float4(1.f, 1.f, 1.f, 0.f);

    // Iterate in domain of beta
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int idxBeta = iy * widthBeta + ix;
            int idx = (iy * sparsity) * width + (ix * sparsity);
            const float4 &iColWgt = make_float4(_wgtImg[idx * 3 + 0], _wgtImg[idx * 3 + 1], _wgtImg[idx * 3 + 2], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
            const float4 &iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
            const float &iVar = _varWgt[idx];
            const float iStd = sqrtf(iVar);

            // Warning! the order should be oppsoite (e.g., cColWgt - iColWgt)
            delta[1] = (cColWgt - iColWgt) / ((cStd) + (iStd) + 1e-2f);
            delta[2] = (cAlbedo - iAlbedo);
            float4 dNormal = cNormal - iNormal;
            delta[3] = make_float4(dNormal.x, dNormal.x, dNormal.x, 0.f);
            delta[4] = make_float4(dNormal.y, dNormal.y, dNormal.y, 0.f);
            delta[5] = make_float4(dNormal.z, dNormal.z, dNormal.z, 0.f);

            float4 predCol = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int i = 0; i < P; ++i)
            {
                predCol += _beta[idxBeta * P + i] * delta[i];
            }

            float weight = regressionWgt(cColWgt, iColWgt, cVar, iVar) + 1e-7f;
            outCol += weight * predCol;
            sumW += weight;
        }
    }

    _outImg[cIdxRecon * 3 + 0] = fmax(0.f, outCol.x / fmaxf(sumW, 1e-7f));
    _outImg[cIdxRecon * 3 + 1] = fmax(0.f, outCol.y / fmaxf(sumW, 1e-7f));
    _outImg[cIdxRecon * 3 + 2] = fmax(0.f, outCol.z / fmaxf(sumW, 1e-7f));
}

__global__ void RegressionKernel(float4 *_beta,
                                 const float *_inImg, const float *_wgtImg,
                                 const float *_varWgt, const float *_albedo, const float *_normal,
                                 int height, int width, int winSize, int sparsity)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x) * sparsity;
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) * sparsity;

    if (cx >= width || cy >= height)
        return;

    const int P = COEFFI_DIM_PILOT;

    const int cIdx = cy * width + cx;
    const int halfWinSize = winSize / 2;

    float4 delta[P];
    float4 A[P * P] = {
        0.f,
    };
    float4 XtB[P];
    for (int i = 0; i < P; ++i)
    {
        XtB[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int j = 0; j < P; ++j)
            A[i * P + j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    delta[0] = make_float4(1.f, 1.f, 1.f, 1.f);

    int sx = cx - halfWinSize;
    int ex = cx + halfWinSize;
    int sy = cy - halfWinSize;
    int ey = cy + halfWinSize;

    const float4 &cCol = make_float4(_inImg[cIdx * 3 + 0], _inImg[cIdx * 3 + 1], _inImg[cIdx * 3 + 2], 0.f);
    const float4 &cColWgt = make_float4(_wgtImg[cIdx * 3 + 0], _wgtImg[cIdx * 3 + 1], _wgtImg[cIdx * 3 + 2], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[cIdx * 3 + 0], _albedo[cIdx * 3 + 1], _albedo[cIdx * 3 + 2], 0.f);
    const float4 &cNormal = make_float4(_normal[cIdx * 3 + 0], _normal[cIdx * 3 + 1], _normal[cIdx * 3 + 2], 0.f);
    const float cVar = _varWgt[cIdx];
    const float cStd = sqrtf(cVar);

    float sumW = 0.f;
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
            int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);

            int idx = y * width + x;

            const float4 &iCol = make_float4(_inImg[idx * 3 + 0], _inImg[idx * 3 + 1], _inImg[idx * 3 + 2], 0.f);
            const float4 &iColWgt = make_float4(_wgtImg[idx * 3 + 0], _wgtImg[idx * 3 + 1], _wgtImg[idx * 3 + 2], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[idx * 3 + 0], _albedo[idx * 3 + 1], _albedo[idx * 3 + 2], 0.f);
            const float4 &iNormal = make_float4(_normal[idx * 3 + 0], _normal[idx * 3 + 1], _normal[idx * 3 + 2], 0.f);
            const float iVar = _varWgt[idx];
            const float iStd = sqrtf(iVar);

            delta[1] = (iColWgt - cColWgt) / ((cStd) + (iStd) + 1e-2f);
            delta[2] = (iAlbedo - cAlbedo);
            float4 dNormal = iNormal - cNormal;
            delta[3] = make_float4(dNormal.x, dNormal.x, dNormal.x, 0.f);
            delta[4] = make_float4(dNormal.y, dNormal.y, dNormal.y, 0.f);
            delta[5] = make_float4(dNormal.z, dNormal.z, dNormal.z, 0.f);

            float weight = regressionWgt(cColWgt, iColWgt, cVar, iVar) + 1e-7f;
            sumW += weight;

            for (int row = 0; row < P; ++row)
            {
                for (int col = row; col < P; ++col)
                {
                    A[row * P + col] += weight * delta[row] * delta[col];
                }
            }
            for (int row = 0; row < P; ++row)
            {
                XtB[row] += weight * delta[row] * iCol;
            }
        }
    }

    for (int row = 0; row < P; ++row)
    {
        for (int col = 0; col < row; ++col)
        {
            A[row * P + col] = A[col * P + row];
        }
    }

    // TODO. epsilon should be adjusted correctly
    for (int row = 0; row < P; ++row)
    {
        A[row * P + row] += make_float4(1e-3f, 1e-3f, 1e-3f, 1e-3f);
    }

    float4 beta[P];
    float4 L[P * P];
    cholesky_color(A, L, P);

    forward_backward_solver(beta, XtB, L, P);

    // storing coefficients
    int cy_coeff = cy / sparsity;
    int cx_coeff = cx / sparsity;
    int cIdx_coeff = cy_coeff * (width / sparsity) + cx_coeff;
    for (int i = 0; i < P; ++i)
        _beta[cIdx_coeff * P + i] = beta[i];
}

void CrossRegressionFunc(const GPUDevice &_dev,
                         // Input
                         const float *_imga, const float *_imgb, const float *_albedo, const float *_normal,
                         // Output
                         float *_outImgA, float *_outImgB, float *_varA, float *_varB,
                         // Intermediate
                         float4 *_betaA, float4 *_betaB,
                         int nBatch, int height, int width, int winSize, int sparsity, int reconScale)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    EstInputVarianceKernel<<<grid, threads, 0, _dev.stream()>>>(_imga, _varA, height, width, 3);
    EstInputVarianceKernel<<<grid, threads, 0, _dev.stream()>>>(_imgb, _varB, height, width, 3);

    dim3 gridSparse(iDivUp(width / sparsity, blockDim), iDivUp(height / sparsity, blockDim));
    RegressionKernel<<<gridSparse, threads, 0, _dev.stream()>>>(_betaA, _imgb, _imga, _varA, _albedo, _normal, height, width, winSize, sparsity);
    RegressionKernel<<<gridSparse, threads, 0, _dev.stream()>>>(_betaB, _imga, _imgb, _varB, _albedo, _normal, height, width, winSize, sparsity);

    dim3 gridRecon(iDivUp(width / reconScale, blockDim), iDivUp(height / reconScale, blockDim));
    BlockRecon<<<gridRecon, threads, 0, _dev.stream()>>>(_outImgA, _betaA, _imgb, _imga, _varA, _albedo, _normal, height, width, winSize, sparsity, reconScale);
    BlockRecon<<<gridRecon, threads, 0, _dev.stream()>>>(_outImgB, _betaB, _imga, _imgb, _varB, _albedo, _normal, height, width, winSize, sparsity, reconScale);
}

void SpatiotemporalFilterFunc(const GPUDevice &_dev,
                              // Input
                              const float *_imga, const float *_imgb, const float *_albedo, const float *_normal, const float *_band,
                              // Output
                              float *_outAccImgA, float *_outAccImgB, float *_outWgtSumA, float *_outWgtSumB,
                              // Intermediate
                              float4 *_gradBandA, float4 *_gradBandB,
                              int nBatch, int height, int width, int winSize)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    SpatiotemporalFilterKernel<<<grid, threads, 0, _dev.stream()>>>(_outAccImgA, _outAccImgB, _outWgtSumA, _outWgtSumB, _gradBandA, _gradBandB,
                                                                    _imga, _imgb, _albedo, _normal, _band, height, width, winSize);
}

void SpatiotemporalFilterGradFunc(const GPUDevice &_dev, const float *_gradAccImgA, const float *_gradAccImgB,
                                  const float *_gradWgtSumA, const float *_gradWgtSumB,
                                  const float *_imga, const float *_imgb,
                                  const float *_albedo, const float *_normal, const float *_band,
                                  float *_outGradBand,
                                  const float4 *_gradBandA, const float4 *_gradBandB,
                                  int nBatch, int height, int width, int winSize)
{
    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    SpatiotemporalFilterGradKernel<<<grid, threads, 0, _dev.stream()>>>(_outGradBand, _gradAccImgA, _gradAccImgB, _gradWgtSumA, _gradWgtSumB,
                                                                        _gradBandA, _gradBandB,
                                                                        height, width, winSize);
}

#endif // GOOGLE_CUDA
