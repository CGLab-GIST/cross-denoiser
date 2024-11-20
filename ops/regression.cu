#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

#define COEFFI_DIM_PILOT 6
#define BAND_RANK 4

// Note: _gradBand for the backprop later
__global__ void SpatialFilterKernel(
    TensorAccessor4D _outAccImgA,
    TensorAccessor4D _outAccImgB,
    TensorAccessor4D _outWgtSumA,
    TensorAccessor4D _outWgtSumB,
    float4 *_gradBandA,
    float4 *_gradBandB,
    const TensorAccessor4D _imgA,
    const TensorAccessor4D _imgB,
    const TensorAccessor4D _albedo,
    const TensorAccessor4D _normal,
    const TensorAccessor4D _band,
    const int batch, const int height, const int width, const int winSize)
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
        cBand[i] = _band[batch][i][cy][cx];

    float deriWgtA[BAND_RANK], deriWgtB[BAND_RANK];
    float4 gradBandA[BAND_RANK], gradBandB[BAND_RANK];
    for (int i = 0; i < BAND_RANK; ++i)
    {
        gradBandA[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        gradBandB[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    const float4 &cColA = make_float4(_imgA[batch][0][cy][cx], _imgA[batch][1][cy][cx], _imgA[batch][2][cy][cx], 0.f);
    const float4 &cColB = make_float4(_imgB[batch][0][cy][cx], _imgB[batch][1][cy][cx], _imgB[batch][2][cy][cx], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[batch][0][cy][cx], _albedo[batch][1][cy][cx], _albedo[batch][2][cy][cx], 0.f);
    const float4 &cNormal = make_float4(_normal[batch][0][cy][cx], _normal[batch][1][cy][cx], _normal[batch][2][cy][cx], 0.f);

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

            const float4 &iColA = make_float4(_imgA[batch][0][y][x], _imgA[batch][1][y][x], _imgA[batch][2][y][x], 0.f);
            const float4 &iColB = make_float4(_imgB[batch][0][y][x], _imgB[batch][1][y][x], _imgB[batch][2][y][x], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[batch][0][y][x], _albedo[batch][1][y][x], _albedo[batch][2][y][x], 0.f);
            const float4 &iNormal = make_float4(_normal[batch][0][y][x], _normal[batch][1][y][x], _normal[batch][2][y][x], 0.f);

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
    _outAccImgA[batch][0][cy][cx] = accColA.x;
    _outAccImgA[batch][1][cy][cx] = accColA.y;
    _outAccImgA[batch][2][cy][cx] = accColA.z;
    _outWgtSumA[batch][0][cy][cx] = sumwA;

    _outAccImgB[batch][0][cy][cx] = accColB.x;
    _outAccImgB[batch][1][cy][cx] = accColB.y;
    _outAccImgB[batch][2][cy][cx] = accColB.z;
    _outWgtSumB[batch][0][cy][cx] = sumwB;

    for (int i = 0; i < BAND_RANK; ++i)
    {
        _gradBandA[cIdx * BAND_RANK + i] = gradBandA[i];
        _gradBandB[cIdx * BAND_RANK + i] = gradBandB[i];
    }
}

__global__ void SpatialFilterGradKernel(
    TensorAccessor4D _outGradBand,
    const TensorAccessor4D _gradAccImgA,
    const TensorAccessor4D _gradAccImgB,
    const TensorAccessor4D _gradWgtSumA,
    const TensorAccessor4D _gradWgtSumB,
    const float4 *_gradBandA,
    const float4 *_gradBandB,
    const int batch, const int height, const int width, const int winSize)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if (cx >= width || cy >= height)
        return;

    const int cIdx = cy * width + cx;

    const float4 &cGradAccImgA = make_float4(_gradAccImgA[batch][0][cy][cx], _gradAccImgA[batch][1][cy][cx], _gradAccImgA[batch][2][cy][cx], 0.f);
    const float4 &cGradAccImgB = make_float4(_gradAccImgB[batch][0][cy][cx], _gradAccImgB[batch][1][cy][cx], _gradAccImgB[batch][2][cy][cx], 0.f);
    const float &cGradWgtSumA = _gradWgtSumA[batch][0][cy][cx];
    const float &cGradWgtSumB = _gradWgtSumB[batch][0][cy][cx];

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

    _outGradBand[batch][0][cy][cx] = outGradBandA[0];
    _outGradBand[batch][1][cy][cx] = outGradBandB[0];
    _outGradBand[batch][2][cy][cx] = (outGradBandA[1] + outGradBandB[1]);
    _outGradBand[batch][3][cy][cx] = (outGradBandA[2] + outGradBandB[2]);
    _outGradBand[batch][4][cy][cx] = (outGradBandA[3] + outGradBandB[3]);
}

std::vector<torch::Tensor> spatial_filter_cuda_forward(
    const torch::Tensor _imgA,   // [B, 3, H, W]
    const torch::Tensor _imgB,   // [B, 3, H, W]
    const torch::Tensor _albedo, // [B, 3, H, W]
    const torch::Tensor _normal, // [B, 3, H, W]
    const torch::Tensor _band,   // [B, BAND_RANK + 1, H, W]
    const int winSize)
{
    CHECK_INPUT(_imgA);
    CHECK_INPUT(_imgB);
    CHECK_INPUT(_albedo);
    CHECK_INPUT(_normal);
    CHECK_INPUT(_band);

    const int nBatch = _imgA.size(0);
    const int height = _imgA.size(2);
    const int width = _imgA.size(3);

    // Input pointers
    auto imgAptr = _imgA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto imgBptr = _imgB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto albedoptr = _albedo.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto normalptr = _normal.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto bandptr = _band.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Output tensors and pointers (4D tensors)
    torch::Tensor outAccA = torch::zeros({nBatch, 3, height, width}, torch::CUDA(torch::kFloat32));
    torch::Tensor outAccB = torch::zeros({nBatch, 3, height, width}, torch::CUDA(torch::kFloat32));
    torch::Tensor outWgtSumA = torch::zeros({nBatch, 1, height, width}, torch::CUDA(torch::kFloat32));
    torch::Tensor outWgtSumB = torch::zeros({nBatch, 1, height, width}, torch::CUDA(torch::kFloat32));
    auto outAccAptr = outAccA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outAccBptr = outAccB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outWgtSumAptr = outWgtSumA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outWgtSumBptr = outWgtSumB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Intermediate tensors (for backward)
    torch::Tensor gradBandA = torch::zeros({nBatch, height, width, BAND_RANK * 4}, torch::CUDA(torch::kFloat32)); // [B, W, H, BAND_RANK * 4]
    torch::Tensor gradBandB = torch::zeros({nBatch, height, width, BAND_RANK * 4}, torch::CUDA(torch::kFloat32)); // [B, W, H, BAND_RANK * 4]
    auto gradBandAptr = reinterpret_cast<float4 *>(gradBandA.data_ptr<float>());
    auto gradBandBptr = reinterpret_cast<float4 *>(gradBandB.data_ptr<float>());

    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < nBatch; b++)
    {
        SpatialFilterKernel<<<grid, threads>>>(
            outAccAptr, outAccBptr, outWgtSumAptr, outWgtSumBptr,
            gradBandAptr, gradBandBptr,
            imgAptr, imgBptr, albedoptr, normalptr, bandptr,
            b, height, width, winSize);
    }

    return {outAccA, outAccB, outWgtSumA, outWgtSumB, gradBandA, gradBandB};
}

torch::Tensor spatial_filter_cuda_backward(
    torch::Tensor _gradAccImgA, // [B, 3, H, W]
    torch::Tensor _gradAccImgB, // [B, 3, H, W]
    torch::Tensor _gradWgtSumA, // [B, 1, H, W]
    torch::Tensor _gradWgtSumB, // [B, 1, H, W]
    torch::Tensor _band,        // [B, BAND_RANK + 1, H, W]
    torch::Tensor _gradBandA,   // [B, W, H, BAND_RANK * 4]
    torch::Tensor _gradBandB,   // [B, W, H, BAND_RANK * 4]
    const int winSize)
{
    CHECK_INPUT(_gradAccImgA);
    CHECK_INPUT(_gradAccImgB);
    CHECK_INPUT(_gradWgtSumA);
    CHECK_INPUT(_gradWgtSumB);
    CHECK_INPUT(_band);
    CHECK_INPUT(_gradBandA);
    CHECK_INPUT(_gradBandB);

    const int nBatch = _gradAccImgA.size(0);
    const int height = _gradAccImgA.size(2);
    const int width = _gradAccImgA.size(3);

    // Input pointers
    auto gradAccImgAptr = _gradAccImgA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto gradAccImgBptr = _gradAccImgB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto gradWgtSumAptr = _gradWgtSumA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto gradWgtSumBptr = _gradWgtSumB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto gradBandAptr = reinterpret_cast<float4 *>(_gradBandA.data_ptr<float>());
    auto gradBandBptr = reinterpret_cast<float4 *>(_gradBandB.data_ptr<float>());

    // Output tensors and pointers (4D tensors)
    torch::Tensor outGradBand = torch::zeros_like(_band); // [B, W, H, BAND_RANK]
    auto outGradBandptr = outGradBand.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

    for (int b = 0; b < nBatch; b++)
    {
        SpatialFilterGradKernel<<<grid, threads>>>(
            outGradBandptr,
            gradAccImgAptr, gradAccImgBptr, gradWgtSumAptr, gradWgtSumBptr,
            gradBandAptr, gradBandBptr,
            b, height, width, winSize);
    }

    return outGradBand;
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

__global__ void EstInputVarianceKernel(
    const TensorAccessor4D _rand, // [B, 3, H, W]
    TensorAccessor4D _outVar,     // [B, 1, H, W]
    const int batch, const int width, const int height, const int winSize)
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
    int numPixels = 0;
    float4 accCol = ZERO4;
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            const float4 &iCol = make_float4(_rand[batch][0][iy][ix], _rand[batch][1][iy][ix], _rand[batch][2][iy][ix], 0.f);
            if (ix != cx || iy != cy)
            {
                accCol += iCol;
                ++numPixels;
            }
        }
    }
    accCol = accCol / (float)numPixels;
    const float4 &cColor = make_float4(_rand[batch][0][cy][cx], _rand[batch][1][cy][cx], _rand[batch][2][cy][cx], 0.f);
    _outVar[batch][0][cy][cx] = norm2(cColor - accCol);
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

__global__ void BlockRecon(
    TensorAccessor4D _outImg,
    const float4 *_beta,
    const TensorAccessor4D _inImg,
    const TensorAccessor4D _wgtImg,
    const TensorAccessor4D _varWgt,
    const TensorAccessor4D _albedo,
    const TensorAccessor4D _normal,
    const int batch, const int height, const int width, const int winSize, const int sparsity, const int reconScale)
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
    const int halfWinSize = winSize / 2;
    const int widthBeta = width / sparsity;

    int sx = max(0, iDivUp(cxOrig - halfWinSize, sparsity));
    int ex = min((width - 1) / sparsity, (cxOrig + halfWinSize) / sparsity);
    int sy = max(0, iDivUp(cyOrig - halfWinSize, sparsity));
    int ey = min((height - 1) / sparsity, (cyOrig + halfWinSize) / sparsity);

    const float4 &cCol = make_float4(_inImg[batch][0][cy][cx], _inImg[batch][1][cy][cx], _inImg[batch][2][cy][cx], 0.f);
    const float4 &cColWgt = make_float4(_wgtImg[batch][0][cy][cx], _wgtImg[batch][1][cy][cx], _wgtImg[batch][2][cy][cx], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[batch][0][cy][cx], _albedo[batch][1][cy][cx], _albedo[batch][2][cy][cx], 0.f);
    const float4 &cNormal = make_float4(_normal[batch][0][cy][cx], _normal[batch][1][cy][cx], _normal[batch][2][cy][cx], 0.f);
    const float &cVar = _varWgt[0][0][cy][cx];
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
            int iyL = iy * sparsity;
            int ixL = ix * sparsity;
            const float4 &iColWgt = make_float4(_wgtImg[batch][0][iyL][ixL], _wgtImg[batch][1][iyL][ixL], _wgtImg[batch][2][iyL][ixL], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[batch][0][iyL][ixL], _albedo[batch][1][iyL][ixL], _albedo[batch][2][iyL][ixL], 0.f);
            const float4 &iNormal = make_float4(_normal[batch][0][iyL][ixL], _normal[batch][1][iyL][ixL], _normal[batch][2][iyL][ixL], 0.f);
            const float &iVar = _varWgt[batch][0][iyL][ixL];
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

    _outImg[batch][0][cy][cx] = fmax(0.f, outCol.x / fmaxf(sumW, 1e-7f));
    _outImg[batch][1][cy][cx] = fmax(0.f, outCol.y / fmaxf(sumW, 1e-7f));
    _outImg[batch][2][cy][cx] = fmax(0.f, outCol.z / fmaxf(sumW, 1e-7f));
}

__global__ void RegressionKernel(
    float4 *_beta,
    const TensorAccessor4D _inImg,
    const TensorAccessor4D _wgtImg,
    const TensorAccessor4D _varWgt,
    const TensorAccessor4D _albedo,
    const TensorAccessor4D _normal,
    const int batch, const int height, const int width, const int winSize, const int sparsity)
{
    const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x) * sparsity;
    const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y) * sparsity;

    if (cx >= width || cy >= height)
        return;

    const int P = COEFFI_DIM_PILOT;

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

    const float4 &cCol = make_float4(_inImg[batch][0][cy][cx], _inImg[batch][1][cy][cx], _inImg[batch][2][cy][cx], 0.f);
    const float4 &cColWgt = make_float4(_wgtImg[batch][0][cy][cx], _wgtImg[batch][1][cy][cx], _wgtImg[batch][2][cy][cx], 0.f);
    const float4 &cAlbedo = make_float4(_albedo[batch][0][cy][cx], _albedo[batch][1][cy][cx], _albedo[batch][2][cy][cx], 0.f);
    const float4 &cNormal = make_float4(_normal[batch][0][cy][cx], _normal[batch][1][cy][cx], _normal[batch][2][cy][cx], 0.f);
    const float cVar = _varWgt[batch][0][cy][cx];
    const float cStd = sqrtf(cVar);

    float sumW = 0.f;
    for (int iy = sy; iy <= ey; ++iy)
    {
        for (int ix = sx; ix <= ex; ++ix)
        {
            int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
            int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);

            int idx = y * width + x;

            const float4 &iCol = make_float4(_inImg[batch][0][y][x], _inImg[batch][1][y][x], _inImg[batch][2][y][x], 0.f);
            const float4 &iColWgt = make_float4(_wgtImg[batch][0][y][x], _wgtImg[batch][1][y][x], _wgtImg[batch][2][y][x], 0.f);
            const float4 &iAlbedo = make_float4(_albedo[batch][0][y][x], _albedo[batch][1][y][x], _albedo[batch][2][y][x], 0.f);
            const float4 &iNormal = make_float4(_normal[batch][0][y][x], _normal[batch][1][y][x], _normal[batch][2][y][x], 0.f);
            const float iVar = _varWgt[batch][0][y][x];
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

std::vector<torch::Tensor> cross_regression_cuda_forward(
    const torch::Tensor _imgA,   // [B, 3, H, W]
    const torch::Tensor _imgB,   // [B, 3, H, W]
    const torch::Tensor _albedo, // [B, 3, H, W]
    const torch::Tensor _normal, // [B, 3, H, W]
    const int winSize, const int sparsity, const int reconScale)
{
    CHECK_INPUT(_imgA);
    CHECK_INPUT(_imgB);
    CHECK_INPUT(_albedo);
    CHECK_INPUT(_normal);

    const int nBatch = _imgA.size(0);
    const int height = _imgA.size(2);
    const int width = _imgA.size(3);
    const int smallWidth = width / sparsity;
    const int smallHeight = height / sparsity;

    // Input pointers
    auto imgAptr = _imgA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto imgBptr = _imgB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto albedoptr = _albedo.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto normalptr = _normal.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Output tensors and pointers (4D tensors)
    auto outVarA = torch::zeros({nBatch, 1, height, width}, torch::CUDA(torch::kFloat32));
    auto outVarB = torch::zeros({nBatch, 1, height, width}, torch::CUDA(torch::kFloat32));
    auto outImgA = torch::zeros({nBatch, 3, height, width}, torch::CUDA(torch::kFloat32));
    auto outImgB = torch::zeros({nBatch, 3, height, width}, torch::CUDA(torch::kFloat32));
    auto outVarAptr = outVarA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outVarBptr = outVarB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outImgAptr = outImgA.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto outImgBptr = outImgB.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // Intermediate tensors
    auto betaA = torch::zeros({COEFFI_DIM_PILOT, smallHeight, smallWidth, 4 /* float4 */}, torch::CUDA(torch::kFloat32));
    auto betaB = torch::zeros({COEFFI_DIM_PILOT, smallHeight, smallWidth, 4 /* float4 */}, torch::CUDA(torch::kFloat32));
    // Here we're taking the raw pointer of underlying tensors and casting it to a float4 pointer
    auto betaAptr = reinterpret_cast<float4 *>(betaA.data_ptr<float>());
    auto betaBptr = reinterpret_cast<float4 *>(betaB.data_ptr<float>());

    const int blockDim = 8;
    dim3 threads(blockDim, blockDim);
    dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));
    dim3 gridSparse(iDivUp(width / sparsity, blockDim), iDivUp(height / sparsity, blockDim));
    dim3 gridRecon(iDivUp(width / reconScale, blockDim), iDivUp(height / reconScale, blockDim));

    for (int b = 0; b < nBatch; ++b)
    {
        EstInputVarianceKernel<<<grid, threads>>>(imgAptr, outVarAptr, b, width, height, 3);
        EstInputVarianceKernel<<<grid, threads>>>(imgBptr, outVarBptr, b, width, height, 3);

        RegressionKernel<<<gridSparse, threads>>>(betaAptr, imgBptr, imgAptr, outVarAptr, albedoptr, normalptr, b, height, width, winSize, sparsity);
        RegressionKernel<<<gridSparse, threads>>>(betaBptr, imgAptr, imgBptr, outVarBptr, albedoptr, normalptr, b, height, width, winSize, sparsity);

        BlockRecon<<<gridRecon, threads>>>(outImgAptr, betaAptr, imgBptr, imgAptr, outVarAptr, albedoptr, normalptr, b, height, width, winSize, sparsity, reconScale);
        BlockRecon<<<gridRecon, threads>>>(outImgBptr, betaBptr, imgAptr, imgBptr, outVarBptr, albedoptr, normalptr, b, height, width, winSize, sparsity, reconScale);
    }

    return {outImgA, outImgB, outVarA, outVarB};
}
