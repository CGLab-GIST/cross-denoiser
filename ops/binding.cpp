#include <torch/extension.h>

std::vector<torch::Tensor> cross_regression_cuda_forward(
    const torch::Tensor _imga,
    const torch::Tensor _imgb,
    const torch::Tensor _albedo,
    const torch::Tensor _normal,
    const int winSize, const int sparsity, const int reconScale);

std::vector<torch::Tensor> reproject_cuda_forward(
    const std::vector<torch::Tensor> _inputList,
    const torch::Tensor _mvec,
    const torch::Tensor _linearZ,
    const torch::Tensor _prevLinearZ,
    const torch::Tensor _normal,
    const torch::Tensor _prevNormal,
    const torch::Tensor _pnFwidth,
    const torch::Tensor _opacity,
    const torch::Tensor _prevOpacity);

std::vector<torch::Tensor> temporal_screening_cuda_forward(
    const torch::Tensor _current1,
    const torch::Tensor _current2,
    const torch::Tensor _prev,
    const torch::Tensor _prev1,
    const torch::Tensor _prev2,
    const torch::Tensor _reprojSuccess,
    const int winCurr, const int winPrev);

torch::Tensor erosion2d_cuda_forward(
    const torch::Tensor _input,
    const int winSize);

std::vector<torch::Tensor> spatial_filter_cuda_forward(
    const torch::Tensor _imgA,   // [B, 3, H, W]
    const torch::Tensor _imgB,   // [B, 3, H, W]
    const torch::Tensor _albedo, // [B, 3, H, W]
    const torch::Tensor _normal, // [B, 3, H, W]
    const torch::Tensor _band,   // [B, BAND_RANK + 1, H, W]
    const int winSize);

torch::Tensor spatial_filter_cuda_backward(
    torch::Tensor _gradAccImgA, // [B, 3, H, W]
    torch::Tensor _gradAccImgB, // [B, 3, H, W]
    torch::Tensor _gradWgtSumA, // [B, 1, H, W]
    torch::Tensor _gradWgtSumB, // [B, 1, H, W]
    torch::Tensor _band,        // [B, BAND_RANK + 1, H, W]
    torch::Tensor _gradBandA,   // [B, W, H, BAND_RANK * 4]
    torch::Tensor _gradBandB,   // [B, W, H, BAND_RANK * 4]
    const int winSize);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cross_regression", &cross_regression_cuda_forward);
    m.def("reproject", &reproject_cuda_forward);
    m.def("temporal_screening", &temporal_screening_cuda_forward);
    m.def("erosion2d", &erosion2d_cuda_forward);
    m.def("spatial_filter_forward", &spatial_filter_cuda_forward);
    m.def("spatial_filter_backward", &spatial_filter_cuda_backward);
}