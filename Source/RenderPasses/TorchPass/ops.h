#include <torch/torch.h>
#include <torch/extension.h>

using namespace torch::autograd;

#include <cuda_runtime.h>

std::vector<torch::Tensor> cross_regression_cuda_forward(
    const torch::Tensor _imga,
    const torch::Tensor _imgb,
    const torch::Tensor _albedo,
    const torch::Tensor _normal,
    const int winSize, const int sparsity, const int reconScale);

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
);

torch::Tensor erosion2d_cuda_forward(
    const torch::Tensor _input,
    const int winSize);

std::vector<torch::Tensor> temporal_screening_cuda_forward(
    const torch::Tensor _current1,
    const torch::Tensor _current2,
    const torch::Tensor _prev,
    const torch::Tensor _prev1,
    const torch::Tensor _prev2,
    const torch::Tensor _reprojSuccess,
    const int winCurr, const int winPrev);

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

class CrossRegression : public Function<CrossRegression>
{
public:
    static tensor_list forward(AutogradContext *ctx, torch::Tensor imga, torch::Tensor imgb, torch::Tensor albedo, torch::Tensor normal, int winSize, int sparsity, int reconScale)
    {
        return cross_regression_cuda_forward(imga, imgb, albedo, normal, winSize, sparsity, reconScale);
    }

    // Dummy backward function
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        return {
            torch::Tensor(), // Gradient for imga
            torch::Tensor(), // Gradient for imgb
            torch::Tensor(), // Gradient for albedo
            torch::Tensor(), // Gradient for normal
            torch::Tensor(), // Placeholder for int arguments (not tensors)
            torch::Tensor(), // Same here
            torch::Tensor()  // Same here
        };
    }
};

class Reproject : public Function<Reproject>
{
public:
    static tensor_list forward(AutogradContext *ctx, std::vector<torch::Tensor> inputList, torch::Tensor mvec, torch::Tensor linearZ, torch::Tensor prevLinearZ, torch::Tensor normal, torch::Tensor prevNormal, torch::Tensor pnFwidth, torch::Tensor opacity, torch::Tensor prevOpacity)
    {
        return reproject_cuda_forward(inputList, mvec, linearZ, prevLinearZ, normal, prevNormal, pnFwidth, opacity, prevOpacity);
    }

    // Dummy backward function
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        return {
            torch::Tensor(), // Gradient for inputList
            torch::Tensor(), // Gradient for mvec
            torch::Tensor(), // Gradient for linearZ
            torch::Tensor(), // Gradient for prevLinearZ
            torch::Tensor(), // Gradient for normal
            torch::Tensor(), // Gradient for prevNormal
            torch::Tensor(), // Gradient for pnFwidth
            torch::Tensor(), // Gradient for opacity
            torch::Tensor()  // Gradient for prevOpacity
        };
    }
};

class Erosion2d : public Function<Erosion2d>
{
public:
    static tensor_list forward(AutogradContext *ctx, torch::Tensor input, int winSize)
    {
        auto output = erosion2d_cuda_forward(input, winSize);
        return {output};
    }

    // Dummy backward function
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_output)
    {
        return {torch::Tensor()};
    }
};

class TemporalScreening : public Function<TemporalScreening>
{
public:
    static tensor_list forward(AutogradContext *ctx, torch::Tensor current1, torch::Tensor current2, torch::Tensor prev, torch::Tensor prev1, torch::Tensor prev2, torch::Tensor reprojSuccess, int winCurr, int winPrev)
    {
        return temporal_screening_cuda_forward(current1, current2, prev, prev1, prev2, reprojSuccess, winCurr, winPrev);
    }

    // Dummy backward function
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        return {
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor() //
        };
    }
};

class SpatialFilter : public Function<SpatialFilter>
{
public:
    static tensor_list forward(AutogradContext *ctx, torch::Tensor imgA, torch::Tensor imgB, torch::Tensor albedo, torch::Tensor normal, torch::Tensor band, int winSize)
    {
        // return {outAccA, outAccB, outWgtSumA, outWgtSumB, gradBandA, gradBandB};
        auto outputTensors = spatial_filter_cuda_forward(imgA, imgB, albedo, normal, band, winSize);
        ctx->save_for_backward({band, outputTensors[4], outputTensors[5]});
        ctx->saved_data["winSize"] = torch::IValue(winSize);
        return {outputTensors[0], outputTensors[1], outputTensors[2], outputTensors[3]};
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto band = saved[0];
        auto gradBandA = saved[1];
        auto gradBandB = saved[2];
        int winSize = ctx->saved_data["winSize"].toInt();
        auto gradOutput = spatial_filter_cuda_backward(grad_outputs[0], grad_outputs[1], grad_outputs[2], grad_outputs[3], band, gradBandA, gradBandB, winSize);
        return {
            torch::Tensor(), // Gradient for imgA
            torch::Tensor(), // Gradient for imgB
            torch::Tensor(), // Gradient for albedo
            torch::Tensor(), // Gradient for normal
            gradOutput,      // Gradient for band
            torch::Tensor()  // Gradient for winSize
        };
    }
};