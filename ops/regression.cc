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

#include <cuda_runtime.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("CrossRegression")
    .Input("imga: float")
    .Input("imgb: float")
    .Input("albedo: float")
    .Input("normal: float")
    .Output("outa: float")
    .Output("outb: float")
    .Output("out_vara: float")
    .Output("out_varb: float")
    .Attr("win_size: int >= 1")
    .Attr("sparsity: int >= 1")
    .Attr("recon_scale: int >= 1 = 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {    
    auto B = c->Dim(c->input(0), 0);
    auto H = c->Dim(c->input(0), 1);
    auto W = c->Dim(c->input(0), 2);
    auto C = c->Dim(c->input(0), 3);

    int64_t recon_scale;
    TF_RETURN_IF_ERROR(c->GetAttr("recon_scale", &recon_scale));

    auto new_H = c->Value(H) / recon_scale;
    auto new_W = c->Value(W) / recon_scale;

    auto output_shape = c->MakeShape({B, new_H, new_W, C});
    c->set_output(0, output_shape);
    c->set_output(1, output_shape);

    c->set_output(2, c->MakeShape({B, H, W, 1}));
    c->set_output(3, c->MakeShape({B, H, W, 1}));

    return Status(); });

REGISTER_OP("SpatiotemporalFilter")
    .Input("imga: float")
    .Input("imgb: float")
    .Input("albedo: float")
    .Input("normal: float")
    .Input("band: float")
    .Output("out_acca: float")
    .Output("out_accb: float")
    .Output("out_wgtsuma: float")
    .Output("out_wgtsumb: float")
    .Attr("win_size: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));

    shape_inference::ShapeHandle img_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &img_shape));
    std::vector<shape_inference::DimensionHandle> dims;
    dims.emplace_back(c->MakeDim(c->Dim(img_shape, 0)));
    dims.emplace_back(c->MakeDim(c->Dim(img_shape, 1)));
    dims.emplace_back(c->MakeDim(c->Dim(img_shape, 2)));
    dims.emplace_back(c->MakeDim(1));

    c->set_output(2, c->MakeShape(dims));
    c->set_output(3, c->MakeShape(dims));

    return Status(); });

REGISTER_OP("SpatiotemporalFilterGrad")
    .Input("grad_acca: float")
    .Input("grad_accb: float")
    .Input("grad_wgtsuma: float")
    .Input("grad_wgtsumb: float")
    .Input("imga: float")
    .Input("imgb: float")
    .Input("img_albedo: float")
    .Input("img_normal: float")
    .Input("band: float")
    .Output("out_grad_band: float")
    .Attr("win_size: int >= 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
    c->set_output(0, c->input(8));
    return Status(); });

void CrossRegressionFunc(const GPUDevice &_dev,
                         const float *_imga, const float *_imgb, const float *_albedo, const float *_normal,
                         float *_outImgA, float *_outImgB, float *_outVarA, float *_outVarB,
                         float4 *_betaA, float4 *_betaB,
                         int nBatch, int height, int width, int winSize, int sparsity, int reconScale);

void SpatiotemporalFilterFunc(const GPUDevice &_dev, const float *_imga, const float *_imgb,
                              const float *_albedo, const float *_normal, const float *_band,
                              float *_outAccImgA, float *_outAccImgB, float *_outWgtSumA, float *_outWgtSumB,
                              float4 *_gradBandA, float4 *_gradBandB,
                              int nBatch, int height, int width, int winSize);

void SpatiotemporalFilterGradFunc(const GPUDevice &_dev, const float *_gradAccImgA, const float *_gradAccImgB,
                                  const float *_gradWgtSumA, const float *_gradWgtSumB,
                                  const float *_imga, const float *_imgb,
                                  const float *_albedo, const float *_normal, const float *_band,
                                  float *_outGradBand,
                                  const float4 *_gradBandA, const float4 *_gradBandB,
                                  int nBatch, int height, int width, int winSize);

class CrossRegressionOp : public OpKernel
{
public:
    explicit CrossRegressionOp(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("win_size", &winSize);
        context->GetAttr("sparsity", &sparsity);
        context->GetAttr("recon_scale", &reconScale);

        OP_REQUIRES(context, reconScale <= sparsity,
                    errors::InvalidArgument("recon_scale must be less than or equal to sparsity"));
    }

    static void allocate_temp(OpKernelContext *context, Tensor *&tensor, const TensorShape &shape)
    {
        if (tensor)
            delete tensor;
        tensor = new Tensor();
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_FLOAT, shape, tensor));
        cudaMemset(tensor->flat<float>().data(), 0, tensor->NumElements() * sizeof(float));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &imga = context->input(0);
        const Tensor &imgb = context->input(1);
        const Tensor &albedo = context->input(2);
        const Tensor &normal = context->input(3);

        TensorShape imgShape = imga.shape();
        auto nBatch = imgShape.dim_size(0);
        auto imgHeight = imgShape.dim_size(1);
        auto imgWidth = imgShape.dim_size(2);

        OP_REQUIRES(context, imgHeight % reconScale == 0 && imgWidth % reconScale == 0,
                    errors::InvalidArgument("Image height and width must be divisible by recon_scale"));

        auto imgHeightRecon = imgHeight / reconScale;
        auto imgWidthRecon = imgWidth / reconScale;
        TensorShape reconShape = {nBatch, imgHeightRecon, imgWidthRecon, imgShape.dim_size(3)};

        Tensor *out_imga = NULL, *out_imgb = NULL;
        Tensor *out_vara = NULL, *out_varb = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, reconShape, &out_imga));
        OP_REQUIRES_OK(context, context->allocate_output(1, reconShape, &out_imgb));
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{1, imgHeight, imgWidth, 1}, &out_vara));
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{1, imgHeight, imgWidth, 1}, &out_varb));

        // Intermediate variables
        if (betaA == NULL)
        {
            int smallHeight = imgHeight / sparsity;
            int smallWidth = imgWidth / sparsity;
            allocate_temp(context, betaA, TensorShape{1, smallHeight, smallWidth, 6 /* COEFFI_DIM_PILOT */ * 4 /* float4 */});
            allocate_temp(context, betaB, TensorShape{1, smallHeight, smallWidth, 6 /* COEFFI_DIM_PILOT */ * 4 /* float4 */});
        }
        float4 *betaAptr = reinterpret_cast<float4 *>(betaA->flat<float>().data());
        float4 *betaBptr = reinterpret_cast<float4 *>(betaB->flat<float>().data());

        CrossRegressionFunc(context->eigen_device<GPUDevice>(),
                            imga.flat<float>().data(), imgb.flat<float>().data(),
                            albedo.flat<float>().data(), normal.flat<float>().data(),
                            out_imga->tensor<float, 4>().data(), out_imgb->tensor<float, 4>().data(),
                            out_vara->tensor<float, 4>().data(), out_varb->tensor<float, 4>().data(),
                            betaAptr, betaBptr,
                            1, imgHeight, imgWidth, winSize, sparsity, reconScale);
    }

private:
    int winSize, sparsity, reconScale;
    static Tensor *betaA, *betaB;
};

Tensor *CrossRegressionOp::betaA = NULL;
Tensor *CrossRegressionOp::betaB = NULL;

class SpatiotemporalFilterInterface : public OpKernel
{
public:
    explicit SpatiotemporalFilterInterface(OpKernelConstruction *context) : OpKernel(context)
    {
        context->GetAttr("win_size", &winSize);
    }

    static void allocate_temp(OpKernelContext *context, Tensor *&tensor, const TensorShape &shape)
    {
        if (tensor)
            delete tensor;
        tensor = new Tensor();
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_FLOAT, shape, tensor));
        cudaMemset(tensor->flat<float>().data(), 0, tensor->NumElements() * sizeof(float));
    }

    static void initialize(OpKernelContext *context, int batch, int height, int width, bool force = false)
    {
        if (gradBandA == NULL || force)
        {
            allocate_temp(context, gradBandA, {batch, height, width, 4 /* BAND_RANK */ * 4 /* float4 */});
            allocate_temp(context, gradBandB, {batch, height, width, 4 /* BAND_RANK */ * 4 /* float4 */});
        }
    }

protected:
    int winSize;
    static Tensor *gradBandA, *gradBandB;
};

Tensor *SpatiotemporalFilterInterface::gradBandA = NULL;
Tensor *SpatiotemporalFilterInterface::gradBandB = NULL;

class SpatiotemporalFilterOp : public SpatiotemporalFilterInterface
{
public:
    explicit SpatiotemporalFilterOp(OpKernelConstruction *context) : SpatiotemporalFilterInterface(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &imga = context->input(0);
        const Tensor &imgb = context->input(1);
        const Tensor &albedo = context->input(2);
        const Tensor &normal = context->input(3);
        const Tensor &band = context->input(4);

        const TensorShape &img_shape = imga.shape();
        TensorShape wgt_sum_shape;
        wgt_sum_shape.AddDim(img_shape.dim_size(0));
        wgt_sum_shape.AddDim(img_shape.dim_size(1));
        wgt_sum_shape.AddDim(img_shape.dim_size(2));
        wgt_sum_shape.AddDim(1);

        Tensor *out_accimga = NULL;
        Tensor *out_accimgb = NULL;
        Tensor *out_wgtsuma = NULL;
        Tensor *out_wgtsumb = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, img_shape, &out_accimga));
        OP_REQUIRES_OK(context, context->allocate_output(1, img_shape, &out_accimgb));
        OP_REQUIRES_OK(context, context->allocate_output(2, wgt_sum_shape, &out_wgtsuma));
        OP_REQUIRES_OK(context, context->allocate_output(3, wgt_sum_shape, &out_wgtsumb));
        auto out_mat_accimga = out_accimga->tensor<float, 4>();
        auto out_mat_accimgb = out_accimgb->tensor<float, 4>();
        auto out_mat_wgtsuma = out_wgtsuma->tensor<float, 4>();
        auto out_mat_wgtsumb = out_wgtsumb->tensor<float, 4>();

        initialize(context, img_shape.dim_size(0), img_shape.dim_size(1), img_shape.dim_size(2));

        float4 *gradBandAptr = reinterpret_cast<float4 *>(gradBandA->flat<float>().data());
        float4 *gradBandBptr = reinterpret_cast<float4 *>(gradBandB->flat<float>().data());

        SpatiotemporalFilterFunc(context->eigen_device<GPUDevice>(),
                                 imga.flat<float>().data(), imgb.flat<float>().data(),
                                 albedo.flat<float>().data(), normal.flat<float>().data(), band.flat<float>().data(),
                                 out_mat_accimga.data(), out_mat_accimgb.data(), out_mat_wgtsuma.data(), out_mat_wgtsumb.data(),
                                 gradBandAptr, gradBandBptr,
                                 img_shape.dim_size(0), img_shape.dim_size(1), img_shape.dim_size(2), winSize);
    }
};
class SpatiotemporalFilterGradOp : public SpatiotemporalFilterInterface
{
public:
    explicit SpatiotemporalFilterGradOp(OpKernelConstruction *context) : SpatiotemporalFilterInterface(context)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &grad_acc_imga = context->input(0);
        const Tensor &grad_acc_imgb = context->input(1);
        const Tensor &grad_wgt_suma = context->input(2);
        const Tensor &grad_wgt_sumb = context->input(3);
        const Tensor &imga = context->input(4);
        const Tensor &imgb = context->input(5);
        const Tensor &albedo = context->input(6);
        const Tensor &normal = context->input(7);
        const Tensor &band = context->input(8);

        const TensorShape &band_shape = band.shape();

        Tensor *out_grad_band = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, band_shape, &out_grad_band));
        auto out_mat_grad_band = out_grad_band->tensor<float, 4>();

        initialize(context, band_shape.dim_size(0), band_shape.dim_size(1), band_shape.dim_size(2));

        float4 *gradBandAptr = reinterpret_cast<float4 *>(gradBandA->flat<float>().data());
        float4 *gradBandBptr = reinterpret_cast<float4 *>(gradBandB->flat<float>().data());

        SpatiotemporalFilterGradFunc(context->eigen_device<GPUDevice>(),
                                     grad_acc_imga.flat<float>().data(), grad_acc_imgb.flat<float>().data(),
                                     grad_wgt_suma.flat<float>().data(), grad_wgt_sumb.flat<float>().data(),
                                     imga.flat<float>().data(), imgb.flat<float>().data(),
                                     albedo.flat<float>().data(), normal.flat<float>().data(), band.flat<float>().data(),
                                     out_mat_grad_band.data(),
                                     gradBandAptr, gradBandBptr,
                                     band_shape.dim_size(0), band_shape.dim_size(1), band_shape.dim_size(2), winSize);
    }
};

REGISTER_KERNEL_BUILDER(Name("CrossRegression").Device(DEVICE_GPU), CrossRegressionOp);
REGISTER_KERNEL_BUILDER(Name("SpatiotemporalFilter").Device(DEVICE_GPU), SpatiotemporalFilterOp);
REGISTER_KERNEL_BUILDER(Name("SpatiotemporalFilterGrad").Device(DEVICE_GPU), SpatiotemporalFilterGradOp);