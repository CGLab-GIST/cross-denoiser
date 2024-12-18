/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "TorchPass.h"
#include "Utils.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#include "tinyexr.h"

namespace
{
    const char kDesc[] = "Insert pass description here";

    const char kInputColor[] = "color";
    const char kInputColor2[] = "color2";
    const char kInputAlbedo[] = "albedo";
    const char kInputNormal[] = "normal";
    const char kInputMvec[] = "mvec";
    const char kInputLinearZ[] = "linearZ";
    const char kInputPnFwidth[] = "pnFwidth";
    const char kInputDiffuseOpacity[] = "diffuseOpacity";

    const char kOutputCrossA[] = "crossA";
    const char kOutputCrossB[] = "crossB";
    const char kOutputOut[] = "out";
    const char kOutputAlpha[] = "alpha";
    const char kOutputSuccess[] = "success";
    const char kOutputPrev[] = "prev";

    const char kPrevNormal[] = "prevNormal";
    const char kPrevLinearZ[] = "prevLinearZ";
    const char kPrevCrossA[] = "prevCrossA";
    const char kPrevCrossB[] = "prevCrossB";
    const char kPrevOut[] = "prevOut";

    const Falcor::ChannelList kInputChannels =
        {
            {kInputColor, "gColor", "Input color buffer", false, ResourceFormat::RGBA32Float},
            {kInputColor2, "gColor2", "Input color buffer", false, ResourceFormat::RGBA32Float},
            {kInputAlbedo, "gAlbedo", "Input albedo buffer", false, ResourceFormat::RGBA32Float},
            {kInputNormal, "gNormal", "Input normal buffer", false, ResourceFormat::RGBA32Float},
            {kInputMvec, "gMvec", "Input mvec buffer", false, ResourceFormat::RG32Float},
            {kInputLinearZ, "gLinearZ", "Input linearZ buffer", false, ResourceFormat::RG32Float},
            {kInputPnFwidth, "gPnFwidth", "Input pnFwidth buffer", false, ResourceFormat::RG32Float},
            {kInputDiffuseOpacity, "gDiffuseOpacity", "Input diffuseOpacity buffer", false, ResourceFormat::RGBA32Float} //
    };
    const Falcor::ChannelList kOutputChannels =
        {
            {kOutputCrossA, "", "Cross-regression image", true, ResourceFormat::RGBA32Float},
            {kOutputCrossB, "", "Cross-regression image", true, ResourceFormat::RGBA32Float},
            {kOutputAlpha, "", "Output image", true, ResourceFormat::R32Float},
            {kOutputSuccess, "", "Output image", true, ResourceFormat::R32Float},
            {kOutputOut, "", "Output image", true, ResourceFormat::RGBA32Float},
            {kOutputPrev, "", "Output image", true, ResourceFormat::RGBA32Float} //
    };
    const Falcor::ChannelList kInternalChannels =
        {
            {kPrevCrossA, "", "Cross-regression image", true, ResourceFormat::RGBA32Float},
            {kPrevCrossB, "", "Cross-regression image", true, ResourceFormat::RGBA32Float},
            {kPrevNormal, "", "Previous normal", true, ResourceFormat::RGBA32Float},
            {kPrevLinearZ, "", "Previous linearZ", true, ResourceFormat::RG32Float},
            {kPrevOut, "", "Previous output", true, ResourceFormat::RGBA32Float} //
    };
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char *getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary &lib)
{
    lib.registerClass("TorchPass", kDesc, TorchPass::create);
}

TorchPass::TorchPass() : BaseMLPass()
{
    CUDAUtils::initCUDA();

    // Create a Net.
    mNetwork->to(torch::kCUDA, true); // Non-blocking: true (pinned memory)
    mOptimizer = std::make_shared<torch::optim::Adam>(mNetwork->parameters(), torch::optim::AdamOptions(kLearningRate).eps(1e-7));
}

TorchPass::SharedPtr TorchPass::create(RenderContext *pRenderContext, const Dictionary &dict)
{
    SharedPtr pPass = SharedPtr(new TorchPass);
    return pPass;
}

std::string TorchPass::getDesc() { return kDesc; }

Dictionary TorchPass::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection TorchPass::reflect(const CompileData &compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);

    for (const auto &channel : kInternalChannels)
    {
        reflector.addInternal(channel.name, channel.desc).format(channel.format);
    }

    return reflector;
}

void *exportBufferToCudaDevice(Buffer::SharedPtr &buf)
{
    if (buf == nullptr)
        return nullptr;
    return CUDAUtils::getSharedDevicePtr(buf->getSharedApiHandle(), (uint32_t)buf->getSize());
}

void writeExrData(const float *data, int width, int height, const char *filename, const std::vector<std::string> &channel_names)
{
    const int num_channels = channel_names.size();

    // Initialize EXR Header and EXR Image
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = num_channels;

    // Separate input data into individual channel arrays
    std::vector<std::vector<float>> images(num_channels);
    for (int c = 0; c < num_channels; ++c)
    {
        images[c].resize(width * height);
    }

    for (int i = 0; i < width * height; i++)
    {
        for (int c = 0; c < num_channels; c++)
        {
            images[c][i] = data[num_channels * i + num_channels - 1 - c]; // Separate per-channel data
        }
    }

    // Set up image pointers (EXR expects data in B, G, R, A order if these channels exist)
    std::vector<float *> image_ptrs(num_channels);
    for (int c = 0; c < num_channels; c++)
    {
        image_ptrs[c] = images[c].data();
    }

    image.images = (unsigned char **)image_ptrs.data();
    image.width = width;
    image.height = height;

    // Set up EXR Header channels
    header.num_channels = num_channels;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * num_channels);

    for (int c = 0; c < num_channels; ++c)
    {
        strncpy(header.channels[c].name, channel_names[c].c_str(), 255);
        header.channels[c].name[255] = '\0';
    }

    // Specify pixel types
    header.pixel_types = (int *)malloc(sizeof(int) * num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * num_channels);

    for (int c = 0; c < num_channels; ++c)
    {
        header.pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;          // Input data is float
        header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_HALF; // Output stored as half (16-bit float)
    }

    // Save the image
    const char *err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, filename, &err);
    if (ret != TINYEXR_SUCCESS)
    {
        fprintf(stderr, "Save EXR error: %s\n", err);
        FreeEXRErrorMessage(err); // Clean up error message
        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);
        return;
    }

    printf("Saved EXR file: %s\n", filename);

    // Cleanup
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void writeExrTensor(torch::Tensor tensor, const std::string &filename, const std::vector<std::string> &channel_names = {})
{
    // Ensure the tensor is on the CPU
    if (tensor.device().is_cuda())
    {
        tensor = tensor.to(torch::kCPU);
    }

    // Change memory layout to ChannelsLast
    tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast);

    std::vector<std::string> channel_names_copy = channel_names;
    if (channel_names_copy.empty())
    {
        if (tensor.size(1) == 1)
            // Single channel
            channel_names_copy = {"Y"};
        else if (tensor.size(1) == 3)
            // RGB
            channel_names_copy = {"R", "G", "B"};
        else if (tensor.size(1) == 4)
            // RGBA
            channel_names_copy = {"R", "G", "B", "A"};
        else
        {
            // Unknown number of channels
            channel_names_copy.resize(tensor.size(1));
            for (int i = 0; i < tensor.size(1); ++i)
            {
                channel_names_copy[i] = std::to_string(i);
            }
        }
    }

    // Write to EXR
    int width = tensor.size(3);
    int height = tensor.size(2);
    writeExrData(tensor.data_ptr<float>(), width, height, filename.c_str(), channel_names_copy);
}

void TorchPass::execute(RenderContext *pRenderContext, const RenderData &renderData)
{
    using namespace torch::indexing;
    const auto frameCount = gpFramework->getGlobalClock().getFrame();

    std::map<std::string, std::shared_ptr<Texture>> inTex, outTex, internalTex;

    // [ Get Texture ] /////////////////////////////////////////////////
    // Input
    for (const auto &channel : kInputChannels)
    {
        inTex[channel.name] = renderData[channel.name]->asTexture();
    }

    // Output
    for (const auto &channel : kOutputChannels)
    {
        outTex[channel.name] = renderData[channel.name]->asTexture();
    }

    // Internal
    for (const auto &channel : kInternalChannels)
    {
        internalTex[channel.name] = renderData[channel.name]->asTexture();
    }

    // [ Process ] /////////////////////////////////////////////////////
    if (mEnabled)
    {
        static bool once = true;
        if (once)
        {
            int width = inTex[kInputColor]->getWidth();
            int height = inTex[kInputColor]->getHeight();
            init(width, height);
            once = false;
        }
        torch::cuda::synchronize();

        PROFILE("TorchPass");
        {
            PROFILE("TorchPass::CopyFromFalcorToCUDA");
            // Copy from Falcor texture to CUDA buffer (only for inputs and internals)
            for (const auto &channel : kInputChannels)
            {
                BaseMLPass::copyTextureToBuffer(pRenderContext, inTex[channel.name], channel.name);
            }
            for (const auto &channel : kInternalChannels)
            {
                BaseMLPass::copyTextureToBuffer(pRenderContext, internalTex[channel.name], channel.name);
            }

            // Ensure buffers are filled
            pRenderContext->flush(true);
            torch::cuda::synchronize();
        }

        if (mInference)
        {
            // Input tensors
            torch::Tensor colorA, colorB, albedo, normal, mvec, linearZ, pnFwidth, opacity;
            // Internal tensors
            torch::Tensor prevNormal, prevLinearZ;
            torch::Tensor prevCrossA, prevCrossB;
            torch::Tensor prevOut;
            // Output tensors (file save)
            torch::Tensor outCrossA, outCrossB;
            torch::Tensor outAlpha, outSuccess;
            torch::Tensor outPrev;
            torch::Tensor out;
            // Intermediate tensors
            torch::Tensor outA, outB, out_log;
            torch::Tensor crossA, crossB;
            torch::Tensor success, successPrev;
            torch::Tensor alpha, band;
            torch::Tensor sumA, sumB, wgtSumA, wgtSumB;
            torch::Tensor avgLoss;

            {
                PROFILE("TorchPass::forward");

                // Tensor alias
                {
                    colorA = torch::log1p_(mTensors[kInputColor].detach());
                    colorB = torch::log1p_(mTensors[kInputColor2].detach());
                    albedo = mTensors[kInputAlbedo].detach();
                    normal = mTensors[kInputNormal].detach();
                    mvec = mTensors[kInputMvec].detach();
                    linearZ = mTensors[kInputLinearZ].detach();
                    pnFwidth = mTensors[kInputPnFwidth].detach();
                    opacity = mTensors[kInputDiffuseOpacity].index({Slice(), Slice(3, 4), Slice(), Slice()}).detach();

                    prevNormal = mTensors[kPrevNormal].detach();
                    prevLinearZ = mTensors[kPrevLinearZ].detach();
                    prevCrossA = mTensors[kPrevCrossA].detach();
                    prevCrossB = mTensors[kPrevCrossB].detach();
                    prevOut = mTensors[kPrevOut].detach();

                    outCrossA = mTensors[kOutputCrossA].detach();
                    outCrossB = mTensors[kOutputCrossB].detach();
                    out = mTensors[kOutputOut].detach();
                    outAlpha = mTensors[kOutputAlpha].detach();
                    outSuccess = mTensors[kOutputSuccess].detach();
                    outPrev = mTensors[kOutputPrev].detach();
                }

                {
                    PROFILE("TorchPass::forward::cross_regression");
                    auto outputs = CrossRegression::apply(colorA, colorB, albedo, normal, kRegressionWidth, kSparsity, 1);
                    crossA = outputs[0];
                    crossB = outputs[1];
                    torch::cuda::synchronize();
                }

                {
                    PROFILE("TorchPass::forward::reproject");
                    std::vector<torch::Tensor> inputList = {prevOut, prevCrossA, prevCrossB};
                    auto outputs_reproj = Reproject::apply(inputList, mvec, linearZ, prevLinearZ, normal, prevNormal, pnFwidth, mOnes1, mOnes1);
                    success = outputs_reproj[0];
                    auto prevOut_old = outputs_reproj[1];
                    auto prevCrossA_old = outputs_reproj[2];
                    auto prevCrossB_old = outputs_reproj[3];

                    success = success.to(torch::kFloat32);
                    success = Erosion2d::apply(success, 3)[0];

                    auto outputs = TemporalScreening::apply(crossA, crossB, prevOut_old, prevCrossA_old, prevCrossB_old, success, 7, 1);
                    prevOut.copy_(outputs[0]);
                    prevCrossA.copy_(outputs[1]);
                    prevCrossB.copy_(outputs[2]);
                    successPrev = outputs[3];
                    torch::cuda::synchronize();
                }

                {
                    PROFILE("TorchPass::forward::network_inference");
                    std::pair<torch::Tensor, torch::Tensor> params = mNetwork->forward(crossA, crossB, prevOut, albedo, normal);
                    band = params.first;
                    alpha = params.second;
                    torch::cuda::synchronize();
                }

                {
                    PROFILE("TorchPass::forward::spatial_filter");
                    auto outputs = SpatialFilter::apply(crossA, crossB, albedo, normal, band, kFilterWidth);
                    sumA = outputs[0];
                    sumB = outputs[1];
                    wgtSumA = outputs[2];
                    wgtSumB = outputs[3];
                    outA = sumA / (wgtSumA + 1e-4);
                    outB = sumB / (wgtSumB + 1e-4);
                    out_log = (sumA + sumB) / (wgtSumA + wgtSumB + 1e-4);
                    torch::cuda::synchronize();
                }

                {
                    PROFILE("TorchPass::forward::temporal_filter");
                    opacity = torch::where(opacity > 0.f, 1.f, 0.f);
                    success = success * successPrev;
                    alpha = success * alpha + (1 - success);
                    out_log = alpha * out_log + (1 - alpha) * prevOut;
                    out_log *= opacity;
                    out.copy_(out_log);
                    torch::cuda::synchronize();
                }

                {
                    PROFILE("TorchPass::forward::copy_output");
                    outCrossA.copy_(crossA);
                    outCrossB.copy_(crossB);
                    outSuccess.copy_(success);
                    outPrev.copy_(prevOut);
                    outAlpha.copy_(alpha);
                    torch::cuda::synchronize();
                }
            }

            if (mBackprop)
            {
                PROFILE("TorchPass::backprop");
                static int epoch = 0;
                Tensor loss1 = torch::square(outA - crossB) / (torch::square(crossB) + 1e-2);
                Tensor loss2 = torch::square(outB - crossA) / (torch::square(crossA) + 1e-2);
                Tensor spatialLoss = (loss1 + loss2) * 0.5f;

                Tensor tLoss1 = torch::square(outA - prevCrossB) / (torch::square(prevCrossB) + 1e-2);
                Tensor tLoss2 = torch::square(outB - prevCrossA) / (torch::square(prevCrossA) + 1e-2);
                Tensor temporalLoss = (tLoss1 + tLoss2) * 0.5f;

                Tensor loss = (spatialLoss + temporalLoss) * 0.5f;
                loss *= opacity;
                avgLoss = loss.mean();

                mOptimizer->zero_grad();
                avgLoss.backward();
                mOptimizer->step();

                if (epoch++ % 30 == 0)
                    std::cout << "Epoch: " << epoch++ << " | Loss: " << avgLoss.item<float>() << std::endl;
                torch::cuda::synchronize();
            }

            {
                PROFILE("TorchPass::copy_previous");
                prevCrossA.copy_(outCrossA);
                prevCrossB.copy_(outCrossB);
                prevNormal.copy_(normal);
                prevLinearZ.copy_(linearZ);
                prevOut.copy_(out_log.detach()); // FIXME: Check why detach is needed to avoid memory leak
                torch::cuda::synchronize();
            }

            {
                PROFILE("TorchPass::output");
                torch::expm1_(outCrossA);
                torch::expm1_(outCrossB);
                torch::expm1_(out).clamp_min_(0);
                torch::expm1_(outPrev);
                torch::cuda::synchronize();
            }

            // Copy from CUDA buffer to Falcor texture
            {
                PROFILE("TorchPass::CopyFromCUDAToFalcor");
                for (const auto &channel : kOutputChannels)
                {
                    if (outTex[channel.name])
                        BaseMLPass::copyBufferToTexture(pRenderContext, channel.name, outTex[channel.name]);
                }
                for (const auto &channel : kInternalChannels)
                {
                    BaseMLPass::copyBufferToTexture(pRenderContext, channel.name, internalTex[channel.name]);
                }
            }
        }
        else
        {
            printf("TorchPass::execute: Inference is disabled\n");
        }
    }
    ////////////////////////////////////////////////////////////////////
}

void TorchPass::init(uint32_t width, uint32_t height)
{
    using namespace torch::indexing;

    // Add interop (Falcor Buffer and CUDA device pointer)
    for (const auto &channel : kInputChannels)
    {
        BaseMLPass::addInterop(channel.name, width, height, channel.format);
    }
    for (const auto &channel : kOutputChannels)
    {
        BaseMLPass::addInterop(channel.name, width, height, channel.format);
    }
    for (const auto &channel : kInternalChannels)
    {
        BaseMLPass::addInterop(channel.name, width, height, channel.format);
    }

    // Init torch tensor from CUDA buffer
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available!" << std::endl;

        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCUDA, 0)
                .requires_grad(false);

        auto processTensor = [](torch::Tensor tensor, const bool reduceTo3 = true) -> torch::Tensor
        {
            // [H, W, ?] -> [H, W, 3]
            if (reduceTo3 && tensor.size(2) >= 3)
                tensor = tensor.index({Slice(), Slice(), Slice(0, 3, None)});
            // [H, W, C] -> [C, H, W]
            tensor = tensor.permute({2, 0, 1});
            // [C, H, W] -> [1, C, H, W]
            tensor = tensor.unsqueeze(0);
            // These tensors do not require gradients
            return tensor.set_requires_grad(false);
        };

        for (const auto &channel : kInputChannels)
        {
            bool isDiffuseOpacity = channel.name == kInputDiffuseOpacity;
            uint32_t channelCount = getFormatChannelCount(channel.format);
            mTensors[channel.name] = torch::from_blob(BaseMLPass::getDevicePtr(channel.name), {height, width, channelCount}, options);
            mTensors[channel.name] = processTensor(mTensors[channel.name], !isDiffuseOpacity);
        }
        for (const auto &channel : kOutputChannels)
        {
            uint32_t channelCount = getFormatChannelCount(channel.format);
            mTensors[channel.name] = torch::from_blob(BaseMLPass::getDevicePtr(channel.name), {height, width, channelCount}, options);
            mTensors[channel.name] = processTensor(mTensors[channel.name]);
        }
        for (const auto &channel : kInternalChannels)
        {
            uint32_t channelCount = getFormatChannelCount(channel.format);
            mTensors[channel.name] = torch::from_blob(BaseMLPass::getDevicePtr(channel.name), {height, width, channelCount}, options);
            mTensors[channel.name] = processTensor(mTensors[channel.name]);
        }

        mOnes1 = torch::ones_like(mTensors[kInputColor].index({Slice(), Slice(0, 1), Slice(), Slice()})).set_requires_grad(false);
        mOnes3 = torch::ones_like(mTensors[kInputColor]).set_requires_grad(false);
    }
    else
    {
        std::cout << "CUDA is not available!" << std::endl;
        assert(false && "CUDA is not available!");
    }
}

void TorchPass::renderUI(Gui::Widgets &widget)
{
    widget.checkbox("Enabled Torch?", mEnabled);
    widget.checkbox("Enabled Inference?", mInference);
    widget.checkbox("Enabled Back-propagation?", mBackprop);
}
