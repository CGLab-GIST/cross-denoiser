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
#pragma once
#include <torch/torch.h> // This should be on top

#include "Falcor.h"
#include "RenderGraph/BasePasses/BaseMLPass.h"

#include "Utils/CudaUtils.h"
#include "Model.h"
#include "ops.h"

using namespace Falcor;

class TorchPass : public RenderPass, public BaseMLPass
{
public:
    using SharedPtr = std::shared_ptr<TorchPass>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext *pRenderContext = nullptr, const Dictionary &dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData &compileData) override;
    virtual void compile(RenderContext *pRenderContext, const CompileData &compileData) override {}
    virtual void execute(RenderContext *pRenderContext, const RenderData &renderData) override;
    virtual void renderUI(Gui::Widgets &widget) override;
    virtual void setScene(RenderContext *pRenderContext, const Scene::SharedPtr &pScene) override {}
    virtual bool onMouseEvent(const MouseEvent &mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent &keyEvent) override { return false; }

private:
    void init(uint32_t width, uint32_t height);

private:
    TorchPass();

    uint32_t mWidth, mHeight;

    // Configurations
    const uint kRegressionWidth = 17;
    const uint kFilterWidth = 11;
    const uint kSparsity = 4;
    const float kLearningRate = 1e-3f;

    // Network
    MainNet mNetwork;
    std::shared_ptr<torch::optim::Adam> mOptimizer;

    // Buffers
    std::map<std::string, torch::Tensor> mTensors;

    // Temporal tensors
    torch::Tensor mOnes1, mOnes3;

    bool mEnabled = true;
    bool mInference = true;
    bool mBackprop = true;
};
