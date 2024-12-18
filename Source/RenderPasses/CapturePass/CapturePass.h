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
#include "Falcor.h"

#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

class CapturePass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<CapturePass>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext *pRenderContext = nullptr, const Dictionary &dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    void parseDictionary(const Dictionary &dict);
    virtual RenderPassReflection reflect(const CompileData &compileData) override;
    virtual void compile(RenderContext *pContext, const CompileData &compileData) override {}
    virtual void execute(RenderContext *pRenderContext, const RenderData &renderData) override;
    virtual void renderUI(Gui::Widgets &widget) override;
    virtual void setScene(RenderContext *pRenderContext, const Scene::SharedPtr &pScene) override;
    virtual bool onMouseEvent(const MouseEvent &mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent &keyEvent) override { return false; }

private:
    CapturePass(const Dictionary &dict);
    void storeTexture(RenderContext *pRenderContext, const uint64_t frameID, const std::string &channelName, Texture::SharedPtr texture);

    std::vector<std::string> mChannels;
    std::vector<std::string> mIncludeAlpha;
    std::string mDirectory;

    Scene::SharedPtr mpScene;

    bool mStart = true;

    bool mAccumulateSubFrames = false;
    int mAccumulateCount = 16;
    int mCountSubFrame = 0;
    uint32_t mFramerate = 30;
    uint64_t mCaptureCount = 0; // global frame counter starts from 1

    // Variables for reference generation (loop sequence multiple times with different seed)
    bool mReferenceMode = false;
    uint32_t mRefStartFrame = 0;  // Start frame for reference generation
    uint32_t mRefEndFrame = 0;    // End frame for reference generation
    uint32_t mRefNumSamples = 0;  // Number of samples for reference generation
    uint32_t mRefSampleCount = 0; // Sample counter

    // Camera matrix
    bool mCaptureCameraMat = false;
    bool mCaptureCameraMatOnly = false;
    std::vector<glm::mat4> mCameraMatrices;
    std::vector<glm::mat4> mViewProjMatrices;
    std::vector<glm::mat4> mProjMatrices;
    std::vector<float3> mCameraPosition;
    std::vector<float3> mCameraTarget;
    std::vector<float3> mCameraUp;
    std::string mCameraMatTemplate;
};
