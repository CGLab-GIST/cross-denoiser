#include "stdafx.h"
#include "BaseMLPass.h"

namespace
{
    const std::string kConvertTexToBufFile = "RenderGraph/BasePasses/ConvertTexToBuf.cs.slang";
    const std::string kConvertBufToTexFile = "RenderGraph/BasePasses/ConvertBufToTex.ps.slang";
}

namespace Falcor
{
    BaseMLPass::BaseMLPass()
    {
        mpConvertTexToBuf = ComputePass::create(kConvertTexToBufFile, "main");
        mpConvertBufToTex = FullScreenPass::create(kConvertBufToTexFile);
        mpFbo = Fbo::create();
    }

    void BaseMLPass::addInterop(const std::string &name, uint32_t width, uint32_t height, ResourceFormat format)
    {
        Interop interop;
        interop.size = Falcor::uint2(width, height);
        interop.buffer = Buffer::createTyped(format,
                                             width * height,
                                             Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget | Resource::BindFlags::Shared);
        interop.devicePtr = exportBufferToCudaDevice(interop.buffer);
        mInterops[name] = interop;
    }

    void *BaseMLPass::getDevicePtr(const char *name)
    {
        return getInterop(name).devicePtr;
    }

    void *BaseMLPass::getDevicePtr(const std::string &name)
    {
        return getInterop(name).devicePtr;
    }

    Buffer::SharedPtr BaseMLPass::getBuffer(const char *name)
    {
        return getInterop(name).buffer;
    }

    const Falcor::uint2 &BaseMLPass::getBufferSize(const char *name)
    {
        return getInterop(name).size;
    }

    void *BaseMLPass::exportBufferToCudaDevice(Buffer::SharedPtr &buf)
    {
        if (buf == nullptr)
            return nullptr;
        return CUDAUtils::getSharedDevicePtr(buf->getSharedApiHandle(), (uint32_t)buf->getSize());
    }

    void BaseMLPass::convertTexToInterop(RenderContext *pContext, const Texture::SharedPtr &tex, const Interop &interop)
    {
        auto vars = mpConvertTexToBuf->getVars();
        vars["GlobalCB"]["gStride"] = interop.size.x;
        vars["gInTex"] = tex;
        vars["gOutBuf"] = interop.buffer;
        mpConvertTexToBuf->execute(pContext, interop.size.x, interop.size.y);
    }

    void BaseMLPass::convertInteropToTex(RenderContext *pContext, const Interop &interop, const Texture::SharedPtr &tex)
    {
        auto vars = mpConvertBufToTex->getVars();
        vars["GlobalCB"]["gStride"] = interop.size.x;
        vars["gInBuf"] = interop.buffer;
        mpFbo->attachColorTarget(tex, 0);
        mpConvertBufToTex->execute(pContext, mpFbo);
    }

    void BaseMLPass::copyTextureToBuffer(RenderContext *pContext, const Texture::SharedPtr &tex, const char *name)
    {
        convertTexToInterop(pContext, tex, getInterop(name));
    }

    void BaseMLPass::copyTextureToBuffer(RenderContext *pContext, const Texture::SharedPtr &tex, const std::string& name)
    {
        convertTexToInterop(pContext, tex, getInterop(name));
    }

    void BaseMLPass::copyBufferToTexture(RenderContext *pContext, const char *name, const Texture::SharedPtr &tex)
    {
        convertInteropToTex(pContext, getInterop(name), tex);
    }

    void BaseMLPass::copyBufferToTexture(RenderContext *pContext, const std::string& name, const Texture::SharedPtr &tex)
    {
        convertInteropToTex(pContext, getInterop(name), tex);
    }

    void BaseMLPass::renderUI(Gui::Widgets &widget)
    {
        widget.checkbox("Enabled", mEnabled);
        if (mEnabled)
        {
            widget.checkbox("Enabled Inference", mInference);
            widget.checkbox("Enabled Back-propagation", mBackprop);
        }
        widget.checkbox("Write frames (path tracing)", mWriteFrames);
        widget.var("Number of frames", mNumWriteFrames, 0);
    }
}
