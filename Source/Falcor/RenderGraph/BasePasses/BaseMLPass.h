#pragma once

#include "Core/Program/GraphicsProgram.h"

#include "Utils/CudaUtils.h"

#include <map>

namespace Falcor
{
    class dlldecl BaseMLPass
    {
        struct Interop
        {
            Buffer::SharedPtr buffer;  // Falcor buffer
            void *devicePtr = nullptr; // CUDA pointer to buffer
            Falcor::uint2 size;
        };

    protected:
        BaseMLPass();

        // Interface for Interop
        void addInterop(const std::string &name, uint32_t width, uint32_t height, ResourceFormat format);
        void *getDevicePtr(const char *name);
        void *getDevicePtr(const std::string &name);
        Buffer::SharedPtr getBuffer(const char *name);
        const Falcor::uint2 &getBufferSize(const char *name);
        void copyTextureToBuffer(RenderContext *pContext, const Texture::SharedPtr &tex, const char *name);
        void copyTextureToBuffer(RenderContext *pContext, const Texture::SharedPtr &tex, const std::string &name);
        void copyBufferToTexture(RenderContext *pContext, const char *name, const Texture::SharedPtr &tex);
        void copyBufferToTexture(RenderContext *pContext, const std::string &name, const Texture::SharedPtr &tex);

        void renderUI(Gui::Widgets &weidget);

    private:
        void *exportBufferToCudaDevice(Buffer::SharedPtr &buf);
        void convertTexToInterop(RenderContext *pContext, const Texture::SharedPtr &tex, const Interop &interop);
        void convertInteropToTex(RenderContext *pContext, const Interop &interop, const Texture::SharedPtr &tex);

        inline const Interop &getInterop(const std::string &name)
        {
            auto it = mInterops.find(name);
            if (it == mInterops.end())
            {
                std::cout << "[BaseMLPass.h] Interop " << name << " does not exist in the map." << std::endl;
                exit(-1);
            }
            return it->second;
        }

    protected:
        // Control
        bool mEnabled = true;
        bool mInference = true;
        bool mBackprop = false;

        bool mWriteFrames = false;
        int mNumWriteFrames = 1;

    private:
        ComputePass::SharedPtr mpConvertTexToBuf;
        FullScreenPass::SharedPtr mpConvertBufToTex;
        Fbo::SharedPtr mpFbo;

        /**
         * @brief A map holding the Interop structures
         */
        std::unordered_map<std::string, Interop> mInterops;
    };
}
