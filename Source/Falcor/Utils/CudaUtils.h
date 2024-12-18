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
#if _ENABLE_CUDA

#include <vector>

namespace Falcor
{
    typedef unsigned long long CUdeviceptr;

    struct dlldecl CUDAUtils
    {
        static unsigned int initCUDA(void);

        // This takes a Windows Handle (e.g., from Falcor::Resource::getApiHandle()) on a resource
        //    that has been declared "shared" [with Resource::BindFlags::Shared] plus a size of that
        //    resource in bytes and returns a CUDA device pointer from that resource that can be
        //    passed into CUDA.  This pointer is a value returned from
        //    cudaExternalMemoryGetMappedBuffer(), so should follow its rules (e.g., the docs claim
        //    you are responsible for calling cudaFree() on this pointer).
        static void *getSharedDevicePtr(HANDLE sharedHandle, uint32_t bytes);

        // Calls cudaFree() on the provided pointer;
        static bool freeSharedDevicePtr(void *ptr);
    };

    class CudaBuffer
    {
    public:
        CudaBuffer() {}

        CUdeviceptr getDevicePtr() { return (CUdeviceptr)mpDevicePtr; }
        size_t getSize() { return mSizeBytes; }

        void allocate(size_t size);
        void resize(size_t size);
        void free();

        template <typename T>
        void allocAndUpload(const std::vector<T> &vt);

        template <typename T>
        bool download(T *t, size_t count);

        template <typename T>
        bool upload(const T *t, size_t count);

    private:
        size_t mSizeBytes = 0;
        void *mpDevicePtr = nullptr;
    };
}

#endif
