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
import Rendering.Materials.Microfacet;
import Rendering.Materials.MaterialShading;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Scene.ShadingData;
import Scene.Material.MaterialData;
import PathState;
import Utils.Math.MathHelpers;

void setNRDPrimaryHitEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const float3 emission)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        if (useNRDDemodulation)
        {
            outputNRD.primaryHitEmission[pixel] = float4(emission, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitEmission[pixel] = 0.f;
        }
    }
}

void setNRDPrimaryHitReflectance(NRDBuffers outputNRD, RWTexture2D<float4> outAlbedo, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const ShadingData sd)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        if (useNRDDemodulation)
        {
            float3 diffuseReflectionAlbedo = (1.f - sd.diffuseTransmission) * (1.f - sd.specularTransmission) * sd.diffuse;
            outputNRD.primaryHitDiffuseReflectance[pixel] = float4(diffuseReflectionAlbedo, 1.f);
            const float NdotV = saturate(dot(sd.N, sd.V));
            const float ggxAlpha = sd.linearRoughness * sd.linearRoughness;
            float3 specularReflectionAlbedo = (1.f - sd.specularTransmission) * sd.specular;
            const float3 specularReflectance = approxSpecularIntegralGGX(specularReflectionAlbedo, ggxAlpha, NdotV);
            outputNRD.primaryHitSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
            float3 albedo = diffuseReflectionAlbedo + specularReflectance;
            // Detect perfect glass material (roughness=0, diffuse=1)
            if (sd.linearRoughness == 0.f && all(sd.diffuse == float3(1.f)))
            {
                albedo = float3(1.f);
            }
            outAlbedo[pixel] = float4(albedo, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitDiffuseReflectance[pixel] = 1.f;
            outputNRD.primaryHitSpecularReflectance[pixel] = 1.f;
            outAlbedo[pixel] = float4(1.f);
        }
    }
}

void setNRDSampleHitDist(NRDBuffers outputNRD, inout PathState path)
{
    if (path.length == 1)
    {
        outputNRD.nrdHitDist[path.getPixel()] = float(path.sceneLength);
    }
}

void setNRDSampleEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, inout PathState path, const bool isPrimaryHit, const float3 emission, const bool wasDeltaOnlyPathBeforeScattering)
{
    const uint2 pixel = path.getPixel();
    if (useNRDDemodulation)
    {
        // Always demodulate emission on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            outputNRD.nrdEmission[pixel] = float4(emission, 1.f);
        }
        // Additionally, accumulate emission along the delta path.
        else if ((path.isDeltaReflectionPrimaryHit() || path.isDeltaTransmissionPath()) && any(emission > 0.f))
        {
            const bool demodulateDeltaReflectionEmission = path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering;
            const bool demodulateDeltaTransmissionEmission = path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering;
            if (demodulateDeltaReflectionEmission || demodulateDeltaTransmissionEmission)
            {
                float3 prevEmission = outputNRD.nrdEmission[pixel].rgb;
                outputNRD.nrdEmission[pixel] = float4(prevEmission + emission, 1.f);
            }
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.nrdEmission[pixel] = float4(0.f);
    }
}

void setNRDSampleReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, inout PathState path, const bool isPrimaryHit, const ShadingData sd, const uint lobes, const bool wasDeltaOnlyPathBeforeScattering)
{
    const uint2 pixel = path.getPixel();
    if (useNRDDemodulation)
    {
        const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Always demodulate reflectance from diffuse and specular paths on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            if (path.isDiffusePrimaryHit())
            {
                float3 diffuseReflectionAlbedo = max(kNRDMinReflectance, (1.f - sd.diffuseTransmission) * (1.f - sd.specularTransmission) * sd.diffuse);
                outputNRD.nrdReflectance[pixel].rgb = diffuseReflectionAlbedo;
            }
            else if (path.isSpecularPrimaryHit())
            {
                const float NdotV = saturate(dot(sd.N, sd.V));
                const float ggxAlpha = sd.linearRoughness * sd.linearRoughness;
                float3 specularReflectionAlbedo = (1.f - sd.specularTransmission) * sd.specular;
                const float3 specularReflectance = max(kNRDMinReflectance, approxSpecularIntegralGGX(specularReflectionAlbedo, ggxAlpha, NdotV));
                outputNRD.nrdReflectance[pixel].rgb = specularReflectance;
            }
            else
            {
                outputNRD.nrdReflectance[pixel] = float4(1.f);
            }
        }
        // Demodulate reflectance from the second vertex along delta reflection paths.
        else if (path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.getMaterialType();
            const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd);

            outputNRD.nrdReflectance[pixel] = float4(deltaReflectance, 1.f);
        }
        // Demodulate reflectance from the first non-delta vertex along delta transmission paths.
        else if (path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.getMaterialType();
            const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd);

            outputNRD.nrdReflectance[pixel] = float4(deltaReflectance, 1.f);
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.nrdReflectance[pixel] = float4(1.f);
    }
}

float3 computeSampleReflectance(const bool useNRDDemodulation, bool isDiffusePrimaryHit, bool isSpecularPrimaryHit, const ShadingData sd)
{
    if (useNRDDemodulation)
    {
        float3 sampleReflectance;
        if (isDiffusePrimaryHit)
        {
            float3 diffuseReflectionAlbedo = (1.f - sd.diffuseTransmission) * (1.f - sd.specularTransmission) * sd.diffuse;
            sampleReflectance = diffuseReflectionAlbedo;
        }
        else if (isSpecularPrimaryHit)
        {
            const float NdotV = saturate(dot(sd.N, sd.V));
            const float ggxAlpha = sd.linearRoughness * sd.linearRoughness;
            float3 specularReflectionAlbedo = (1.f - sd.specularTransmission) * sd.specular;
            const float3 specularReflectance = approxSpecularIntegralGGX(specularReflectionAlbedo, ggxAlpha, NdotV);
            sampleReflectance = specularReflectance;
        }
        else
        {
            sampleReflectance = 1.f;
        }
        return sampleReflectance;
    }
    else
    {
        return 1.f;
    }
}

/** Write out delta reflection guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaReflectionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float hitDist)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaReflectionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaReflectionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaReflectionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaReflectionPathLength[pixelPos] = pathLength;
    outputNRD.deltaReflectionHitDist[pixelPos] = hitDist;
}

/** Write out delta transmission guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaTransmissionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float3 posW)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaTransmissionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaTransmissionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaTransmissionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaTransmissionPathLength[pixelPos] = pathLength;
    outputNRD.deltaTransmissionPosW[pixelPos] = float4(posW, 0.f);
}
