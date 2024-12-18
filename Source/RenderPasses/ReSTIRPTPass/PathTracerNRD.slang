/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/Material/MaterialDefines.slangh"

import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.ClothMaterial;
import Rendering.Materials.HairMaterial;
import Rendering.Materials.MaterialShading;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Volumes.PhaseFunction;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDHelpers;
import RenderPasses.Shared.Denoising.NRDConstants;
import NRDHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import PathBuilder;
import PathReservoir;

import PathTracer;

#include "Utils/Math/MathConstants.slangh"

__exported import PathState;
__exported import Params;

/** Extends the PathTracer with variants of handleHit() to trace NRD guide paths.
*/
extension PathTracer
{
    /** Handle hit on delta reflection materials.
        After handling the hit, the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaReflectionHit(inout PathState path)
    {
        TriangleHit th = path.hit.getTriangleHit();

        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Load vertex/material attributes. This is a long latency operation.
        ShadingData sd;

        if (kPrimaryLodMode == TexLODMode::RayDiffs && path.length == 0 && path.hit.getType() == HitType::Triangle)
        {
            // Filtered lookups at primary hit
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            sd = loadShadingDataFootprint(path.hit, -path.dir, true, ddx, ddy);
        }
        else
        {
            float lod = path.length > 0 ? params.lodBias : 0.f;
            sd = loadShadingData(path.hit, -path.dir, path.length == 0, lod);
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        // Check if this is the primary hit.
        const bool isPrimaryHit = path.length == 0;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        uint lobes = getBSDFLobes(sd);

        const MaterialType materialType = sd.getMaterialType();
        const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;

        if (isPrimaryHit)
        {
            // Terminate without the write-out if the path doesn't start as delta reflection.
            if (!hasDeltaLobes)
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
                path.terminate();
                return;
            }

            // Add primary ray length to the path length.
            float primaryHitDist = length(sd.posW - path.origin);
            path.sceneLength += float16_t(primaryHitDist);
            // Hijack pdf that we don't need.
            path.pdf += primaryHitDist;
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * sd.emissive;

            // Terminate after scatter ray on last vertex has been processed or non-delta lobe exists.
            const bool lastVertex = hasFinishedSurfaceBounces(path);
            const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;
            const bool isEmissive = any(sd.emissive > 0.f);

            if (lastVertex || hasNonDeltaLobes || isEmissive)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, sd.N, sd.linearRoughness, float(path.sceneLength), hitDist);

                path.terminate();
                return;
            }
        }

        // For glass in reflections, force guide paths to always follow transmission/reflection based on albedos.
        // This is pretty hacky but works best our of the possible options.
        // Stable guide buffers are a necessity.
        const MaterialData md = gScene.getMaterial(sd.materialID);
        bool isTransmissive = md.specularTransmission > 0.f;
        if (isTransmissive && (1.f - sd.specularTransmission <= sd.specularTransmission))
        {
            sd.setActiveLobes((uint)LobeType::DeltaTransmission);
        }
        else
        {
            sd.setActiveLobes((uint)LobeType::DeltaReflection);
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Hijack pdf that we don't need.
        float primaryHitDist = path.pdf;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, path);

        path.pdf = primaryHitDist;

        if (!valid)
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            path.terminate();
            return;
        }
    }

    /** Handle hit on delta transmission materials.
        After handling the hit, a new scatter (delta transmission only) ray is generated or the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaTransmissionHit(inout PathState path)
    {
        TriangleHit th = path.hit.getTriangleHit();

        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Load vertex/material attributes. This is a long latency operation.
        ShadingData sd;

        if (kPrimaryLodMode == TexLODMode::RayDiffs && path.length == 0 && path.hit.getType() == HitType::Triangle)
        {
            // Filtered lookups at primary hit
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            sd = loadShadingDataFootprint(path.hit, -path.dir, true, ddx, ddy);
        }
        else
        {
            float lod = path.length > 0 ? params.lodBias : 0.f;
            sd = loadShadingData(path.hit, -path.dir, path.length == 0, lod);
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        const bool isPrimaryHit = path.length == 0;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        uint lobes = getBSDFLobes(sd);

        const bool hasDeltaTransmissionLobe = ((lobes & (uint)LobeType::DeltaTransmission) != 0);

        if (isPrimaryHit && !hasDeltaTransmissionLobe)
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, 0.f);
            path.terminate();
            return;
        }

        if (isPrimaryHit)
        {
            // Add primary ray length to the path length.
            path.sceneLength += float16_t(length(sd.posW - path.origin));
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * sd.emissive;
        }

        // Terminate the delta transmission path.
        const bool lastVertex = hasFinishedSurfaceBounces(path);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Fetch volume absorption from the material. This field only exist in basic materials for now.
        bool semiOpaque = false;
        const MaterialData md = gScene.getMaterial(sd.materialID);
        semiOpaque = any(md.volumeAbsorption > 100.f);

        const MaterialType materialType = sd.getMaterialType();
        const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;

        if (lastVertex || semiOpaque)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd), emission, sd.N, sd.linearRoughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Set the active lobes only to delta transmission.
        sd.setActiveLobes((uint)LobeType::DeltaTransmission);

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, path);

        // Delta transmission was not possible, fallback to delta reflection if it's allowed.
        if (!valid && isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
        {
            sd.setActiveLobes((uint)LobeType::DeltaTransmission | (uint)LobeType::DeltaReflection);
            valid = generateScatterRay(sd, path);
        }

        if (!valid)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd), emission, sd.N, sd.linearRoughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }
    }
}
