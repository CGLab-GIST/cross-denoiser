/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Params;
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import LoadShadingData;
import Rendering.Materials.MaterialShading;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import NRDHelpers;

struct PathReusePass
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    // Resources
    RestirPathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture1D<float2> neighborOffsets;

    ByteAddressBuffer nRooksPattern;

    NRDBuffers outputNRD;                                 ///< Output NRD data.

    RWTexture2D<float4> outputColor;                      ///< Output resolved color.
    RWTexture2D<float4> outputColor2;                     ///< Output resolved color (dual).
    RWTexture2D<float4> outputIndirect;                      ///< Output resolved color.
    StructuredBuffer<PathReservoir> outputReservoirs;     // reservoir from previous pass
    RWStructuredBuffer<PathReservoir> temporalReservoirs; // resulting reservoir for next frame
    StructuredBuffer<PixelReconnectionData> reconnectionDataBuffer;
    StructuredBuffer<PathReuseMISWeight> misWeightBuffer;

    RWTexture2D<float4> outputNRDDiffuseRadianceHitDist;    ///< Output resolved diffuse color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDSpecularRadianceHitDist;   ///< Output resolved specular color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    RWTexture2D<float4> outputNRDResidualRadianceHitDist;   ///< Output resolved residual color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    Texture2D<float4> primaryHitEmission;                     ///< Output per-pixel primary hit emission. Only valid if kOutputNRDData == true.

    int gSpatialRoundId;
    int  gNumSpatialRounds;
    uint gSpatialReusePattern;
    // for default pattern
    int gNeighborCount;
    float gGatherRadius;
    // for small window
    int gSmallWindowRadius;

    bool gIsLastRound;

    bool gFeatureBasedRejection;

    uint gSppId;

    bool fixSpatialSeed;
    uint spatialSeed;
    bool fixTemporalSeed;
    uint temporalSeed;
    uint temporalSeedOffset;

    uint gTileWidth;
    bool useTileSeed;

    Texture2D<float4> directLighting;
    Texture2D<float4> directLighting2;
    bool useDirectLighting;

    bool isValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    bool isValidGeometry(ShadingData centralSd, ShadingData neighborSd)
    {
        if (!gFeatureBasedRejection) return true;
        float3 camPos = gScene.camera.data.posW;
        float centralDist = distance(camPos, centralSd.posW);
        float neighborDist = distance(camPos, neighborSd.posW);
        return dot(centralSd.N, neighborSd.N) >= 0.5f && abs(centralDist - neighborDist) < 0.1f * centralDist;
    }

    ShadingData getPixelShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);

            TriangleHit th = PrimaryHit.getTriangleHit();
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    int getNeighborCount()
    {
        if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathReuse) return 16; //include self
        else
        {
            if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Default)
            {
                return gNeighborCount; // does not include self
            }
            else if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::SmallWindow)
            {
                int smallWindowDiameter = 2 * gSmallWindowRadius + 1;
                return smallWindowDiameter * smallWindowDiameter; // count self, but not use self as neighbor
            }
            else if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Tile)
            {
                return gNeighborCount;
            }
        }
        return 0;
    }

    int2 getPathReuseNextNeighborPixel(int4 NRookQuery, int2 pixel, int i)
    {
        //
        if (PathReusePattern(gSpatialReusePattern) == PathReusePattern::Block)
        {
            int2 block = pixel / 4;
            return 4 * block + int2(i % 4, i / 4);
        }
        else
        {
            // fetch neighbors
            int groupId = NRookQuery.z;
            uint2 words = nRooksPattern.Load2(NRookQuery.w * 256 + groupId * 8);
            int rowId = (words[i / 8] >> (4 * (i % 8))) & 0xF;

            return int2(i, rowId) - NRookQuery.xy + pixel;
        }
    }

    int2 getNextNeighborPixel(const uint startIndex, int2 pixel, int i)
    {
        int2 neighborPixel = int2(0,0);

        if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Default)
        {
            uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
            neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gGatherRadius);
        }
        else if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::SmallWindow)
        {
            int smallWindowDiameter = 2 * gSmallWindowRadius + 1;
            neighborPixel = pixel + int2(-gSmallWindowRadius + (i % smallWindowDiameter),
                -gSmallWindowRadius + (i / smallWindowDiameter));
            if (all(neighborPixel == pixel)) neighborPixel = int2(-1);
        }
        else if (SpatialReusePattern(gSpatialReusePattern) == SpatialReusePattern::Tile)
        {
            uint2 tileId = pixel / gTileWidth;
            uint2 tileOffset = tileId * gTileWidth;
            uint2 tileDim = min(params.frameDim - tileOffset, gTileWidth);
            uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
            neighborPixel = tileOffset + uint2(abs(neighborOffsets[neighborIndex]) * tileDim);
            if (all(neighborPixel == pixel)) neighborPixel = int2(-1);
        }

        return neighborPixel;
    }

    void ReSTIR(const uint2 pixel)
    {
        uint2 tileSeed = pixel % gTileWidth;

        uint2 spatialSeed2 = fixSpatialSeed ? spatialSeed : (useTileSeed ? tileSeed : pixel);
        uint temporalSeed2 = fixTemporalSeed ? temporalSeed : (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples + 1 + gSpatialRoundId;

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint centralOffset = params.getReservoirOffset(pixel);
        PackedHitInfo centralPrimaryHitPacked;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);

        if (!isValidPackedHitInfo(centralPrimaryHitPacked)) return;

        //
        // Reuse the path
        var sg = TinyUniformSampleGenerator(spatialSeed2, temporalSeedOffset + temporalSeed2);
        PathReservoir centerReservoir = outputReservoirs[centralOffset];
        var dstReservoir = reusePath(0, centerReservoir, pixel, centralOffset, sg, centralPrimaryHitPacked, centralPrimarySd);

        // Store the resulting reservoir for the next frame
        if (PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathReuse)
            temporalReservoirs[centralOffset] = dstReservoir;

        // Calculate color
        float3 color = dstReservoir.F * dstReservoir.weight;
        if (any(isnan(color) || isinf(color) || color < 0.f)) color = 0.f;
        if (gIsLastRound)
        {
            outputIndirect[pixel] += float4(color / kSamplesPerPixel, 1.f);

            if (useDirectLighting)
            {
                if (gSppId % 2 == 0)
                    color += directLighting[pixel].rgb;
                else
                    color += directLighting2[pixel].rgb;
                if (any(isnan(color) || isinf(color))) color = 0.f;
            }

            int dvider = (kSamplesPerPixel == 1) ? 1 : kSamplesPerPixel / 2;
            if (gSppId % 2 == 0)
            {
                outputColor[pixel] += float4(color / dvider, 1.f);
            }
            else
            {
                outputColor2[pixel] += float4(color / dvider, 1.f);
            }

            // HJ: Developed for NRD 3.1.0 from Falcor 7
            if (kOutputNRDData && PathSamplingMode(kPathSamplingMode) == PathSamplingMode::ReSTIR)
            {
                float3 radiance = dstReservoir.F * dstReservoir.weight;
                if (any(isnan(radiance) || isinf(radiance) || radiance < 0.f)) radiance = 0.f;

                //// Direct lighting should be denosied separately in NRD
                // if (useDirectLighting)
                //     radiance += directLighting[pixel].rgb;

                // HJ: We set the radiance only. Others (e.g., primary path type) are to be set by PathTracer.slang
                NRDRadiance data = UnpackNRDRadiance(outputNRD.nrdRadiance[pixel]);
                data.setRadiance(radiance);
                outputNRD.nrdRadiance[pixel] = PackNRDRadiance(data);
            }
        }
    }

    PathReservoir reusePath(
        //
        const uint id,
        const in PathReservoir centralReservoir,
        //
        const uint2 pixel, const uint centralOffset,
        inout TinyUniformSampleGenerator sg,
        PackedHitInfo centralPrimaryHitPacked, ShadingData centralPrimarySd)
    {
        PathReservoir dstReservoir = centralReservoir;

        float3 initialColor = dstReservoir.F * dstReservoir.weight;

        float centralM = dstReservoir.M;

        if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathReuse ||
            (ReSTIRMISKind(kSpatialReSTIRMISKind) == ReSTIRMISKind::Talbot || ReSTIRMISKind(kSpatialReSTIRMISKind) == ReSTIRMISKind::Pairwise))
        {
            dstReservoir.init();
        }
        else
        {
            dstReservoir.prepareMerging();
        }

        ReconnectionData dummyRcData;
        dummyRcData.Init();

        int neighborCount = getNeighborCount();
        const uint startIndex = sampleNext1D(sg) * kNeighborOffsetCount;

        /////////////////////
        /// PAIRWISE RMIS ///
        /////////////////////
        if (ReSTIRMISKind(kSpatialReSTIRMISKind) == ReSTIRMISKind::Pairwise)
        {
            int validNeighborCount = 0;
            float canonicalWeight = 1;

            for (int i = 0; i < neighborCount; ++i)
            {
                int2 neighborPixel = getNextNeighborPixel(startIndex, pixel, i);

                if (!isValidScreenRegion(neighborPixel)) continue;

                PackedHitInfo neighborPrimaryHitPacked;
                ShadingData neighborPrimarySd = getPixelShadingData(neighborPixel, neighborPrimaryHitPacked);
                if (!isValidPackedHitInfo(neighborPrimaryHitPacked)) continue;
                if (!isValidGeometry(centralPrimarySd, neighborPrimarySd)) continue;

                PathReservoir neighborReservoir = getOutputReservoir(id, params.getReservoirOffset(neighborPixel));

                float dstJacobian;

                validNeighborCount++;

                float prefix_approxPdf = 0.f;

                float prefixJacobian;

                ReconnectionData rcData;
                if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && centralReservoir.pathFlags.rcVertexLength() > 1)
                {
                    rcData = getReconnectionData(id, centralOffset, 2 * i);
                }
                else
                    rcData = dummyRcData;

                float3 prefixIntegrand = computeShiftedIntegrand(params, prefixJacobian, neighborPrimaryHitPacked, neighborPrimarySd,
                    centralPrimarySd, centralReservoir, rcData, true);

                prefix_approxPdf = PathReservoir::computeWeight(prefixIntegrand) * prefixJacobian;

                canonicalWeight += 1;
                if (prefix_approxPdf > 0.f)
                    canonicalWeight -= prefix_approxPdf * neighborReservoir.M / (prefix_approxPdf * neighborReservoir.M + centralReservoir.M * PathReservoir::computeWeight(centralReservoir.F) / (neighborCount));

                PathReservoir tempDstReservoir = dstReservoir;

                if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && neighborReservoir.pathFlags.rcVertexLength() > 1)
                {
                    rcData = getReconnectionData(id, centralOffset, 2 * i + 1);
                }
                else
                    rcData = dummyRcData;

                bool possibleToBeSelected = shiftAndMergeReservoir(params, false, dstJacobian, centralPrimaryHitPacked, centralPrimarySd, tempDstReservoir,
                    neighborPrimarySd, neighborReservoir, rcData, true, sg, true, 1.f, true); // "true" means hypothetically selected as the sample

                float neighborWeight = 0.f;

                if (possibleToBeSelected)
                {
                    neighborWeight = PathReservoir::computeWeight(neighborReservoir.F) / dstJacobian * neighborReservoir.M
                        / ((PathReservoir::computeWeight(neighborReservoir.F) / dstJacobian) * neighborReservoir.M + PathReservoir::computeWeight(tempDstReservoir.F) * centralReservoir.M / (neighborCount));
                    if (isnan(neighborWeight) || isinf(neighborWeight)) neighborWeight = 0.f;
                }

                mergeReservoirWithResamplingMIS(params, tempDstReservoir.F, dstJacobian, dstReservoir, tempDstReservoir, neighborReservoir, sg, true, neighborWeight);
            }

            mergeReservoirWithResamplingMIS(params, centralReservoir.F, 1.f, dstReservoir, centralReservoir, centralReservoir, sg, true, canonicalWeight);

            if (dstReservoir.weight > 0)
            {
                dstReservoir.finalizeGRIS();
                dstReservoir.weight /= (validNeighborCount + 1); // compensate for the fact that pairwise resampling MIS was not divided by (k+1)
            }
        }

        if (dstReservoir.weight < 0.f) dstReservoir.weight = 0.f;
        if (isnan(dstReservoir.weight) || isinf(dstReservoir.weight)) dstReservoir.weight = 0.f;

        return dstReservoir;
    }

    PathReservoir getOutputReservoir(const uint id, const uint offset)
    {
        switch (id)
        {
        case 0:
            return outputReservoirs[offset];
        }
        PathReservoir emptyReservoir;
        return emptyReservoir;
    }

    ReconnectionData getReconnectionData(const uint id, const uint offset, const uint i)
    {
        switch (id)
        {
        case 0:
            return reconnectionDataBuffer[offset].data[i];
        }
        ReconnectionData emptyRcData;
        return emptyRcData;
    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
};

cbuffer CB
{
    PathReusePass gPathReusePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathReusePass.execute(dispatchThreadId.xy);
}
