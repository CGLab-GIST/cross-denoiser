/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Scene.Raytracing;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import PathTracer;
import PathState;
import Params;
import PathReservoir;
import PathTracerNRD;

ParameterBlock<PathTracer> gPathTracer;

cbuffer CB
{
    int gSampleId;
}

void tracePath(uint pathID, inout PathReservoir giReservoir, uint sampleIdx)
{
    PathState path;
    gPathTracer.generatePath(pathID, path);

    printSetPixel(path.getPixel());
    logSetPixel(path.getPixel());

    while (path.isActive())
    {
        if (path.isHit())
        {
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            gPathTracer.handleHit(path);
#endif
            if (path.isTerminated()) break;
            gPathTracer.nextVertex(path);
        }
        else
        {
            gPathTracer.handleMiss(path);
        }
    }

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
    gPathTracer.finalize(path);
    gPathTracer.writeOutput(path, giReservoir, sampleIdx);
#endif
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    // Reset buffer
    gPathTracer.outputAlbedo[pixel] = float4(0.0f);

    // Skip pixel if there is no hit in the vbuffer.
    HitInfo hit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!hit.isValid()) return;

    PathReservoir giReservoir;

    static const uint itersPerShaderPass = PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing ? kSamplesPerPixel : kCandidateSamples;

    if (itersPerShaderPass == 1)
    {
        // Handle fixed 1 spp case.
        uint pathID = pixel.x | (pixel.y << 12) | (gSampleId << 24);
        tracePath(pathID, giReservoir, 0);
    }
    else//(itersPerShaderPass > 1)
    {
        // Handle fixed multiple spp case.
        for (uint sampleIdx = 0; sampleIdx < itersPerShaderPass; ++sampleIdx)
        {
            uint pathID = pixel.x | (pixel.y << 12) | (sampleIdx << 24);
            tracePath(pathID, giReservoir, sampleIdx);
        }
    }
}
