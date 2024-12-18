import Params;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDData;

struct ResolvePass
{
    uint gSppId;
    RestirPathTracerParams params; ///< Runtime parameters.

    NRDBuffers outputNRD;             ///< Output NRD data.

    RWTexture2D<float4> outputNRDDiffuseRadianceHitDist;
    RWTexture2D<float4> outputNRDSpecularRadianceHitDist;
    RWTexture2D<float4> outputNRDDeltaReflectionRadianceHitDist;
    RWTexture2D<float4> outputNRDDeltaTransmissionRadianceHitDist;
    RWTexture2D<float4> outputNRDResidualRadianceHitDist;

    void execute(const uint2 pixel)
    {
        const uint spp = kSamplesPerPixel;
        const float invSpp = spp > 0 ? 1.f / spp : 1.f;

        // NRD data is always written per-sample and needs to be resolved.
        if (kOutputNRDData)
        {
            float3 diffuseRadiance = 0.f;
            float3 specularRadiance = 0.f;
            float3 deltaReflectionRadiance = 0.f;
            float3 deltaTransmissionRadiance = 0.f;
            float3 residualRadiance = 0.f;
            float hitDist = outputNRD.nrdHitDist[pixel];

            {
                const NRDRadiance radianceData = UnpackNRDRadiance(outputNRD.nrdRadiance[pixel]);
                const NRDPathType pathType = radianceData.getPathType();
                float3 radiance = radianceData.getRadiance();

                float3 reflectance = outputNRD.nrdReflectance[pixel].rgb;
                float3 emission = outputNRD.nrdEmission[pixel].rgb;
                // HJ: Here, we do not subtract the emission, because the emission is already extractred into direct lighting.
                // float3 demodulatedRadiance = max(0.f, (radiance - emission)) / reflectance;
                float3 demodulatedRadiance = max(0.f, (radiance)) / reflectance;

                switch (pathType)
                {
                case NRDPathType::Diffuse:
                    diffuseRadiance += demodulatedRadiance;
                    break;
                case NRDPathType::Specular:
                    specularRadiance += demodulatedRadiance;
                    break;
                case NRDPathType::DeltaReflection:
                    deltaReflectionRadiance += demodulatedRadiance;
                    break;
                case NRDPathType::DeltaTransmission:
                    deltaTransmissionRadiance += demodulatedRadiance;
                    break;
                default:
                    // Do not demodulate residual.
                    residualRadiance += radiance;
                    break;
                }

                if (gSppId == 0)
                {
                    outputNRDDiffuseRadianceHitDist[pixel] = float4(invSpp * diffuseRadiance, invSpp * hitDist);
                    outputNRDSpecularRadianceHitDist[pixel] = float4(invSpp * specularRadiance, invSpp * hitDist);
                    outputNRDDeltaReflectionRadianceHitDist[pixel] = float4(invSpp * deltaReflectionRadiance, 0.f);
                    outputNRDDeltaTransmissionRadianceHitDist[pixel] = float4(invSpp * deltaTransmissionRadiance, 0.f);
                    outputNRDResidualRadianceHitDist[pixel] = float4(invSpp * residualRadiance, invSpp * hitDist);
                }
                else
                {
                    outputNRDDiffuseRadianceHitDist[pixel] += float4(invSpp * diffuseRadiance, invSpp * hitDist);
                    outputNRDSpecularRadianceHitDist[pixel] += float4(invSpp * specularRadiance, invSpp * hitDist);
                    outputNRDDeltaReflectionRadianceHitDist[pixel] += float4(invSpp * deltaReflectionRadiance, 0.f);
                    outputNRDDeltaTransmissionRadianceHitDist[pixel] += float4(invSpp * deltaTransmissionRadiance, 0.f);
                    outputNRDResidualRadianceHitDist[pixel] += float4(invSpp * residualRadiance, invSpp * hitDist);
                }
            }
        }
    }
}

cbuffer CB
{
    ResolvePass gResolvePass;
    RestirPathTracerParams params; ///< Runtime parameters.
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gResolvePass.params.frameDim))
        return;

    gResolvePass.execute(pixel);
}
