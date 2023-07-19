#ifndef __HAIR_SHADOW_DISCARD_HLSL__
#define __HAIR_SHADOW_DISCARD_HLSL__

#define NAN (sqrt(-1))

SamplerState densitySampler_bilinear_clamp_sampler;

bool ShouldDiscard(uint primID, float density, float threshold, float weight)
{
    density *= weight;

    uint modiD = primID % 128;

    return float(modiD) < density;
}
//randomly discard segments based on weights. Different parameters for shadows and other passes
void DiscardPrim_float(in Texture3D densityTex, in float3 uvw, in float cullThresholdShadow, in float cullWeightShadow, in float2 cullThresholdMinMax, in float2 cullWeightMinMax, in float2 cameraDistanceMinMax, in float cameraDistance,  in float3 posOS, in float primID,  out float3 posOSOut)
{
    posOSOut = posOS;

    #if SHADERPASS == SHADERPASS_SHADOWS
    float threshold = cullThresholdShadow;
    float weight = cullWeightShadow;

    float density = densityTex.SampleLevel(densitySampler_bilinear_clamp_sampler, uvw, 0).x;
    if(density > threshold)
    {
        if(ShouldDiscard(primID, density, threshold, weight))
            posOSOut.x = NAN;

    }
    #else
    float t = smoothstep(cameraDistanceMinMax.x, cameraDistanceMinMax.y, cameraDistance);
    float threshold = lerp(cullThresholdMinMax.x, cullThresholdMinMax.y, t);
    float weight = lerp(cullWeightMinMax.x, cullWeightMinMax.y, t);
    if(weight > 0.f)
    {
        float density = densityTex.SampleLevel(densitySampler_bilinear_clamp_sampler, uvw, 0).x;
        if(density > threshold)
        {
            if(ShouldDiscard(primID, density, threshold, weight))
                posOSOut.x = NAN;

        }
    }
    
    #endif
}

#endif//__HAIR_SHADOW_DISCARD_HLSL__
