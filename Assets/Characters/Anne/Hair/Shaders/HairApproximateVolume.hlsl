#ifndef __HAIRAPPROXIMATEVOLUME_HLSL__
#define __HAIRAPPROXIMATEVOLUME_HLSL__

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Random.hlsl"

SAMPLER (s_hair_approx_noise_point_repeat_sampler);

//copy pasted from HairSimComputeStrandCountProbe.hlsl because some of the functions conflict with whatever is included by the shadergraph.
#define HALF_SQRT_INV_PI2    0.5 * 0.56418958354775628694 
#define HALF_SQRT_3_DIV_PI2  0.5 * 0.97720502380583984317

float DecodeStrandCount2(float3 L, float4 strandCountProbe)
{
    float4 Ylm = float4(
        HALF_SQRT_INV_PI2,
        HALF_SQRT_3_DIV_PI2 * L.y,
        HALF_SQRT_3_DIV_PI2 * L.z,
        HALF_SQRT_3_DIV_PI2 * L.x
        );

    return abs(dot(strandCountProbe, Ylm));
}


void CalculateApproximateHairCoverage_float(
    in float3 in_WorldPos,
    in float4 in_StrandProbeCoeff,
    in float in_OpaqDensity,
    in float in_DitherStrengthStrandCenter,
    in float in_DitherStrengthStrandEdge,
    in float in_StrandIndex,
    in float2 in_StrandUV,
    in float2 in_StrandUVMultiplier,
    in UnityTexture2D noiseTex, 
    out float out_Coverage)
{
#if HAIR_VERTEX_DILATE_STRIPS && (SHADERPASS != SHADERPASS_SHADOWS)
    float4 clip = TransformWorldToHClip(in_WorldPos);
    float2 scrnCoord = ((clip.xy / clip.w) + 1.f) * _ScreenSize.xy * 0.5f;

    float3 towardsCameraDir = GetWorldSpaceNormalizeViewDir(in_WorldPos);
    
    float strandCountForward = DecodeStrandCount2(towardsCameraDir, in_StrandProbeCoeff);
    float strandCountBackwards = DecodeStrandCount2(-towardsCameraDir, in_StrandProbeCoeff);
    float strandCountTotal = strandCountForward + strandCountBackwards;

    float segmentCount = 16.f;
    float segmentV = frac(in_StrandUV.y * segmentCount);
    in_StrandUV.y = segmentV;

    //rotate noise around segment center
    float angle = (in_StrandIndex * PI * 0.125f) + (uint(_FrameCount) % 16u) / 15.f;
    //float angle = InterleavedGradientNoise(scrnCoord, (_FrameCount % 8) + in_StrandIndex);
    angle *= 2.f * PI;

    float2 rotation;
    sincos( angle, rotation.y, rotation.x );
    float2x2 rotationMatrix = { rotation.x, rotation.y, -rotation.y, rotation.x };

    float2 rotatedUV = in_StrandUV;
    //rotatedUV.x = 0.5f;
    rotatedUV -= 0.5f;
    rotatedUV = mul(rotationMatrix, rotatedUV);
    rotatedUV += 0.5f;
    rotatedUV *= in_StrandUVMultiplier;

    float noise = SAMPLE_TEXTURE2D_LOD(noiseTex,s_hair_approx_noise_point_repeat_sampler, rotatedUV, 0).x;
    float ditherStrength = lerp(in_DitherStrengthStrandCenter, in_DitherStrengthStrandEdge, saturate(abs(in_StrandUV.x - 0.5f) * 2.f));
    
    float dither = 1.f - ditherStrength + ditherStrength * noise;
    
    float alpha = dither * saturate(strandCountTotal / max(1e-6f, in_OpaqDensity));

    out_Coverage = alpha;
    #else
    out_Coverage = 1.f;
    #endif
}

#endif