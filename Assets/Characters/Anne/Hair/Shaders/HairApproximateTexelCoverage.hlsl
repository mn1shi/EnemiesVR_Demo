#ifndef __HAIRAPPROXIMATECOVERAGE_HLSL__
#define __HAIRAPPROXIMATECOVERAGE_HLSL__

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Assets/_Packages/com.unity.demoteam.hair/Runtime/HairVertex.hlsl"

void CalculateApproximateCoverageWithDilation_float(
    in float3 positionOS,
    in float3 tangentOS,
    in float cameraDistance,
    in float camDistDilationMinDist,
    in float camDistDilationMultiplier,
    in float camDistDilationMax,
    in float width,
    in float height,
    out float coverage)
{
    #if SHADERPASS == SHADERPASS_SHADOWS
    coverage = 1.f;
    #else
    //TODO: instead of calculating the NDC width, could just approximate the size in screenspace, for example assuming point to be a sphere: https://iquilezles.org/articles/sphereproj/
    float halfWidth = _GroupMaxParticleDiameter * 0.5f;
    float4 a = TransformObjectToHClip(positionOS + tangentOS * halfWidth);
    float4 b = TransformObjectToHClip(positionOS - tangentOS * halfWidth);

    a.xyz /= a.w;
    b.xyz /= b.w;

    float2 delta = a.xy - b.xy;

    float dilationMultiplier = 1.f +  min(max(cameraDistance - camDistDilationMinDist, 0.f) * camDistDilationMultiplier, camDistDilationMax);

    delta *= float2(width, height) * 0.5f;

    coverage = saturate(length(delta) * dilationMultiplier);
    #endif
}


void CalculateApproximateCoverage_float(
    in float3 positionOS,
    in float3 tangentOS,
    in float width,
    in float height,
    out float coverage)
{
    CalculateApproximateCoverageWithDilation_float(positionOS, tangentOS, 0.f, 0.f, 0.f, 0.f,width, height, coverage);
}

#endif