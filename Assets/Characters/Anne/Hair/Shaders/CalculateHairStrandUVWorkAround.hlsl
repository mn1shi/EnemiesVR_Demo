#ifndef __CALCULATEHAIRSTRANDUVFROMPARTICLEINDEX_HLSL__
#define __CALCULATEHAIRSTRANDUVFROMPARTICLEINDEX_HLSL__

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Assets/_Packages/com.unity.demoteam.hair/Runtime/HairVertex.hlsl"

void CalculateHairStrandUV_float(
    in float vertexFloat,
    in float3 position,
    out float2 uv)
{
    int vertexID = floor(vertexFloat);
    
    #if HAIR_VERTEX_ID_STRIPS
    uint linearParticleIndex = vertexID >> 1;
    #else
    uint linearParticleIndex = vertexID;
    #endif

    DECLARE_STRAND(linearParticleIndex / STRAND_PARTICLE_COUNT);
    const uint i = strandParticleBegin + (linearParticleIndex % STRAND_PARTICLE_COUNT) * strandParticleStride;
    const uint i_next = i + strandParticleStride;
    const uint i_tail = strandParticleEnd - strandParticleStride;

    float3 p0 = LoadPosition(i);
    float3 p1 = (i == i_tail)
        ? p0
        : LoadPosition(i_next);

    const float vPerSegment = 1.f/STRAND_PARTICLE_COUNT;
    float yVertex = float(linearParticleIndex % STRAND_PARTICLE_COUNT) * vPerSegment;
    float2 segment = (p1 - p0);
    float segmentLen = length(segment);
    float segmentToPosLen = length(position - p0);
    float segmentFraction = segmentLen == 0.f ? 0.f : saturate(segmentToPosLen / segmentLen) * vPerSegment;
    uv = float2(0.5f, yVertex + segmentFraction);
    
}

#endif