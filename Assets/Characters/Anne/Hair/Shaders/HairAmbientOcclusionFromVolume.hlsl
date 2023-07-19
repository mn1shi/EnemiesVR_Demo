#ifndef __HAIRAMBIENTOCCLUSIONFROMVOLUME_HLSL__
#define __HAIRAMBIENTOCCLUSIONFROMVOLUME_HLSL__


void CalculateApproximateAmbientOcclusion_float(
    in Texture3D densityTex, 
	in float3 uvw, 
	in uint strandIndex,
	in float4 occlusionParams,
    out float outOcclusion)
{
	float density = densityTex.SampleLevel(s_trilinear_clamp_sampler, uvw, 0).x;
	float multiplier = occlusionParams.y;
	float maxOcclusion = min(abs(occlusionParams.x), 1.f);

	float denom = density * multiplier;
	outOcclusion =  1.f - maxOcclusion + maxOcclusion * saturate(1.f / denom);;
	
}

#endif