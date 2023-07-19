#ifndef __TONGUESALIVAMESHEXTRUDE_HLSL__
#define __TONGUESALIVAMESHEXTRUDE_HLSL__

void ExtrudeTongueSaliveMesh_float(in float3 normal, in float3 tongueForwardDir, in float extrudeFactor, in float LengthShortenFactor, out float3 offsetOut)
{
    #if SHADERPASS == SHADERPASS_SHADOWS
    offsetOut = 0;
    #else
    offsetOut = normal * extrudeFactor - tongueForwardDir * LengthShortenFactor;
    #endif
    
}
#endif
