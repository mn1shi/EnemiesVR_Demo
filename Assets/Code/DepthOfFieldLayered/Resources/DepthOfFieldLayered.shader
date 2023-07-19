Shader "Hidden/Shader/DepthOfFieldLayered"
{
    HLSLINCLUDE

    #pragma target 4.5
    #pragma only_renderers d3d11 playstation xboxone xboxseries vulkan metal switch

    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
    #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/FXAA.hlsl"
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/RTUpscale.hlsl"

    struct Attributes
    {
        uint vertexID : SV_VertexID;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct Varyings
    {
        float4 positionCS : SV_POSITION;
        float2 texcoord   : TEXCOORD0;
        UNITY_VERTEX_OUTPUT_STEREO
    };

    Varyings Vert(Attributes input)
    {
        Varyings output;
        UNITY_SETUP_INSTANCE_ID(input);
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
        output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
        output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);
        return output;
    }

    float _Intensity;
    float2 _ResolutionScale;
    TEXTURE2D_X(_InputTexture);
    TEXTURE2D_X(_DoFLayerTextureBlurred);
    bool _PostScreenSizeMatch;

    float4 CompositeDoFLayer(Varyings input) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

        uint2 positionSS = input.texcoord * _PostProcessScreenSize.xy;
        float3 main = LOAD_TEXTURE2D_X(_InputTexture, positionSS).xyz;

        // Remove the taa jitter, since this layer doesn't go through taa resolve (it doesn't have to, it always 
        // has some minimum blur, and fire and smoke don't have sharp features anyway)
        float2 uv = input.texcoord - _TaaJitterStrength.zw;
        
        // Clamp uvs to 1 pixel off the edge, since we might be in a bigger texture, and we don't want to pull in garbage with linear sampling.
        // We need linear sampling, since uvs have sub-pixel jitter offset now.
        uv = min(uv, 1 - _PostProcessScreenSize.zw);

        float3 layer = 0;
        // TODO: this is a hack. Resolution scale should be sufficient to make the first case work as well
        // (_PostScreenSizeMatch is true with FSR, CAS or no upscaling), but something doesn't work well when aspect ratio changes.
        if (_PostScreenSizeMatch)
            layer = LOAD_TEXTURE2D_X(_DoFLayerTextureBlurred, uv * _PostProcessScreenSize.xy).xyz;
        else
            layer = SAMPLE_TEXTURE2D_X_LOD(_DoFLayerTextureBlurred, s_linear_clamp_sampler, uv * _ResolutionScale, 0.0).xyz;

        return float4(main + layer * _Intensity, 1);
    }

    float4 BlitCustomPassColor(Varyings input) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

        uint2 positionSS = input.texcoord * _ScreenSize.xy;
        return LoadCustomColor(positionSS);
    }

    ENDHLSL

    SubShader
    {
        Pass
        {
            Name "LayeredDepthOfField"

            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment CompositeDoFLayer
                #pragma vertex Vert
            ENDHLSL
        }

        Pass
        {
            Name "BlitCustomPassColor"

            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment BlitCustomPassColor
                #pragma vertex Vert
            ENDHLSL
        }
    }
    Fallback Off
}
