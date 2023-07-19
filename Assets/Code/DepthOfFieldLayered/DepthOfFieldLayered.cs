using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using System;

[Serializable, VolumeComponentMenu("Post-processing/Custom/DepthOfFieldLayered"), SupportedOnRenderPipeline(typeof(HDRenderPipelineAsset))]
public sealed class DepthOfFieldLayered : CustomPostProcessVolumeComponent, IPostProcessComponent
{
    [Tooltip("Controls the intensity of the effect.")]
    public ClampedFloatParameter intensity = new ClampedFloatParameter(0, 0, 1);
    public ClampedIntParameter sampleCountSqrt = new ClampedIntParameter(11, 2, 16);
    public ClampedFloatParameter blurScale = new ClampedFloatParameter(0, 0, 1);
    public ClampedFloatParameter minimumBlur = new ClampedFloatParameter(0, 0, 1);
    public MinFloatParameter layerToFocusDistance = new MinFloatParameter(0, 0);

    Material m_Material;

    public bool IsActive() => m_Material != null && intensity.value > 0f;

    // Do not forget to add this post process in the Custom Post Process Orders list (Project Settings > Graphics > HDRP Settings).
    public override CustomPostProcessInjectionPoint injectionPoint => CustomPostProcessInjectionPoint.AfterPostProcessBlurs;
    public override bool visibleInSceneView => false;

    Shader m_CompositeShader;
    ComputeShader m_DepthOfFieldKernel;
    ComputeShader m_DepthOfFieldGather;
    
    static readonly int layerTexture = Shader.PropertyToID("DoFLayerTexture");
    static readonly int layerTextureBlurred = Shader.PropertyToID("DoFLayerTextureBlurred");
    int m_DepthOfFieldKernelKernel;
    int m_DepthOfFieldGatherKernel;
    ComputeBuffer m_KernelBuffer;

    public override void Setup()
    {
        m_CompositeShader = Resources.Load<Shader>("DepthOfFieldLayered");
        m_DepthOfFieldKernel = Resources.Load<ComputeShader>("DepthOfFieldKernel");
        m_DepthOfFieldGather = Resources.Load<ComputeShader>("DepthOfFieldGather");

        m_Material = new Material(m_CompositeShader);
        m_DepthOfFieldKernelKernel = m_DepthOfFieldKernel.FindKernel("KParametricBlurKernel");
        m_DepthOfFieldGatherKernel = m_DepthOfFieldGather.FindKernel("KMainNear");
    }

    static class Uniforms
    {
        public static readonly int _Params = Shader.PropertyToID("_Params");
        public static readonly int _Params1 = Shader.PropertyToID("_Params1");
        public static readonly int _Params2 = Shader.PropertyToID("_Params2");
        public static readonly int _TexelSize = Shader.PropertyToID("_TexelSize");
        public static readonly int _InputTexture = Shader.PropertyToID("_InputTexture");
        public static readonly int _InputCoCTexture = Shader.PropertyToID("_InputCoCTexture");
        public static readonly int _InputDilatedCoCTexture = Shader.PropertyToID("_InputCoCTexture");
        public static readonly int _OutputTexture = Shader.PropertyToID("_OutputTexture");
        public static readonly int _OutputAlphaTexture = Shader.PropertyToID("_OutputAlphaTexture");
        public static readonly int _BokehKernel = Shader.PropertyToID("_BokehKernel");
        public static readonly int _BlurAmount = Shader.PropertyToID("_BlurAmount");
        public static readonly int _Intensity = Shader.PropertyToID("_Intensity");
        public static readonly int _DoFLayerTextureBlurred = Shader.PropertyToID("_DoFLayerTextureBlurred");
        public static readonly int _ResolutionScale = Shader.PropertyToID("_ResolutionScale");
        public static readonly int _PostScreenSizeMatch = Shader.PropertyToID("_PostScreenSizeMatch");
    }

    public override void Render(CommandBuffer cmd, HDCamera camera, RTHandle source, RTHandle destination)
    {
        if (m_Material == null)
            return;

        int nearSamples = sampleCountSqrt.value;

        // Generate kernel sample positions
        float anamorphism = 0;
        float ngonFactor = 1;
        int bladeCount = 8;
        float rotation = 0;
        if (m_KernelBuffer != null && m_KernelBuffer.count != nearSamples * nearSamples)
        {
            m_KernelBuffer.Release();
            m_KernelBuffer = null;
        }
        if (m_KernelBuffer == null)
        {
            m_KernelBuffer = new ComputeBuffer(nearSamples * nearSamples, sizeof(uint));
            cmd.SetComputeVectorParam(m_DepthOfFieldKernel, Uniforms._Params1, new Vector4(nearSamples, ngonFactor, bladeCount, rotation));
            cmd.SetComputeVectorParam(m_DepthOfFieldKernel, Uniforms._Params2, new Vector4(anamorphism, 0f, 0f, 0f));
            cmd.SetComputeBufferParam(m_DepthOfFieldKernel, m_DepthOfFieldKernelKernel, Uniforms._BokehKernel, m_KernelBuffer);
            cmd.DispatchCompute(m_DepthOfFieldKernel, m_DepthOfFieldKernelKernel, Mathf.CeilToInt((nearSamples * nearSamples) / 64f), 1, 1);
        }

        // Copy the custom pass color texture, since otherwise there's no way to bind it to
        // the compute shader without modifying hdrp :/ An unnecessary fullscreen blit.
        var desc = destination.rt.descriptor;
        desc.depthBufferBits = 0;
        cmd.GetTemporaryRT(layerTexture, desc);
        cmd.SetRenderTarget(layerTexture);
        CoreUtils.SetViewport(cmd, /*only used as scaled size reference:*/ destination);
        cmd.DrawProcedural(Matrix4x4.identity, m_Material, shaderPass: 1, MeshTopology.Triangles, 3, 1);

        // RW texture for blur write
        desc.enableRandomWrite = true;
        cmd.GetTemporaryRT(layerTextureBlurred, desc);

        // Calculate blur amount
        // TODO: just a quick hack, calculate this properly from physical camera properties
        float blurAmount = Mathf.Max(minimumBlur.value * 10, layerToFocusDistance.value * blurScale.value * 100);

        // Do blur
        int barrelClipping = 0;
        int nearMaxBlur = 1;
        // Use the viewport resolution instead of the RTHandle source/destination texture resolution, which might be larger
        // TODO: this will be cleaner once the custom post process pass becomes a proper render graph pass.
        int targetWidth = camera.actualWidth;
        int targetHeight = camera.actualHeight;
        
        int threadGroup8X = (targetWidth + 7) / 8;
        int threadGroup8Y = (targetHeight + 7) / 8;
        cmd.SetComputeVectorParam(m_DepthOfFieldGather, Uniforms._Params, new Vector4(nearSamples, nearSamples * nearSamples, barrelClipping, nearMaxBlur));
        cmd.SetComputeVectorParam(m_DepthOfFieldGather, Uniforms._TexelSize, new Vector4(targetWidth, targetHeight, 1f / targetWidth, 1f / targetHeight));
        cmd.SetComputeTextureParam(m_DepthOfFieldGather, m_DepthOfFieldGatherKernel, Uniforms._InputTexture, layerTexture);
        cmd.SetComputeTextureParam(m_DepthOfFieldGather, m_DepthOfFieldGatherKernel, Uniforms._OutputTexture, layerTextureBlurred);
        cmd.SetComputeBufferParam(m_DepthOfFieldGather, m_DepthOfFieldGatherKernel, Uniforms._BokehKernel, m_KernelBuffer);
        cmd.SetComputeFloatParam(m_DepthOfFieldGather, Uniforms._BlurAmount, blurAmount);
        cmd.DispatchCompute(m_DepthOfFieldGather, m_DepthOfFieldGatherKernel, threadGroup8X, threadGroup8Y, 1);

        // Final composite over the scene
        var resolutionScale = new Vector4(targetWidth/(float)desc.width, targetHeight/(float)desc.height);
        bool postScreenSizeMatch = targetWidth == camera.postProcessScreenSize.x && targetHeight == camera.postProcessScreenSize.y;

        m_Material.SetInt(Uniforms._PostScreenSizeMatch, postScreenSizeMatch ? 1 : 0);
        m_Material.SetFloat(Uniforms._Intensity, intensity.value);
        m_Material.SetVector(Uniforms._ResolutionScale, resolutionScale);
        m_Material.SetTexture(Uniforms._InputTexture, source);
        cmd.SetGlobalTexture(Uniforms._DoFLayerTextureBlurred, layerTextureBlurred);
        HDUtils.DrawFullScreen(cmd, m_Material, destination, properties: null, shaderPassId: 0);
        
        cmd.ReleaseTemporaryRT(layerTexture);
        cmd.ReleaseTemporaryRT(layerTextureBlurred);
    }

    public override void Cleanup()
    {
        CoreUtils.Destroy(m_Material);
        m_KernelBuffer?.Release();
        m_KernelBuffer = null;
    }
}