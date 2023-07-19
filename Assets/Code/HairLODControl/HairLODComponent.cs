using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

// TEMP: Global Volume/Settings control for Hair Simulation LOD 
[SupportedOnRenderPipeline(typeof(HDRenderPipelineAsset))]
public class HairLODComponent : VolumeComponent
{
    public ClampedFloatParameter primaryGlobalLODScale = new(1f, 0f, 1f);
    public ClampedFloatParameter secondaryGlobalLODScale = new(1f, 0f, 1f);
    public ClampedFloatParameter shadingFraction = new(1f, 0.001f, 1f);
    public ClampedFloatParameter strandCountMultiplier = new(1f, 0.001f, 100.0f);

}
