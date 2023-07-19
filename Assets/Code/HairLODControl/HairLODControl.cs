using System;
using System.Collections;
using System.Collections.Generic;
using Unity.DemoTeam.Hair;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

[ExecuteAlways]
public class HairLODControl : MonoBehaviour
{
    public AnimationCurve distanceToLod;
    public HairInstance[] hairInstances = null;
    public bool onlyConsiderMainCamera = true;

    private HDAdditionalMeshRendererSettings[] relevantMeshRenderers;
    private void OnEnable()
    {
        relevantMeshRenderers = null;
        if (hairInstances != null)
        {
            List<HDAdditionalMeshRendererSettings> rendererSettings = new List<HDAdditionalMeshRendererSettings>();
            foreach (var inst in hairInstances)
            {
                if (inst != null)
                {
                    HDAdditionalMeshRendererSettings[] mrsList =
                        inst.transform.GetComponentsInChildren<HDAdditionalMeshRendererSettings>();
                    foreach (var mrs in mrsList)
                    {
                        rendererSettings.Add(mrs);
                    }
                }
                
            }

            if (rendererSettings.Count > 0)
            {
                relevantMeshRenderers = rendererSettings.ToArray();
            }
        }
    }

    void UpdateHairInstanceLOD(float distanceToCamera)
    {
        if (hairInstances != null)
        {
            var hdCam = HDCamera.GetOrCreate(Camera.main);
            var stack = hdCam?.volumeStack;
            var hairLODComponent = stack?.GetComponent<HairLODComponent>();

            if (hairLODComponent != null)
            {
                if (relevantMeshRenderers != null)
                {
                    float lod = distanceToLod.Evaluate(distanceToCamera);

                    foreach (var rend in relevantMeshRenderers)
                    {
                        //TODO: should maybe make the hair thicker when it gets fewer to account for increasing alpha in shader
                        if (rend != null)
                        {
                            rend.rendererLODMode = LineRendering.RendererLODMode.Fixed;
                            float rendererLod = lod;
                            
                            if (hairLODComponent.shadingFraction.overrideState)
                                rend.shadingSampleFraction = hairLODComponent.shadingFraction.value;
                            if (hairLODComponent.strandCountMultiplier.overrideState)
                                rendererLod *= hairLODComponent.strandCountMultiplier.value;
                                
                            rend.rendererLODFixed = Mathf.Clamp01(rendererLod);
                        }
                    }
                }


                foreach (var hairInstance in hairInstances)
                {
                    if (hairInstance != null)
                    {
                        /*if (hairLODComponent.primaryGlobalLODScale.overrideState)
                            hairInstance.settingsSystem.kLODSearchValue *= hairLODComponent.primaryGlobalLODScale.value;

                        if (hairLODComponent.secondaryGlobalLODScale.overrideState)
                            hairInstance.settingsSystem.kLODSearchValue *=
                                hairLODComponent.secondaryGlobalLODScale.value;*/
                    }
                }
            }
        }
    }
    
    
    // Update is called once per frame
    void Update()
    {
        List<Camera> cameras = new List<Camera>();
        if (onlyConsiderMainCamera)
        {
            if (Camera.main != null)
            {
                cameras.Add(Camera.main);
            }
        }
        else
        {
            cameras.AddRange(Camera.allCameras);
            #if UNITY_EDITOR
            cameras.AddRange(SceneView.GetAllSceneCameras());
            #endif
        }
        
        if (cameras.Count == 0)
        {
            UpdateHairInstanceLOD(0);
        }
        else
        {
            float distance = float.MaxValue;
            foreach (var cam in cameras)
            {
                Vector3 camPos = cam.transform.position;
                float distanceToCamera = Vector3.Distance(camPos, transform.position);
                if(distance > distanceToCamera)
                {
                    distance = distanceToCamera;
                }
            }
            
            UpdateHairInstanceLOD(distance);
        }
       
        
        

    }
}