using System.Linq;
using UnityEngine;
using Unity.DemoTeam.Hair;
#if UNITY_EDITOR
    using UnityEditor;
    using UnityEditor.Build;
    using UnityEditor.Build.Reporting;
#endif

class HairInstanceStripper : MonoBehaviour
#if UNITY_EDITOR
    , IProcessSceneWithReport
#endif
{
    public HairInstance[] instancesToStrip = System.Array.Empty<HairInstance>();
    public HairInstance[] instancesToStripOnNoQualitySelection = System.Array.Empty<HairInstance>();

#if UNITY_EDITOR
    public int callbackOrder => -10000;

    public void OnProcessScene(UnityEngine.SceneManagement.Scene scene, BuildReport report)
    {
        if (report == null) return;    
        
        foreach (var hairInstanceStripper in scene.GetRootGameObjects().SelectMany(go => go.GetComponentsInChildren<HairInstanceStripper>()))
        {
            foreach (var hairInstance in hairInstanceStripper.instancesToStrip)
            {
                Debug.Log($"[HairInstanceStripper] Stripping '{hairInstance.name}' from scene '{scene.name}'.");
                DestroyImmediate(hairInstance.gameObject);
            }
            
            if(report.summary.platformGroup != BuildTargetGroup.Standalone)
            {
                foreach (var hairInstance in hairInstanceStripper.instancesToStripOnNoQualitySelection)
                {
                    Debug.Log($"[HairInstanceStripper] Stripping '{hairInstance.name}' from scene '{scene.name}'.");
                    DestroyImmediate(hairInstance.gameObject);
                }
            }

            DestroyImmediate(hairInstanceStripper.gameObject);
        }

        foreach (var hairInstance in scene.GetRootGameObjects().SelectMany(go => go.GetComponentsInChildren<HairInstance>(true)))
        {
            var groupAssetReferences = hairInstance.strandGroupDefaults.groupAssetReferences;
            if (groupAssetReferences != null)
            {
                foreach (var groupAssetReference in groupAssetReferences)
                {
                    var hairAsset = groupAssetReference.hairAsset; 
                    if (hairAsset)
                    {
                        hairAsset.settingsProcedural.mappedDensity = null;
                        hairAsset.settingsProcedural.mappedDirection = null;
                        hairAsset.settingsProcedural.mappedParameters = null;
						hairAsset.settingsLODClusters.baseLODParamsUVMapped.baseLODClusterMaps = null;
                    }
                }
            }
        }
    }
#endif
}
