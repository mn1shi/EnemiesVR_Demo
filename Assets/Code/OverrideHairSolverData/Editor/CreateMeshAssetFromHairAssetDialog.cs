#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.DemoTeam.Hair;
using Unity.DemoTeam.DigitalHuman;
using UnityEditor;
using UnityEngine;

public class CreateMeshAssetFromHairAssetDialog : EditorWindow
{
    private HairAsset hairAsset;
    
    //[MenuItem("Tools/Generate Mesh From Hair Asset")]
    static void Init()
    {
        var window = GetWindow(typeof(CreateMeshAssetFromHairAssetDialog));
        window.titleContent = new GUIContent("Generate Mesh From Hair Asset");
        window.Show();
    }

    void OnGUI()
    {
        hairAsset = EditorGUILayout.ObjectField("HairAsset: ",hairAsset, typeof(HairAsset), false) as HairAsset;

        if (hairAsset && hairAsset.strandGroups.Length > 0)
        {
            if (GUILayout.Button("Create Mesh"))
            {
                Mesh m = new Mesh();
                m.vertices = hairAsset.strandGroups[0].particlePosition;
                m.RecalculateNormals();
                
                string path = EditorUtility.SaveFilePanel(
                    "Save Mesh",
                    "",
                    "HairInstanceMesh" + ".asset",
                    "asset");
                if (path.Length != 0 && path.Contains(Application.dataPath))
                {
                    path = path.Replace(Application.dataPath + "/", "Assets/");
                    AssetDatabase.CreateAsset(m, path);
                }
                
                
            }
        }
    }
    
}
#endif