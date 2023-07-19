using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

// Copy-pasta from FocusCameraControl.cs


[ExecuteAlways]
[DisallowMultipleComponent]
public class DepthOfFieldLayeredFocus : MonoBehaviour
{
    public Transform focusTarget;
    public Transform layerPosition;
    public int volumePriority = 100;
    [Range(0, 1)]
    public float minimumBlur = 0; 


    [Tooltip("Enabling debug will leave the backing data visible and editable in the scene.")]
    public bool debug;

    GameObject m_Volume;
    VolumeProfile m_Profile;
    DepthOfFieldLayered m_DepthOfFieldLayered;

    HideFlags kHideFlags => debug ? HideFlags.None : HideFlags.NotEditable | HideFlags.DontSaveInBuild | HideFlags.DontSaveInEditor | HideFlags.HideInHierarchy | HideFlags.HideInInspector;
    
    void OnEnable()
    {
        m_Volume = new GameObject("DepthOfFieldLayeredFocus") {hideFlags = kHideFlags};
        var volume = m_Volume.AddComponent<Volume>();
        volume.isGlobal = true;
        volume.priority = volumePriority;

        m_Profile = volume.sharedProfile = ScriptableObject.CreateInstance<VolumeProfile>();
        m_Profile.hideFlags = kHideFlags;
        m_Profile.name= "DoFFocusDistanceOverride";
        
        m_DepthOfFieldLayered = m_Profile.Add<DepthOfFieldLayered>();
        m_DepthOfFieldLayered.hideFlags = kHideFlags;
        m_DepthOfFieldLayered.layerToFocusDistance.overrideState = true;
        m_DepthOfFieldLayered.minimumBlur.overrideState = true;
    }

    void OnDisable()
    {
        void SafeDestroy(Object obj) { if (Application.isPlaying) Destroy(obj); else DestroyImmediate(obj); }
        
        m_Profile.Remove<DepthOfField>();
        SafeDestroy(m_Volume);
        SafeDestroy(m_Profile);
        SafeDestroy(m_DepthOfFieldLayered);
    }

    void LateUpdate()
    {
        if (layerPosition && focusTarget)
        {
            m_DepthOfFieldLayered.active = true;
            m_DepthOfFieldLayered.layerToFocusDistance.value = Vector3.Distance(layerPosition.transform.position, focusTarget.transform.position);
            m_DepthOfFieldLayered.minimumBlur.value = minimumBlur;
        }
        else
        {
            m_DepthOfFieldLayered.active = false;
        }
    }
}
