using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEditorInternal;
using UnityEngine;

namespace Unity.ZivaRTPlayer.Editor
{
    [CustomEditor(typeof(global::ZivaRTPlayer))]
    [CanEditMultipleObjects]
    class ZivaRTPlayerEditor : UnityEditor.Editor
    {

        class Styles
        {
            public static readonly GUIContent bounds = EditorGUIUtility.TrTextContent("Custom bounds",
                "The bounding box that encapsulates the mesh.");
        }

        SerializedProperty  m_Rig;
        SerializedProperty  m_SourceMesh;
        SerializedProperty  m_AnimationRoot;
        SerializedProperty  m_GameObjectRoot;
        SerializedProperty  m_UseMeshTransformForRegistration;
        SerializedProperty  m_Implementation;
        SerializedProperty  m_SchedulingMode;
        SerializedProperty  m_RecalculateTangentFrames;
        SerializedProperty  m_ExecutionSchedule;
        SerializedProperty  m_MultithreadedJointUpdates;
        SerializedProperty  m_CalculateMotionVectors;
        SerializedProperty  m_EnableCorrectives;
        SerializedProperty  m_EnableSkinning;
        SerializedProperty  m_DrawDebugVertices;
        SerializedProperty  m_SuppressGeometryWarningMessages;
        SerializedProperty  m_CustomBounds;
        SerializedProperty  m_UseCustomBounds;

        BoxBoundsHandle m_BoundsHandle = new BoxBoundsHandle();

        void OnEnable()
        {
            m_Rig = serializedObject.FindProperty("m_Rig");
            m_SourceMesh = serializedObject.FindProperty("m_SourceMesh");
            m_AnimationRoot = serializedObject.FindProperty("m_AnimationRoot");
            m_GameObjectRoot = serializedObject.FindProperty("m_GameObjectRoot");
            m_UseMeshTransformForRegistration = serializedObject.FindProperty("m_UseMeshTransformForRegistration");
            m_Implementation = serializedObject.FindProperty("m_Implementation");
            m_SchedulingMode = serializedObject.FindProperty("m_SchedulingMode");
            m_RecalculateTangentFrames = serializedObject.FindProperty("m_RecalculateTangentFrames");
            m_ExecutionSchedule = serializedObject.FindProperty("m_ExecutionSchedule");
            m_MultithreadedJointUpdates = serializedObject.FindProperty("m_MultithreadedJointUpdates");
            m_CalculateMotionVectors = serializedObject.FindProperty("m_CalculateMotionVectors");
            m_EnableCorrectives = serializedObject.FindProperty("m_EnableCorrectives");
            m_EnableSkinning = serializedObject.FindProperty("m_EnableSkinning");
            m_DrawDebugVertices = serializedObject.FindProperty("DrawDebugVertices");
            m_SuppressGeometryWarningMessages = serializedObject.FindProperty("SuppressGeometryWarningMessages");
            m_CustomBounds = serializedObject.FindProperty("m_CustomBounds");
            m_UseCustomBounds = serializedObject.FindProperty("m_UseCustomBounds");
        }

        GUIContent s_EditModeButton;

        GUIContent editModeButton
        { 
            get
            {
                if (s_EditModeButton == null)
                {
                    s_EditModeButton = new GUIContent(
                        EditorGUIUtility.IconContent("EditCollider").image,
                        EditorGUIUtility.TrTextContent("Edit bounding volume.\n\n - Hold Alt after clicking control " +
                        "handle to pin center in place.\n - Hold Shift after clicking control handle to scale uniformly.").text
                    );
                }
                return s_EditModeButton;
            }
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            if (m_UseCustomBounds.boolValue == true)
            { 
                EditMode.DoEditModeInspectorModeButton(
                    EditMode.SceneViewEditMode.Collider,
                    "Edit Bounds",
                    editModeButton,
                    null,
                    this
                );
                EditorGUILayout.PropertyField(m_CustomBounds, Styles.bounds);
            }

            Object prevMesh = m_SourceMesh.objectReferenceValue;
            EditorGUILayout.PropertyField(m_UseCustomBounds);
            EditorGUILayout.PropertyField(m_Rig);
            EditorGUILayout.PropertyField(m_SourceMesh);
            EditorGUILayout.PropertyField(m_AnimationRoot);
            EditorGUILayout.PropertyField(m_GameObjectRoot);
            EditorGUILayout.PropertyField(m_UseMeshTransformForRegistration);
            EditorGUILayout.PropertyField(m_Implementation);
            EditorGUILayout.PropertyField(m_SchedulingMode);
            EditorGUILayout.PropertyField(m_RecalculateTangentFrames);
            EditorGUILayout.PropertyField(m_ExecutionSchedule);
            EditorGUILayout.PropertyField(m_MultithreadedJointUpdates);
            EditorGUILayout.PropertyField(m_CalculateMotionVectors);
            EditorGUILayout.PropertyField(m_EnableCorrectives); 
            EditorGUILayout.PropertyField(m_EnableSkinning); 
            EditorGUILayout.PropertyField(m_DrawDebugVertices);
            EditorGUILayout.PropertyField(m_SuppressGeometryWarningMessages);

            // if source mesh has changed we need to re-run initialization
            if (prevMesh != m_SourceMesh.objectReferenceValue)
            {
                global::ZivaRTPlayer player = (global::ZivaRTPlayer)target;
                player.m_SourceMeshChanged = true;
            }
            serializedObject.ApplyModifiedProperties();
        }

        public void OnSceneGUI()
        {
            if (!target)
                return;
            global::ZivaRTPlayer player = (global::ZivaRTPlayer)target;

            if (player.GameObjectRoot == null)
                return;

            if (!m_UseCustomBounds.boolValue)
            {
                Bounds bounds = player.AutoBounds;
                Vector3 center = bounds.center;
                Vector3 size = bounds.size;
                Handles.matrix = player.GameObjectRoot.localToWorldMatrix;
                Handles.DrawWireCube(center, size);
            }
            else
            {
                using (new Handles.DrawingScope(player.GameObjectRoot.localToWorldMatrix))
                {
                    Bounds bounds = player.CustomBounds;
                    m_BoundsHandle.center = bounds.center;
                    m_BoundsHandle.size = bounds.size;

                    // only display interactive handles if edit mode is active
                    m_BoundsHandle.handleColor = EditMode.editMode == EditMode.SceneViewEditMode.Collider
                        && EditMode.IsOwner(this) ? m_BoundsHandle.wireframeColor : Color.clear;

                    EditorGUI.BeginChangeCheck();
                    m_BoundsHandle.DrawHandle();
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(player, "Resize Bounds");
                        player.CustomBounds = new Bounds(m_BoundsHandle.center, m_BoundsHandle.size);
                    }
                }
            }
        }
    }
}
