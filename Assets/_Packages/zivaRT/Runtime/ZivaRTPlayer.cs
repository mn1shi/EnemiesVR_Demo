#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.ZivaRTPlayer;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Assertions;
using UnityEngine.Jobs;
using UnityEngine.Playables;
using UnityEngine.Profiling;
using UnityEngine.Serialization;

[assembly: InternalsVisibleTo("Unity.ZivaRTPlayer.Tests")]
[assembly: InternalsVisibleTo("Unity.ZivaRTPlayer.EditorTests")]

/// <summary>
/// A ZivaRT runtime component that can be attached to a GameObject.
/// When configured properly this will allow for Ziva to be used to
/// deform skin a mesh.
/// </summary>
[RequireComponent(typeof(Animator))]
[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
[ExecuteInEditMode]
public class ZivaRTPlayer : MonoBehaviour, IAnimationWindowPreview
{
    [Header("Deformation Inputs")]
    [Tooltip("ZivaRT asset used to deform the target character mesh.")]
    [SerializeField]
    ZivaRTRig m_Rig;

    // Mapping from unity mesh to ZivaRT mesh
    // unity mesh vertex 'i' maps to ZivaRT mesh vertex 'm_VertexOrderMapping[i]'
    // The name 'vertex order mapping' was chosen with thoughts given to any future naming requirements e.g. 'face order mapping'
    // Another strong candidate for this variables name was 'vertex index mapping', but 'order' was chosen over 'index' to disambiguate from the vertex indices list used to define faces
    [SerializeField]
    [HideInInspector]
    int[] m_VertexOrderMapping;

    // Bounding information per bone
    [SerializeField]
    [HideInInspector]
    float3[] m_Centers;
    [SerializeField]
    [HideInInspector]
    float3[] m_Extents;
    // array of bone indices that influence vertices 
    [SerializeField]
    [HideInInspector]
    int[] m_BoneIndices;

    ComputeBoundsJob m_ComputeBoundsJob;

    Bounds m_AutoBounds = new Bounds(); // generated automatically from current pose bounds

    /// <summary>
    /// Automatically generated bounds used for visibility culling.
    /// </summary>
    public Bounds AutoBounds
    {
        get => m_AutoBounds;
    }

    [SerializeField]
    Bounds m_CustomBounds;

    /// <summary>
    /// The custom bounds used for visibility culling.
    /// </summary>
    public Bounds CustomBounds
    {
        get => m_CustomBounds;
        set
        {
            if (m_CustomBounds == value)
                return;

            m_CustomBounds = value;
        }
    }

    [Tooltip("If checked, the custom bounds set above are used for visibility culling. " +
        "If unchecked the bounds are computed automatically.")]
    [SerializeField]
    private bool m_UseCustomBounds;

    /// <summary>
    /// If this is true then custom bounds are used for mesh visibility culling.
    /// If this is false then bounds are computed automatically.
    /// </summary>
    public bool UseCustomBounds
    {
        get => m_UseCustomBounds;
        set
        {
            if (m_UseCustomBounds == value)
                return;

            m_UseCustomBounds = value;
        }
    }

    /// <summary>
    /// The ZivaRT rig asset used to deform the target character mesh.
    /// </summary>
    public ZivaRTRig Rig
    {
        get => m_Rig;
        set
        {
            if (m_Rig == value)
                return;

            m_Rig = value;
            m_Reinitialize = true;
        }
    }

    [SerializeField]
    [Tooltip("Source Mesh that will be deformed by the Ziva Rig. A copy will be created for the deformation")]
    Mesh m_SourceMesh;

    internal bool m_SourceMeshChanged = false;

    /// <summary>
    /// Source Mesh that will be deformed by the Ziva Rig.
    /// A copy will be created for the deformation which means
    /// this source mesh will not be modified directly.
    /// </summary>
    public Mesh SourceMesh
    {
        get => m_SourceMesh;
        set
        {
            if (m_SourceMesh == value)
                return;

            m_SourceMesh = value;
            m_Reinitialize = true;
            m_SourceMeshChanged = true;
        }
    }

    [SerializeField]
    [HideInInspector]
    float m_MeshRescale = 1;

    [SerializeField]
    [HideInInspector]
    float m_MeshRescaleRcp = 1f;

    /// <summary>
    /// Scale factor that should be applied to the Ziva RT vertices
    /// to match the mesh scaling applied on the imported mesh file
    /// </summary>
    public float MeshRescale
    {
        get => m_MeshRescale;
        set
        {
            if (m_MeshRescale == value)
                return;

            m_MeshRescale = value;
            m_MeshRescaleRcp = 1f / value;
            m_Reinitialize = true;
        }
    }

    [SerializeField]
    [Tooltip("The animation root for which the animation should be applied from")]
    Transform m_AnimationRoot;

    /// <summary>
    /// The animation root for which the animation should be applied from
    /// </summary>
    public Transform AnimationRoot
    {
        get => m_AnimationRoot;
        set
        {
            if (m_AnimationRoot == value)
                return;

            m_AnimationRoot = value;
            m_Reinitialize = true;
        }
    }

    [SerializeField]
    [Tooltip("GameObject root")]
    Transform m_GameObjectRoot;
    /// <summary>
    /// The GameObject root for which the animation should be applied from
    /// </summary>
    public Transform GameObjectRoot
    {
        get => m_GameObjectRoot;
        set
        {
            if (m_GameObjectRoot == value)
                return;

            m_GameObjectRoot = value;
            m_Reinitialize = true;
        }
    }

    [Tooltip("Apply the local transform of the Skinned Mesh during vertex correspondence registration step. " +
             "Can help resolve registration failures caused by mismatched mesh transforms.")]
    [SerializeField]
    bool m_UseMeshTransformForRegistration = true;

    /// <summary>
    /// Apply the local transform of the Skinned Mesh during vertex correspondence registration step.
    /// </summary>
    public bool UseMeshTransformForRegistration
    {
        get => m_UseMeshTransformForRegistration;
        set
        {
            if (m_UseMeshTransformForRegistration == value)
                return;

            m_UseMeshTransformForRegistration = value;
            m_Reinitialize = true;
        }
    }

    /// <summary>
    /// The technique used to perform the Ziva deformation
    /// on a Ziva deformed object.
    /// </summary>
    public enum ImplementationType
    {
        /// <summary>
        /// The deformation is executed using CPU side burst jobs
        /// with the results being updated onto the attached mesh.
        /// The results can be accessed on the CPU side for further
        /// operations.
        /// </summary>
        BurstJobs,
        /// <summary>
        /// The deformation is executed on the GPU using a compute
        /// shader with the result only being accessible on the GPU.
        /// </summary>
        ComputeShaders,
        /// <summary>
        /// The deformation is executed on the CPU using standard C#
        /// with the results being updated onto the attached mesh.
        /// The results can be accessed on the CPU side for further
        /// operations.
        /// </summary>
        Mono
    }

    [Header("Runtime Settings")]
    [Tooltip("Which type of ZivaRT solver to use.")]
    [SerializeField]
    ImplementationType m_Implementation = ImplementationType.BurstJobs;
    /// <summary>
    /// Select which internal implementation is used to perform the
    /// deformation. Different implementations have different properties
    /// (CPU vs GPU).
    /// </summary>
    public ImplementationType Implementation
    {
        get => m_Implementation;
        set
        {
            if (m_Implementation == value)
                return;

            m_Implementation = value;
            m_Reinitialize = true;
        }
    }

    [RequiredImplementation(ImplementationType.BurstJobs)]
    [Tooltip("How to run Job Solver's jobs.")]
    [SerializeField]
    SchedulingMode m_SchedulingMode = SchedulingMode.MultipleWorkerThreads;
    internal SchedulingMode SchedulingMode  // internal to be accessable in tests
    {
        get => m_SchedulingMode;
        set
        {
            if (m_SchedulingMode == value)
                return;

            m_SchedulingMode = value;
            m_Reinitialize = true;
        }
    }

    [Tooltip("Recompute target mesh normals and tangents each frame.")]
    [FormerlySerializedAs("<recalculateTangentFrames>k__BackingField")]
    [SerializeField]
    RecomputeTangentFrames m_RecalculateTangentFrames = RecomputeTangentFrames.NormalsAndTangents;
    /// <summary>
    /// Should the solver recompute target mesh normals and tangents each frame?
    /// </summary>
    public RecomputeTangentFrames RecalculateTangentFrames
    {
        get => m_RecalculateTangentFrames;
        set => m_RecalculateTangentFrames = value;
    }


    /// <summary>
    /// This determines at which point in Unity's execution
    /// order deformation calculation starts
    /// </summary>
    public enum ExecutionScheduleType
    {
        /// <summary>
        /// The deformation calculation is scheduled during the Update phase if
        /// GameObject containing the ZivaRTPlayer component is visible
        /// </summary>
        Update,
        /// <summary>
        /// The deformation calculation is scheduled during the LateUpdate phase
        /// if GameObject containing the ZivaRTPlayer component is visible
        /// </summary>
        LateUpdate,
        /// <summary>
        /// The deformation calculation is scheduled manually by calling RunSolver()
        /// </summary>
        Manual
    }

    [Tooltip("Whether to schedule deformation calculations manually, in the LateUpdate phase or normal Update phase.")]   
    [SerializeField]
    ExecutionScheduleType m_ExecutionSchedule = ExecutionScheduleType.LateUpdate;
    /// <summary>
    /// Should deformation calculations be started manually, in the LateUpdate phase or normal Update phase?
    /// </summary>
    public ExecutionScheduleType ExecutionSchedule
    {
        get => m_ExecutionSchedule;
        set => m_ExecutionSchedule = value;
    }

    [field: Tooltip("Whether or not to execute joint transform updates on multiple threads.  This delays solver execution to LateUpdate")]
    [FormerlySerializedAs("<multithreadedJointUpdates>k__BackingField")]
    [SerializeField]
    bool m_MultithreadedJointUpdates = true;
    /// <summary>
    /// Should joint transform updates execute on multiple threads? This delays solver execution to LateUpdate
    /// </summary>
    public bool MultithreadedJointUpdates
    {
        get => m_MultithreadedJointUpdates;
        set => m_MultithreadedJointUpdates = value;
    }

    [Tooltip("Calculate motion vectors for deformed positions")]
    [SerializeField]
    bool m_CalculateMotionVectors = true;
    /// <summary>
    /// Should motion vectors for deformed positions be calculated?
    /// </summary>
    public bool CalculateMotionVectors
    {
        get => m_CalculateMotionVectors;
        set
        {
            if (m_CalculateMotionVectors == value)
                return;

            m_CalculateMotionVectors = value;
            m_Reinitialize = true;
        }
    }

    [Header("Debugging")]
    [Tooltip("Whether to apply pre-skinning correctives from the ZivaRT solver (used for debugging).")]
    [SerializeField]
    bool m_EnableCorrectives = true;
    /// <summary>
    /// Whether to apply pre-skinning correctives from the ZivaRT solver (generally used for debugging).
    /// </summary>
    public bool EnableCorrectives
    {
        get => m_EnableCorrectives;
        set
        {
            if (m_EnableCorrectives == value)
                return;

            m_EnableCorrectives = value;
            m_Reinitialize = true;
        }
    }

    [Tooltip("Whether to apply skinning to the mesh using the ZivaRT solver (used for debugging).")]
    [FormerlySerializedAs("<enableSkinning>k__BackingField")]
    [SerializeField]
    bool m_EnableSkinning = true;
    /// <summary>
    /// Whether to apply skinning to the mesh using the ZivaRT solver (used for debugging).
    /// </summary>
    public bool EnableSkinning
    {
        get => m_EnableSkinning;
        set => m_EnableSkinning = value;
    }

    // RUNTIME THINGS
    Mesh m_TargetMesh;

    Transform[] m_JointTransforms;
    TransformAccessArray m_TransformAccessArray;
    NativeArray<int> m_BoneIndexMap;

    int m_LastFrameSolved = -1;
    int m_LastFrameSynced = -1;
    Solver m_SelectedSolver;

    NativeArray<PropertyStreamHandle> m_ExtraParameterStreamHandles;
    NativeArray<float> m_ExtraParameterValues;
    NativeArray<int> m_ParentIndices;
    NativeArray<float> m_PoseNativeArray;
    NativeArray<float> m_BonesNativeArray;

    AnimationScriptPlayable m_AnimationWindowPlayable;
    AnimationJob m_AnimationWindowJob;
    bool m_AnimationWindowPreviewStarted = false;

    //Animation job for IAnimationWindowPreview
    struct AnimationJob : IAnimationJob
    {
        public ZivaRTPlayer ZrtPlayer;

        //This is required to implement the interface IAnimationJob
        public void ProcessRootMotion(AnimationStream stream) { }

        public void ProcessAnimation(AnimationStream stream)
        {
            if (ZrtPlayer.m_NumExtraParameters == 0)
                return;

            if (stream.isValid)
            {
                //Get the Extra Parameters from the animation stream
                AnimationStreamHandleUtility.ReadFloats(stream, ZrtPlayer.m_ExtraParameterStreamHandles, ZrtPlayer.m_ExtraParameterValues);
            }
        }
    }

    [field: NonSerialized]
    bool m_Reinitialize { get; set; } = true;

    /// <summary>
    /// Draw the vertices of the imported ZivaRT asset's rest shape in the Scene view.
    /// Can be useful to diagnose vertex mapping issues.
    /// </summary>
    [Tooltip("Draw the vertices of the imported ZivaRT asset's rest shape in the Scene view.")]
    public bool DrawDebugVertices = false;

    /// <summary>
    /// Don't print out warnings about issues discovered with the mesh, eg: degenerate face UVs.
    /// </summary>
    [Tooltip("Silence warnings caused by issues found in the Source Mesh' geometry, e.g. degenerate UVs.")]
    public bool SuppressGeometryWarningMessages = false;

    int m_NumExtraParameters;
    JobHandle m_BoneExtractionJobHandle;

    void Awake()
    {
        Implementation = m_Implementation;
    }

    void OnEnable()
    {
        m_Reinitialize = true;
        ZivaRTPlayerManager.Instance.RegisterPlayer(this);
    }

    void OnDisable()
    {
        ZivaRTPlayerManager.Instance.UnregisterPlayer(this);
        Cleanup();
    }

    internal void Cleanup()
    {
        CleanUpNativeStreams();
        m_SelectedSolver?.Dispose();
    }

    void CleanUpNativeStreams()
    {
        m_BoneExtractionJobHandle.Complete();

        if (m_ExtraParameterStreamHandles.IsCreated)
            m_ExtraParameterStreamHandles.Dispose();

        if (m_ExtraParameterValues.IsCreated)
            m_ExtraParameterValues.Dispose();

        if (m_TransformAccessArray.isCreated)
            m_TransformAccessArray.Dispose();

        if (m_ParentIndices.IsCreated)
            m_ParentIndices.Dispose();

        if (m_BoneIndexMap.IsCreated)
            m_BoneIndexMap.Dispose();

        if (m_PoseNativeArray.IsCreated)
            m_PoseNativeArray.Dispose();

        if (m_BonesNativeArray.IsCreated)
            m_BonesNativeArray.Dispose();

        m_ComputeBoundsJob.ReleaseBuffers();
    }

    void Start()
    {
        m_Reinitialize = true;
        m_AnimationWindowPreviewStarted = false;
        // If the scene starts and m_TargetMesh is null, the renderer component's bounds will be just a point located
        // at the pivot position. If that point isn't in the camera's frustum, the character will be culled,
        // OnWillRenderObject will not get called, and the target mesh will never be created.
        // So let's make sure we are at the very least completely initialized at the start no matter the visibility.
        if (m_TargetMesh == null)
            InitializeTargetMesh();
    }

    void OnWillRenderObject()
    {
        // when manual update is on we don't rely on visibility
        if (m_ExecutionSchedule == ExecutionScheduleType.Manual)
        {
            return;
        }

        // if we are just visible for the first time
        // we need to kick our deformation
        if (!m_DeformationKickedThisFrame)
        {
            StartDeformation();
            FinishDeformation();
        }

        ZivaRTPlayerManager.Instance.RegisterWasVisible(this);
        SyncZivaSolver();
    }

    internal void ZivaUpdate()
    {
        if (m_ExecutionSchedule == ExecutionScheduleType.Update)
        {
            StartDeformation();
            if (!MultithreadedJointUpdates)
                FinishDeformation();
        }
    }

    internal void ZivaLateUpdate()
    {
        if (m_ExecutionSchedule == ExecutionScheduleType.LateUpdate)
        {
            StartDeformation();
            FinishDeformation();
        }
        else if ((MultithreadedJointUpdates) && (m_ExecutionSchedule != ExecutionScheduleType.Manual))
        {
            FinishDeformation();
        }
    }

    internal bool BuildVertexOrderMapping()
    {
        if (SourceMesh == null)
        {
            m_VertexOrderMapping = null; // no unity vertices available to do the mapping
        }

        // only do this if the mesh changed and is not null. If both m_PrevMesh and source meshes are null
        // then m_VertexOrderMapping gets reset in the code above
        if ((SourceMesh != null) && m_SourceMeshChanged)
        {
            // Do mesh registration to find a correspondence between vertices.
            Profiler.BeginSample("BuildUnityToZivaIndexMap");
            var meshTransformMatrix = GetMeshTransformForRegistration();
            m_VertexOrderMapping = BuildVertexOrderMappingInternal(Rig, SourceMesh, meshTransformMatrix);
            Profiler.EndSample();
        }

        m_SourceMeshChanged = false;
        // BuildVertexOrderMappingInternal() will emit an error message if we failed to find a vertex correspondence
        // between meshes.
        return m_VertexOrderMapping != null;
    }

    internal void BuildSerializableBoundsData()
    {
        int numBones = Rig.m_Skinning.RestPoseInverse.Length / 12;   // 4x3 matrices        
        Vector3[] zivaPoints = ToVector3Array(Rig.m_Character.RestShape);
        float3[] min = new float3[numBones];
        float3[] max = new float3[numBones];
        bool[] used = new bool[numBones];

        for (int i = 0; i < numBones; i++)
        {
            min[i] = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
            max[i] = new float3(float.MinValue, float.MinValue, float.MinValue);
            used[i] = false;
        }

        SparseMatrix skinningWeights = Rig.m_Skinning.SkinningWeights;
        for (int vertexIndex = 0; vertexIndex < skinningWeights.NumCols; vertexIndex++)
        {
            Vector3 point = zivaPoints[vertexIndex];
            for (int bonePerVertexIndex = skinningWeights.ColStarts[vertexIndex]; bonePerVertexIndex < skinningWeights.ColStarts[vertexIndex + 1]; bonePerVertexIndex++)
            {
                int boneIndex = skinningWeights.RowIndices[bonePerVertexIndex];
                min[boneIndex] = math.min(min[boneIndex], point);
                max[boneIndex] = math.max(max[boneIndex], point);
                used[boneIndex] = true;
            }
        }

        // calculate total number of used 
        int totalUsedBones = 0;
        for (int boneIndex = 0; boneIndex < numBones; boneIndex++)
        {
            if (used[boneIndex])
                totalUsedBones++;
        }

        m_Centers = new float3[totalUsedBones];
        m_Extents = new float3[totalUsedBones];
        m_BoneIndices = new int[totalUsedBones];

        int usedBoneIndex = 0;
        for (int boneIndex = 0; boneIndex < numBones; boneIndex++)
        {
            if (used[boneIndex])
            {
                m_Extents[usedBoneIndex] = (max[boneIndex] - min[boneIndex]) * 0.5f;
                m_Centers[usedBoneIndex] = min[boneIndex] + m_Extents[usedBoneIndex];
                m_BoneIndices[usedBoneIndex] = boneIndex;
                usedBoneIndex++;
            }
        }
    }

    void InitializeTargetMesh()
    {
        if (m_TargetMesh != null)
        {
            DestroyImmediate(m_TargetMesh);
            m_TargetMesh = null;
        }
        if (SourceMesh != null)
            m_TargetMesh = Instantiate(SourceMesh);

        var meshFilter = GetComponent<MeshFilter>();
        meshFilter.sharedMesh = m_TargetMesh;
    }


    // internal, because sometimes we need to force the initialization in tests
    // before the solver is run for the 1st time with new paramaters
    internal void Initialize()
    {
        CleanUpNativeStreams();

        // If we don't have a valid Ziva asset and SkinnedMesh, tear it all down.
        if (Rig == null || SourceMesh == null)
            return;

        InitializeTargetMesh();

        if (m_SelectedSolver != null)
        {
            m_SelectedSolver.Dispose();
            m_SelectedSolver = null;
        }

        m_NumExtraParameters = Rig == null ? 0 : Rig.m_Character.NumExtraParameters;

        // Set up mapping from Ziva joints to Unity bones, based on names.
        CreateBoneIndexMap(Rig.m_Character.JointNames);

        // Animation transform streams require ziva transforms read in CreateBoneIndexMap
        InitializeAnimationStreams();

        InitializeRig();

        if (!BuildVertexOrderMapping())
            return;

        Profiler.BeginSample("BuildVertexToFaceMaps");
        int numZivaVertices = Rig.m_Character.NumVertices;
        var tangentFramesInfo = MeshTangentFramesInfo.Build(
            SourceMesh, m_VertexOrderMapping, numZivaVertices, SuppressGeometryWarningMessages);
        Profiler.EndSample();

        if (Implementation == ImplementationType.BurstJobs)
            m_SelectedSolver = new JobSolver(SchedulingMode);
        else if (Implementation == ImplementationType.ComputeShaders)
        {
#if UNITY_2021_2_OR_NEWER
            // There's a bug in Unity versions earlier than 2022.1.7 that causes the compute shader compilation to fail.
            // The solver runs, but the results are incorrect.
#if !UNITY_APPLE || (UNITY_APPLE && UNITY_2022_2_OR_NEWER)
            if (SystemInfo.supportsComputeShaders)
            {
                m_SelectedSolver = new GPUSolver();
            }
            else
#endif
#endif
            {
                Debug.LogWarning("Compute shaders are not supported on Unity versions before " +
                    "2021.2 or on Apple devices with Unity versions earlier than 2022.1.7. " +
                    "Falling back to Burst solver");
                m_SelectedSolver = new JobSolver(SchedulingMode);
                Implementation = ImplementationType.BurstJobs;
            }
        }
        else
            m_SelectedSolver = new ReferenceSolver();

        MeshFilter meshFilter = GetComponent<MeshFilter>();
        m_SelectedSolver.Init(Rig, meshFilter.sharedMesh, m_VertexOrderMapping, tangentFramesInfo, RecalculateTangentFrames,
            CalculateMotionVectors, ZivaRTPlayerManager.Instance.ShaderData);

        if (Implementation != ImplementationType.Mono)
        {
            m_ComputeBoundsJob.ReleaseBuffers();
            m_ComputeBoundsJob.Initialize(m_Centers.Length);
            m_ComputeBoundsJob.Centers.CopyFrom(m_Centers);
            m_ComputeBoundsJob.Extents.CopyFrom(m_Extents);
            m_ComputeBoundsJob.BoneIndices.CopyFrom(m_BoneIndices);
        }
        m_Reinitialize = false;
    }

    void OnDrawGizmos()
    {
        if (DrawDebugVertices && Rig != null)
        {
            var meshTransformMatrix = GetMeshTransformForRegistration();
            var zivaPoints = ConvertAndAlignZivaVertices(Rig, SourceMesh, meshTransformMatrix, out _);

            // Draw the points in the space of the Unity mesh, for easier comparison
            Gizmos.matrix = m_GameObjectRoot.localToWorldMatrix;

            var pointSize = 0.1f * m_MeshRescale;

            var up = pointSize * Vector3.up;
            var left = pointSize * Vector3.left;
            var fwd = pointSize * Vector3.forward;
            for (int i = 0; i < zivaPoints.Length; ++i)
            {
                Vector3 pt = zivaPoints[i];

                // Draw a "locator" cross centered on the point.
                Gizmos.DrawLine(pt - up, pt + up);
                Gizmos.DrawLine(pt - left, pt + left);
                Gizmos.DrawLine(pt - fwd, pt + fwd);
            }
        }
    }
    Matrix4x4 GetMeshTransformForRegistration()
    {
        Matrix4x4 meshTransformMatrix = Matrix4x4.identity;
        if (UseMeshTransformForRegistration && GameObjectRoot != null)
        {
            var meshTransform = GameObjectRoot;
            meshTransformMatrix = Matrix4x4.TRS(
                meshTransform.localPosition, meshTransform.localRotation, meshTransform.localScale);
        }

        meshTransformMatrix *= Matrix4x4.Scale(new Vector3(m_MeshRescaleRcp, m_MeshRescaleRcp, m_MeshRescaleRcp));
        return meshTransformMatrix;
    }

    void CreateBoneIndexMap(string[] zivaJointNames)
    {
        m_JointTransforms = GetComponentsInChildren<Transform>();

        Assert.IsFalse(m_TransformAccessArray.isCreated);
        m_TransformAccessArray = new TransformAccessArray(m_JointTransforms);

        Assert.IsFalse(m_ParentIndices.IsCreated);
        m_ParentIndices = new NativeArray<int>(m_JointTransforms.Length, Allocator.Persistent);

        m_ParentIndices[0] = -1;
        for (int i = 1; i < m_JointTransforms.Length; i++)
        {
            for (int j = i - 1; j >= 0; j--)
            {
                if (m_JointTransforms[j] == m_JointTransforms[i].parent)
                {
                    m_ParentIndices[i] = j;
                    break;
                }
            }
        }

        // Create bone index map so we drive the Ziva solver from the correct Unity bone motions.
        // This is currently done via joint names.
        Assert.IsFalse(m_BoneIndexMap.IsCreated);
        m_BoneIndexMap = new NativeArray<int>(zivaJointNames.Length, Allocator.Persistent);
        // Indices of unmatched Ziva joints
        List<int> unmatchedJoints = new List<int>();

        const int unmatchedJoint = -1;

        for (int j = 0; j < zivaJointNames.Length; ++j)
        {
            m_BoneIndexMap[j] = unmatchedJoint;
            var zivaJointName = zivaJointNames[j];
            for (int b = 0; b < m_JointTransforms.Length; ++b)
            {
                if (zivaJointName.Equals(m_JointTransforms[b].name))
                {
                    m_BoneIndexMap[j] = b;
                    break;
                }
            }
            if (m_BoneIndexMap[j] == unmatchedJoint)
            {
                unmatchedJoints.Add(j);
            }
        }
        if (unmatchedJoints.Count > 0)
        {
            Debug.LogWarning(string.Format(
                "Could not find corresponding Unity bones for {0} Ziva joints. Those joints will not be animated." +
                " Unmatched joints: {1}",
                unmatchedJoints.Count,
                string.Join(", ", unmatchedJoints.Select(x => zivaJointNames[x]))));

            List<int> prefixedBoneNames = new List<int>();
            for (int b = 0; b < m_JointTransforms.Length; ++b)
            {
                if (m_JointTransforms[b].name.Contains(":"))
                {
                    prefixedBoneNames.Add(b);
                }
            }
            if (prefixedBoneNames.Count > 0)
            {
                Debug.LogError(string.Format(
                "Prefixes detected in {0} Unity bones. These bones can not be matched to Ziva joints since Ziva" +
                " doesn't support prefixed joint names: {1}",
                unmatchedJoints.Count,
                string.Join(", ", prefixedBoneNames.Select(x => m_JointTransforms[x].name))));
            }
        }
    }

    static Vector3[] ConvertAndAlignZivaVertices(
        ZivaRTRig zivaAsset,
        Mesh unityMesh,
        Matrix4x4 meshLocalTransform,
        out Bounds zivaBounds)
    {
        // Move Ziva asset vertices into Unity space, and then into the local space of the Unity mesh.
        var zivaPoints = ToVector3Array(zivaAsset.m_Character.RestShape);
        var toMeshLocalSpace = meshLocalTransform.inverse;
        for (int i = 0; i < zivaPoints.Length; ++i)
        {
            zivaPoints[i] = ToUnitySpace(zivaPoints[i]);
            zivaPoints[i] = toMeshLocalSpace.MultiplyPoint3x4(zivaPoints[i]);
        }

        // Translate Ziva vertices to align bounding box centroids.
        // This allows us to ignore any global translation that may have crept in.
        var unityBounds = unityMesh.bounds;
        zivaBounds = ComputeBounds(zivaPoints);
        var centroidTranslationZivaToUnity = unityBounds.center - zivaBounds.center;
        for (int i = 0; i < zivaPoints.Length; ++i)
            zivaPoints[i] += centroidTranslationZivaToUnity;

        return zivaPoints;
    }

    int[] BuildVertexOrderMappingInternal(
        ZivaRTRig zivaAsset,
        Mesh unityMesh,
        Matrix4x4 meshLocalTransform)
    {
        if (unityMesh == null)
        {
            Debug.LogError($"Cannot build vertex order map, because no Unity mesh is associated with" +
                $" {gameObject.name} ZivaRT asset.");
            return null;
        }
        // Move Ziva vertices into Unity space, and then align them with the Unity mesh.
        var zivaPoints = ConvertAndAlignZivaVertices(zivaAsset, unityMesh, meshLocalTransform, out var zivaBounds);

        // Use a relative tolerance value based on the maximum extent of the mesh.
        var distTolerance = 1e-4f * Mathf.Max(zivaBounds.extents.x, zivaBounds.extents.y, zivaBounds.extents.z);

        // Do the vertex registration:
        var map = MeshRegistration.BuildIndexMap(unityMesh.vertices, zivaPoints, distTolerance);
        Assert.AreEqual(map.Length, unityMesh.vertexCount);

        // Check that we found a corresponding Ziva vertex for all unity vertices.
        for (int i = 0; i < map.Length; ++i)
        {
            if (map[i] == -1)
            {
                Debug.LogError(string.Format(
                    "Mesh registration failed: No corresponding Ziva vertex found for Skinned Mesh vertex #{0}: {1}." +
                        "\nZiva mesh bounds: {2} -- Unity mesh bounds: {3}" +
                        "\nZiva # vertices: {4} -- Unity # vertices: {5}",
                    i, unityMesh.vertices[i], ComputeBounds(zivaPoints), unityMesh.bounds,
                    zivaPoints.Length, unityMesh.vertexCount));
                return null;
            }
        }

        // Check whether we found a Unity vertex corresponding to each Ziva vertex.
        // This is not an error, but it may indicate an unexpected mismatch between assets.
        bool[] zivaVertMatched = new bool[zivaPoints.Length];
        for (int i = 0; i < map.Length; ++i)
        {
            zivaVertMatched[map[i]] = true;
        }
        int numZivaVertsMatched = zivaVertMatched.Count(c => c); // Count "true" entries
        if (numZivaVertsMatched < zivaPoints.Length)
        {
            Debug.LogWarningFormat(
                "{0} ZivaRT vertex positions could not be mapped to Unity Mesh vertices." +
                    "\nThese vertex positions will be computed but then ignored.",
                zivaPoints.Length - numZivaVertsMatched);
        }

        return map;
    }

    void StartDeformation()
    {
        if (m_Reinitialize)
        {
            Profiler.BeginSample("Initialize");
            Initialize();
            Profiler.EndSample();
        }

        Debug.Assert(!m_Reinitialize, "ZivaRT component not initialized!");

        // we are not initialized...
        if (m_Reinitialize)
            return;

        m_DeformationKickedThisFrame = true;

        // Read extra parameters directly into the pose buffer.
        Profiler.BeginSample("ReadExtraParameters");
        ReadExtraParameters(m_PoseNativeArray);
        Profiler.EndSample();

        //Access transforms before scheduling CalculateJointMatricesJob to avoid immediate job synchronization
        Transform rootToUse = AnimationRoot;
        if (rootToUse == null)
            rootToUse = GameObjectRoot;
        if (rootToUse == null)
            rootToUse = transform;

        var parentWorldToLocalMatrix = rootToUse.parent.worldToLocalMatrix;
        var transformRoot = transform.localToWorldMatrix;

        var jointLocalMatrices = new NativeArray<float4x4>(
            m_JointTransforms.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var calculateJointMatricesJobHandle = new CalculateJointMatricesJob
        {
            JointLocalMatrices = jointLocalMatrices,
        }.ScheduleReadOnly(m_TransformAccessArray, 256);

        var jointLocalToWorldMatrices = new NativeArray<float4x4>(
            m_JointTransforms.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        m_BoneExtractionJobHandle = new BoneExtractionJob
        {
            NumJoints = Rig.m_Character.NumJoints,
            NumExtraParameters = Rig.m_Character.NumExtraParameters,
            JointLocalMatrices = jointLocalMatrices,
            Bones = m_BonesNativeArray,
            Pose = m_PoseNativeArray,
            BoneIndexMap = m_BoneIndexMap,
            JointLocalToWorldMatrices = jointLocalToWorldMatrices,
            ParentWorldToLocalMatrix = parentWorldToLocalMatrix,
            ParentIndices = m_ParentIndices,
            TransformRoot = transformRoot,
            MeshRescale = MeshRescale,
        }.Schedule(calculateJointMatricesJobHandle);
    }

    void FinishDeformation()
    {
        // Determine which tangents to recompute.
        var tangentsToRecompute = RecalculateTangentFrames;
        if (tangentsToRecompute == RecomputeTangentFrames.NormalsAndTangents && !m_SelectedSolver.HasValidUVs)
        {
            tangentsToRecompute = RecomputeTangentFrames.NormalsOnly;
            Debug.LogWarning($"Cannot calculate full tangents on {gameObject.name} because there is no UV set.");
        }

        JobSolver.JobFuture<float> poses = new JobSolver.JobFuture<float>
        {
            dependencies = m_BoneExtractionJobHandle,
            data = m_PoseNativeArray
        };

        JobSolver.JobFuture<float> bones = new JobSolver.JobFuture<float>
        {
            dependencies = m_BoneExtractionJobHandle,
            data = m_BonesNativeArray
        };

        m_LastFrameSolved++;
        Profiler.BeginSample("StartAsyncSolve");
        m_SelectedSolver.StartAsyncSolve(
            poses,
            bones,
            tangentsToRecompute,
            EnableCorrectives,
            EnableSkinning);
        Profiler.EndSample();
    }

    void UpdateBounds()
    {
        Profiler.BeginSample("UpdateBounds");
        m_ComputeBoundsJob.RelativeTransforms = m_SelectedSolver.RelativeTransforms;
        m_ComputeBoundsJob.Run();
        m_AutoBounds.SetMinMax(m_ComputeBoundsJob.MinMax[0], m_ComputeBoundsJob.MinMax[1]);
        Profiler.EndSample();
    }

    void SyncZivaSolver()
    {
        // Wait on the solve, but make sure to only wait once per frame.
        if (m_LastFrameSynced != m_LastFrameSolved)
        {
            m_LastFrameSynced = m_LastFrameSolved;
            Profiler.BeginSample("WaitOnAsyncSolve");
            m_SelectedSolver.WaitOnAsyncSolve();
            Profiler.EndSample();

            // reference solver uses Mesh.RecalculateBounds()
            if (Implementation != ImplementationType.Mono)
            {
                if (m_UseCustomBounds)
                {
                    m_TargetMesh.bounds = m_CustomBounds;
                }
                else
                {
                    UpdateBounds();
                    m_TargetMesh.bounds = m_AutoBounds;
                }
            }
        }
    }

    /// <summary>
    /// Runs the ZivaRT solver in it's current configuration. Note that by default the solver is already run if the object
    /// containing the ZivaRT component is visible. This method should only be used in advanced scenarios where there is
    /// a requirement to explicitely invoke the solver.
    /// </summary>
    public void RunSolver()
    {
        if (ExecutionSchedule != ExecutionScheduleType.Manual)
        {
            Debug.LogError("'RunSolver()' Can only be called when 'Execution Type' is set to 'Manual'");
            return;
        }
        StartDeformation();
        FinishDeformation();
        SyncZivaSolver();
    }

    static Vector3 ToUnitySpace(Vector3 zivaVector)
    {
        zivaVector.x *= -1.0f;
        return zivaVector;
    }

    static Vector3[] ToVector3Array(float[] zivaShapeArray)
    {
        Assert.IsTrue(zivaShapeArray.Length % 3 == 0);
        Vector3[] result = new Vector3[zivaShapeArray.Length / 3];
        for (int v = 0; v < zivaShapeArray.Length / 3; ++v)
        {
            Vector3 pt = new Vector3(
                zivaShapeArray[3 * v + 0], zivaShapeArray[3 * v + 1], zivaShapeArray[3 * v + 2]);
            result[v] = pt;
        }
        return result;
    }

    static Bounds ComputeBounds(Vector3[] pts)
    {
        Assert.IsTrue(pts.Length > 0);
        var bounds = new Bounds(pts[0], Vector3.zero);
        foreach (Vector3 pt in pts)
        {
            bounds.Encapsulate(pt);
        }
        return bounds;
    }

    void InitializeAnimationStreams()
    {
        var animator = GetComponent<Animator>();
        if (animator == null)
        {
            Debug.LogWarning($"No Animator component found on ZivaRT Player component on {gameObject.name}." +
                " Component may not be initialized properly.");
            return;
        }

        Assert.IsFalse(m_ExtraParameterValues.IsCreated);
        Assert.IsFalse(m_ExtraParameterStreamHandles.IsCreated);

        // Create native arrays to store stream handles and values for extra parameters.
        m_ExtraParameterStreamHandles = new NativeArray<PropertyStreamHandle>(
            m_NumExtraParameters, Allocator.Persistent);
        m_ExtraParameterValues = new NativeArray<float>(m_NumExtraParameters, Allocator.Persistent);

        // extra a stream handle for each extra parameter.
        for (int ep = 0; ep < m_NumExtraParameters; ep++)
        {
            m_ExtraParameterStreamHandles[ep] = animator.BindStreamProperty(
                transform, typeof(Animator), Rig.m_ExtraParameterNames[ep]);
        }
    }

    void InitializeRig()
    {
        Assert.IsFalse(m_PoseNativeArray.IsCreated);
        Assert.IsFalse(m_BonesNativeArray.IsCreated);

        // Initialize our pose vector with the values from the rest pose.
        var poseLength = Rig.m_Character.PoseVectorSize;
        m_PoseNativeArray = new NativeArray<float>(poseLength, Allocator.Persistent);

        var restExtraParams = new NativeSlice<float>(m_PoseNativeArray, 0, Rig.m_Character.RestExtraParameters.Length);
        restExtraParams.CopyFrom(Rig.m_Character.RestExtraParameters);

        var restLocalTransforms = new NativeSlice<float>(
            m_PoseNativeArray, Rig.m_Character.RestExtraParameters.Length, Rig.m_Character.RestLocalTransforms.Length);
        restLocalTransforms.CopyFrom(Rig.m_Character.RestLocalTransforms);

        // Initialize our world-space bone transform array from the rest pose.
        var bonesLength = Rig.m_Character.RestWorldTransforms.Length;
        Assert.AreEqual(bonesLength / 12, Rig.m_Character.NumJoints);

        m_BonesNativeArray = new NativeArray<float>(bonesLength, Allocator.Persistent);
        m_BonesNativeArray.CopyFrom(Rig.m_Character.RestWorldTransforms);
    }

    void ReadExtraParameters(NativeArray<float> poseParams)
    {
        var animator = GetComponent<Animator>();

        // If we have no extra parameters, don't both trying to get any.
        if (m_NumExtraParameters == 0)
            return;

        // A check for the Animator will have been done at initialization, and an appropriate warning issued if not set.
        // So just return here.
        if (animator == null)
        {
            return;
        }

        if (m_AnimationWindowPreviewStarted)
        {
            // We are using the AnimationWindow. We have the m_ExtraParameterValues from the AnimationJob's ProcessAnimation()
            m_ExtraParameterValues.CopyTo(poseParams.GetSubArray(0, m_NumExtraParameters));
        }
        else
        {
            // Use an animation stream to extract the extra parameter values.
            AnimationStream stream = new AnimationStream();
            if (animator.OpenAnimationStream(ref stream))
            {
                AnimationStreamHandleUtility.ReadFloats(stream, m_ExtraParameterStreamHandles, m_ExtraParameterValues);
                m_ExtraParameterValues.CopyTo(poseParams.GetSubArray(0, m_NumExtraParameters));
                animator.CloseAnimationStream(ref stream);
            }
        }
    }

    void OnValidate()
    {
        m_Reinitialize = true;
    }

    bool m_DeformationKickedThisFrame { get; set; } = false;
    internal void ResetForFrame()
    {
        m_DeformationKickedThisFrame = false;
    }

    /// <summary>
    /// Final vertex positions after ZivaRT correctives and skinning
    /// have been applied. Not available if Compute Shader
    /// implementation is selected.
    /// </summary>
    public void GetPositions(Vector3[]positions)
    {
        Assert.IsNotNull(positions);
        Assert.AreEqual(positions.Length, m_TargetMesh.vertexCount);
        m_SelectedSolver.GetPositions(positions);
    }

    /// <summary>
    /// Final vertex normals after ZivaRT correctives and skinning
    /// have been applied. Not available if Compute Shader
    /// implementation is selected.
    /// </summary>
    public void GetNormals(Vector3[] normals)
    {
        Assert.IsNotNull(normals);
        Assert.AreEqual(normals.Length, m_TargetMesh.vertexCount);
        m_SelectedSolver.GetNormals(normals);
    }

    /// <summary>
    /// Final vertex tangents after ZivaRT correctives and skinning
    /// have been applied. Not available if Compute Shader
    /// implementation is selected.
    /// </summary>
    public void GetTangents(Vector4[] tangents)
    {
        Assert.IsNotNull(tangents);
        Assert.AreEqual(tangents.Length, m_TargetMesh.vertexCount);
        m_SelectedSolver.GetTangents(tangents);
    }

    /// <summary>
    /// Motion vectors. Not available if Compute Shader
    /// or Mono implementation is selected.
    /// </summary>
    public void GetMotionVectors(Vector3[] motionVectors)
    {
        Assert.IsNotNull(motionVectors);
        Assert.AreEqual(motionVectors.Length, m_TargetMesh.vertexCount);
        m_SelectedSolver.GetMotionVectors(motionVectors);
    }

    /// <summary>
    /// Notification callback when the Animation window starts previewing an AnimationClip.
    /// Makes extra parameters be updated when the Animation window plays or otherwise updates.
    /// </summary>
    void IAnimationWindowPreview.StartPreview()
    {
        m_AnimationWindowPreviewStarted = true;
        m_Reinitialize = true;
    }

    /// <summary>
    /// Notification callback when the Animation window stops previewing an AnimationClip.
    /// Stops extra parameters from being updated when the Animation window plays or otherwise updates.
    /// </summary>
    void IAnimationWindowPreview.StopPreview()
    {
        m_AnimationWindowPreviewStarted = false;
    }

    /// <summary>
    /// Notification callback when the Animation Window updates its PlayableGraph before sampling an AnimationClip.
    /// </summary>
    /// <param name="graph">The Animation window PlayableGraph.</param>
    void IAnimationWindowPreview.UpdatePreviewGraph(PlayableGraph graph)
    {
        m_AnimationWindowPlayable.SetJobData(m_AnimationWindowJob);
    }

    /// <summary>
    /// Appends custom Playable nodes to the Animation window PlayableGraph and sets it up to receive
    /// data from the ZivaRT Player.
    /// </summary>
    /// <param name="graph">The Animation window PlayableGraph.</param>
    /// <param name="inputPlayable">Current root of the PlayableGraph.</param>
    /// <returns></returns>
    Playable IAnimationWindowPreview.BuildPreviewGraph(PlayableGraph graph, Playable inputPlayable)
    {
        Animator animator = GetComponent<Animator>();

        m_AnimationWindowJob = new AnimationJob { ZrtPlayer = this };

        m_AnimationWindowPlayable = AnimationScriptPlayable.Create(graph, m_AnimationWindowJob, 1);
        graph.Connect(inputPlayable, 0, m_AnimationWindowPlayable, 0);

        return m_AnimationWindowPlayable;
    }
}
