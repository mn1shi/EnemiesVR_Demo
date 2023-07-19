#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace Unity.ZivaRTPlayer
{
    internal class GPUSolver : Solver
    {
        ZivaRTRig m_Rig;

        CopyBufferKernel m_CopyBufferKernel;
        CorrectivesKernels m_CorrectivesKernels;
        SkinningKernel m_SkinningKernel;
        RemapVerticesKernel m_RemapKernel;

        bool m_CalculateMotionVectors;
        MotionVectorsKernel m_MotionVectorsKernel;

        TangentsKernels m_TangentsKernels;
        RecomputeTangentFrames m_RecomputeTangentFrames;

        GraphicsBuffer m_MeshGraphicsBuffer;
        RelativeTransformsCalculator m_RelativeTransformsCalculator;
        GraphicsBuffer m_MotionVectorGraphicsBuffer;

        const int k_ZivaRTStream = 0;   // Vertex buffer index where our shaders expect positions
                                        // normals and tangents to be. Motion vectors will go there
                                        // too if not present in source mesh.

        public override NativeArray<float3x4> RelativeTransforms
        {
            get
            {
                return m_RelativeTransformsCalculator.RelativeTransforms;
            }
        }

        // Refactor the target mesh into format that is expected by our compute shader
        // Positions, Normals and Tangents if required have to be on stream 0 and Texcoord5 
        // attribute needs to exist if motion vector calculation is required
        static internal void CreateRequiredAttributes(Mesh targetMesh, RecomputeTangentFrames recomputeTangentFrames,
            bool calculateMotionVectors)
        {      
            VertexAttributeDescriptor[] newAttributes = new VertexAttributeDescriptor[3]; // normals, tangents, movecs
            int newAttributesCount = VertexBufferInterface.SetupStream(newAttributes, targetMesh, recomputeTangentFrames,
                calculateMotionVectors, k_ZivaRTStream);
           
            VertexAttributeDescriptor[] currentAttributes = targetMesh.GetVertexAttributes();
            // remap Positions, Normals and Tangents (if exist) to be on stream 0
            for (int i = 0; i < currentAttributes.Length; i++)
            {
                if ((currentAttributes[i].attribute == VertexAttribute.Position) ||
                   (currentAttributes[i].attribute == VertexAttribute.Normal) ||
                   (currentAttributes[i].attribute == VertexAttribute.Tangent))
                {
                    currentAttributes[i].stream = k_ZivaRTStream;
                }
            }

            VertexAttributeDescriptor[] finalAttributes = currentAttributes;
            VertexBufferInterface.CombineAttributes(ref finalAttributes, currentAttributes, newAttributes, newAttributesCount);

            // update mesh format, this also preserves old attribute values
            targetMesh.SetVertexBufferParams(targetMesh.vertexCount, finalAttributes);
        }

        public override bool Init(
            ZivaRTRig rig,
            Mesh targetMesh,
            int[] vertexIndexMap,
            MeshTangentFramesInfo tangentFramesInfo,
            RecomputeTangentFrames recomputeTangentFrames,
            bool calculateMotionVectors,
            ZivaShaderData shaderData)
        {
            // Uncomment this if you need to debug the source of native memory leaks:
            // Unity.Collections.NativeLeakDetection.Mode =
            // Unity.Collections.NativeLeakDetectionMode.EnabledWithStackTrace;

            Assert.IsTrue(SystemInfo.supportsComputeShaders);

            CreateRequiredAttributes(targetMesh, recomputeTangentFrames, calculateMotionVectors);
            if (!base.Init(rig, targetMesh, vertexIndexMap, tangentFramesInfo, recomputeTangentFrames, calculateMotionVectors, shaderData))
                return false;

            Assert.AreEqual(vertexIndexMap.Length, targetMesh.vertexCount);

            // We only currently support Tensor Skinning in the GPU solver.
            if (rig.m_CorrectiveType != CorrectiveType.TensorSkin)
            {
                Debug.LogWarning("Unsupported corrective type for GPU solver!");
                return false;
            }

            m_Rig = rig;
           
            targetMesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
            m_MeshGraphicsBuffer = targetMesh.GetVertexBuffer(k_ZivaRTStream);

            // ZRT data stores rest positions as float[], convert to float3[]
            var restPositions = new float3[rig.m_Character.NumVertices];
            for (int i = 0; i < restPositions.Length; ++i)
            {
                restPositions[i] = new float3(
                    rig.m_Character.RestShape[3 * i + 0],
                    rig.m_Character.RestShape[3 * i + 1],
                    rig.m_Character.RestShape[3 * i + 2]);
            }

            bool success;

            success = m_CopyBufferKernel.Init(restPositions, shaderData);
            if (!success)
                return false;

            success = m_CorrectivesKernels.Init(rig, m_CopyBufferKernel.DstPositionBuffer, shaderData);
            if (!success)
                return false;

            success = m_SkinningKernel.Init(rig, m_CopyBufferKernel.DstPositionBuffer, shaderData);
            if (!success)
                return false;

            success = m_RemapKernel.Init(vertexIndexMap, targetMesh, m_SkinningKernel.SkinnedPositions, m_MeshGraphicsBuffer, shaderData);
            if (!success)
                return false;

            this.m_CalculateMotionVectors = calculateMotionVectors;
            if (calculateMotionVectors)
            {
                m_MotionVectorGraphicsBuffer = targetMesh.GetVertexBuffer(
                    targetMesh.GetVertexAttributeStream(VertexAttribute.TexCoord5));
                success = m_MotionVectorsKernel.Init(
                    targetMesh.vertexCount,
                    m_MotionVectorGraphicsBuffer,
                    targetMesh.GetVertexAttributeOffset(VertexAttribute.TexCoord5),
                    m_MeshGraphicsBuffer,
                    targetMesh.GetVertexAttributeOffset(VertexAttribute.Position),
                    shaderData);
                if (!success)
                    return false;
            }

            success = m_TangentsKernels.Init(targetMesh, tangentFramesInfo, m_MeshGraphicsBuffer, shaderData);
            if (!success)
                return false;

            m_RelativeTransformsCalculator = new RelativeTransformsCalculator(m_Rig);
            m_IsFirstTime = true;

            return true;
        }

        public override void Dispose()
        {
            m_CopyBufferKernel.Release();
            m_CorrectivesKernels.Release();
            m_SkinningKernel.Release();
            m_RemapKernel.Release();
            m_MotionVectorsKernel.Release();

            m_TangentsKernels.Release();
            m_MeshGraphicsBuffer.Dispose();
            m_RelativeTransformsCalculator.Dispose();
            m_MotionVectorGraphicsBuffer?.Dispose();
        }

        public override void StartAsyncSolve(
            JobSolver.JobFuture<float> currentPose,
            JobSolver.JobFuture<float> jointWorldTransforms,
            RecomputeTangentFrames recomputeTangentFramesToUse,
            bool doCorrectives,
            bool doSkinning)
        {
            m_RecomputeTangentFrames = recomputeTangentFramesToUse;

            // Copy rest positions into accumulation buffer.
            m_CopyBufferKernel.Dispatch();

            if (doCorrectives)
            {
                m_CorrectivesKernels.Dispatch(currentPose);
            }

            if (doSkinning)
            {
                DispatchSkinning(jointWorldTransforms);
            }
            else
            {
                // Convert pre-skinning positions to unity space, since our skinning step handles
                // that conversion for us.
                // SLOW! This is for debugging.
                var shape = new Vector3[m_Rig.m_Character.NumVertices];
                m_CopyBufferKernel.DstPositionBuffer.GetData(shape);
                ToUnitySpace(ref shape);
                m_SkinningKernel.SkinnedPositions.SetData(shape);
            }

            m_RemapKernel.Dispatch();

            if (m_RecomputeTangentFrames != RecomputeTangentFrames.None)
            {
                m_TangentsKernels.Dispatch();
            }

            if (m_CalculateMotionVectors)
            {
                m_MotionVectorsKernel.Dispatch(m_IsFirstTime);
                m_IsFirstTime = false;
            }
        }

        public override void GetPositions(Vector3[] positions)
        {
            Debug.LogError("Positions not available when using the Compute solver.");
        }

        public override void GetNormals(Vector3[] normals)
        {
            Debug.LogError("Normals not available when using the Compute solver.");
        }

        public override void GetTangents(Vector4[] tangents)
        {
            Debug.LogError("Tangents not available when using the Compute solver.");
        }

        public override void GetMotionVectors(Vector3[] motionVectors)
        {
            Debug.LogError("Motion vectors not available when using the Mono solver.");
        }
             
        public override void WaitOnAsyncSolve()
        {
        }

        void DispatchSkinning(NativeArray<float> worldTransformsFlattened)
        {
            Profiler.BeginSample("UpdateJointTransforms");
            m_SkinningKernel.UpdateRelativeJointTransforms(
                m_RelativeTransformsCalculator.WorldToRelative(worldTransformsFlattened));
            Profiler.EndSample();

            Profiler.BeginSample("DispatchSkinning");
            m_SkinningKernel.Dispatch();
            Profiler.EndSample();
        }

        static void ToUnitySpace(ref Vector3[] shape)
        {
            for (int v = 0; v < shape.Length; ++v)
            {
                // Invert x-component of each vertex
                shape[v].x *= -1.0f;
            }
        }
    }
}
#endif
