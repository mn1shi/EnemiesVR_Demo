#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.ZivaRTPlayer
{
    internal struct SkinningKernel
    {
        static readonly string k_Name = "Skinning";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_InfluenceStarts;
        ComputeBuffer m_InfluenceIndices;
        ComputeBuffer m_SkinningWeights;

        ComputeBuffer m_RelativeJointTransforms;

        ComputeBuffer m_SkinnedPositions;
        public ComputeBuffer SkinnedPositions { get { return m_SkinnedPositions; } }

        public void Release()
        {
            m_InfluenceStarts?.Release();
            m_InfluenceIndices?.Release();
            m_SkinningWeights?.Release();
            m_RelativeJointTransforms?.Release();
            m_SkinnedPositions?.Release();
        }

        public bool Init(ZivaRTRig rig, ComputeBuffer preSkinningPositions, ZivaShaderData shaderData)
        {
            Release();

            int numVertices = rig.m_Character.NumVertices;

            var computeShaderResource = shaderData.m_Skinning;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load Skinning.compute shader");
                return false;
            }

            m_ComputeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)numVertices / threadGroupSize);

            // Constant inputs to skinning shader:
            m_ComputeShader.SetInt("totalNumSkinnedPositions", numVertices);

            var weightsMatrix = rig.m_Skinning.SkinningWeights;
            m_InfluenceStarts = new ComputeBuffer(weightsMatrix.ColStarts.Length, sizeof(int));
            m_InfluenceStarts.SetData(weightsMatrix.ColStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "influenceStarts", m_InfluenceStarts);
            m_InfluenceIndices = new ComputeBuffer(weightsMatrix.RowIndices.Length, sizeof(int));
            m_InfluenceIndices.SetData(weightsMatrix.RowIndices);
            m_ComputeShader.SetBuffer(m_KernelIdx, "influenceIndices", m_InfluenceIndices);
            m_SkinningWeights = new ComputeBuffer(weightsMatrix.W.Length, sizeof(float));
            m_SkinningWeights.SetData(weightsMatrix.W);
            m_ComputeShader.SetBuffer(m_KernelIdx, "weights", m_SkinningWeights);

            // Dynamically changing inputs to skinning shader:
            Assert.AreEqual(rig.m_Skinning.RestPoseInverse.Length / 12, weightsMatrix.NumRows);
            m_RelativeJointTransforms = new ComputeBuffer(weightsMatrix.NumRows, 12 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "jointTransforms", m_RelativeJointTransforms);

            // Output of skinning shader:
            Assert.AreEqual(weightsMatrix.NumCols, rig.m_Character.NumVertices);
            m_SkinnedPositions = new ComputeBuffer(weightsMatrix.NumCols, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "skinnedPositions", m_SkinnedPositions);

            // Inputs received from other kernels:
            Assert.AreEqual(preSkinningPositions.count, rig.m_Character.NumVertices);
            Assert.AreEqual(preSkinningPositions.stride, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "preskinningPositions", preSkinningPositions);

            return true;
        }

        public void UpdateRelativeJointTransforms(NativeArray<float3x4> transformsRelativeToRestPose)
        {
            Assert.IsTrue(m_RelativeJointTransforms.IsValid());
            m_RelativeJointTransforms.SetData(transformsRelativeToRestPose);
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }
}
#endif
