#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using UnityEngine;

namespace Unity.ZivaRTPlayer
{
    internal struct MotionVectorsKernel
    {
        static readonly string s_Name = "MotionVectors";
        static readonly string s_FirstTimeName = "FirstTimeMotionVectors";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_FirstTimeKernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_PreviousPositions;
        public void Release()
        {
            m_PreviousPositions?.Release();
        }

        public bool Init(
            int vertexCount, GraphicsBuffer motionVectorGraphicsBuffer, int motionVectorOffset,
            GraphicsBuffer meshGraphicsBuffer, int positionOffset, ZivaShaderData shaderData)
        {
            Release();

            var computeShaderResource = shaderData.m_ComputeMotionVectors;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load ComputeMotionVectors.compute");
                return false;
            }

            m_ComputeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            int numVertices = vertexCount;

            m_KernelIdx = m_ComputeShader.FindKernel(s_Name);
            m_FirstTimeKernelIdx = m_ComputeShader.FindKernel(s_FirstTimeName);
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)numVertices / threadGroupSize);

            m_ComputeShader.SetInt("totalNumVertices", numVertices);

            UpdateMeshGraphicsBuffer(motionVectorGraphicsBuffer, motionVectorOffset, meshGraphicsBuffer, positionOffset);

            m_PreviousPositions = new ComputeBuffer(numVertices, 3 * sizeof(float));            
            m_ComputeShader.SetBuffer(m_KernelIdx, "previousPositions", m_PreviousPositions);
            m_ComputeShader.SetBuffer(m_FirstTimeKernelIdx, "previousPositions", m_PreviousPositions);

            return true;
        }

        public void Dispatch(bool firstTime)
        {
            m_ComputeShader.Dispatch(firstTime ? m_FirstTimeKernelIdx : m_KernelIdx, m_NumWorkGroups, 1, 1);
        }

        public void UpdateMeshGraphicsBuffer(GraphicsBuffer motionVectorGraphicsBuffer, int motionVectorOffset,
            GraphicsBuffer meshGraphicsBuffer, int positionOffset)
        {
            m_ComputeShader.SetInt("vertexStride", meshGraphicsBuffer.stride);
            m_ComputeShader.SetInt("positionOffset", positionOffset);
            m_ComputeShader.SetBuffer(m_KernelIdx, "currentPositions", meshGraphicsBuffer);
            m_ComputeShader.SetBuffer(m_FirstTimeKernelIdx, "currentPositions", meshGraphicsBuffer);

            m_ComputeShader.SetInt("motionVertexStride", motionVectorGraphicsBuffer.stride);
            m_ComputeShader.SetInt("motionVectorOffset", motionVectorOffset);
            m_ComputeShader.SetBuffer(m_KernelIdx, "motionVectors", motionVectorGraphicsBuffer);
            m_ComputeShader.SetBuffer(m_FirstTimeKernelIdx, "motionVectors", motionVectorGraphicsBuffer);
        }
    }
}
#endif
