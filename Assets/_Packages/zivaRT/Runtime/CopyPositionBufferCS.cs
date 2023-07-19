#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.ZivaRTPlayer
{
    internal struct CopyBufferKernel
    {
        static readonly string k_Name = "CopyPositionBuffer";
        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_SrcPosBuffer;
        ComputeBuffer m_DstPosBuffer;
        public ComputeBuffer DstPositionBuffer { get { return m_DstPosBuffer; } }

        public void Release()
        {
            m_SrcPosBuffer?.Release();
            m_DstPosBuffer?.Release();
        }

        public bool Init(float3[] positionsToCopy, ZivaShaderData shaderData)
        {
            Release();

            int numVertices = positionsToCopy.Length;

            var computeShaderResource = shaderData.m_CopyPositionBuffer;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load CopyPositionBuffer.compute shader");
                return false;
            }

            m_ComputeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)numVertices / threadGroupSize);

            m_ComputeShader.SetInt("copyBufferLength", numVertices);

            m_SrcPosBuffer = new ComputeBuffer(numVertices, 3 * sizeof(float));
            m_SrcPosBuffer.SetData(positionsToCopy);
            m_ComputeShader.SetBuffer(m_KernelIdx, "srcPosBuffer", m_SrcPosBuffer);

            m_DstPosBuffer = new ComputeBuffer(numVertices, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "dstPosBuffer", m_DstPosBuffer);

            return true;
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }
}
#endif
