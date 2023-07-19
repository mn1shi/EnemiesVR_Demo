#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.ZivaRTPlayer
{
    internal struct RemapVerticesKernel
    {
        ComputeShader m_ComputeShader;

        readonly static string k_Name = "RemapVertices";
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_IndexMap;

        public void Release()
        {
            m_IndexMap?.Release();
        }

        public bool Init(
            int[] vertexIndexMap, Mesh mesh, ComputeBuffer inVertices, GraphicsBuffer meshGraphicsBuffer,
            ZivaShaderData shaderData)
        {
            Release();

            int numVertices = vertexIndexMap.Length;

            var computeShaderResource = shaderData.m_RemapVertices;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load RemapVertices.compute shader");
                return false;
            }
            m_ComputeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)numVertices / threadGroupSize);

            // Static input:
            m_ComputeShader.SetInt("totalNumRemappedVertices", numVertices);

            m_IndexMap = new ComputeBuffer(numVertices, sizeof(int));
            m_IndexMap.SetData(vertexIndexMap);
            m_ComputeShader.SetBuffer(m_KernelIdx, "indexMap", m_IndexMap);

            UpdateMeshGraphicsBuffer(mesh, meshGraphicsBuffer);

            // Input (from other shader)
            m_ComputeShader.SetBuffer(m_KernelIdx, "inVertices", inVertices);

            return true;
        }

        public void UpdateMeshGraphicsBuffer(Mesh mesh, GraphicsBuffer meshGraphicsBuffer)
        {
            m_ComputeShader.SetInt("positionOffset", mesh.GetVertexAttributeOffset(VertexAttribute.Position));
            m_ComputeShader.SetInt("vertexStride", mesh.GetVertexBufferStride(0));

            // Output
            m_ComputeShader.SetBuffer(m_KernelIdx, "remappedVertices", meshGraphicsBuffer);
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }
}
#endif
