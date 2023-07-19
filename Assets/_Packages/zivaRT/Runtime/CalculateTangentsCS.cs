#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.ZivaRTPlayer
{
    internal struct TangentsKernels
    {
        public static readonly string FaceTangentsName = "FaceTangents";
        public static readonly string VertexTangentsName = "VertexTangents";

        ComputeShader m_ComputeShader;

        int m_FaceTangentsKernelIdx;
        int m_VertexTangentsKernelIdx;

        int m_NumFaceWorkGroups;
        int m_NumVertexWorkGroups;

        ComputeBuffer m_Triangles;
        ComputeBuffer m_UVToTriTransforms;

        ComputeBuffer m_FaceNormals;
        ComputeBuffer m_FaceTangents;

        ComputeBuffer m_VertexNormalFaces;
        ComputeBuffer m_VertexNormalFacesStarts;
        ComputeBuffer m_VertexTangentFaces;
        ComputeBuffer m_VertexTangentFacesStarts;

        public void Release()
        {
            m_Triangles?.Release();
            m_UVToTriTransforms?.Release();
            m_FaceNormals?.Release();
            m_FaceTangents?.Release();
            m_VertexNormalFaces?.Release();
            m_VertexNormalFacesStarts?.Release();
            m_VertexTangentFaces?.Release();
            m_VertexTangentFacesStarts?.Release();
        }

        static float2x2[] ConvertTo2x2(Matrix4x4[] transforms4x4)
        {
            var transforms2x2 = new float2x2[transforms4x4.Length];
            for (int i = 0; i < transforms4x4.Length; ++i)
            {
                var transform4x4 = transforms4x4[i];
                transforms2x2[i].c0 = new float2(transform4x4.m00, transform4x4.m10);
                transforms2x2[i].c1 = new float2(transform4x4.m01, transform4x4.m11);
            }
            return transforms2x2;
        }

        public bool Init(
            Mesh mesh, MeshTangentFramesInfo tangentFramesInfo, GraphicsBuffer meshGraphicsBuffer,
            ZivaShaderData shaderData)
        {
            Release();

            int numVertices = mesh.vertexCount;
            var meshTriangles = mesh.triangles;
            int numFaces = meshTriangles.Length / 3;

            var computeShaderResource = shaderData.m_CalculateTangents;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load CalculateTangents.compute shader");
                return false;
            }

            m_ComputeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            // Set up FaceTangents kernel
            {
                m_FaceTangentsKernelIdx = m_ComputeShader.FindKernel(FaceTangentsName);
                m_ComputeShader.GetKernelThreadGroupSizes(
                    m_FaceTangentsKernelIdx, out uint faceThreadGroupSize, out uint _, out uint _);
                m_NumFaceWorkGroups = (int)Math.Ceiling((float)numFaces / faceThreadGroupSize);

                // Constant inputs to faceTangents shader:
                m_ComputeShader.SetInt("totalNumFaceTangents", numFaces);

                m_Triangles = new ComputeBuffer(numFaces, 3 * sizeof(int));
                m_Triangles.SetData(meshTriangles);
                m_ComputeShader.SetBuffer(m_FaceTangentsKernelIdx, "triangles", m_Triangles);

                m_UVToTriTransforms = new ComputeBuffer(numFaces, 2 * 2 * sizeof(float));
                var uvTriTransforms2x2 = ConvertTo2x2(tangentFramesInfo.UVToTriTransforms);
                m_UVToTriTransforms.SetData(uvTriTransforms2x2);
                m_ComputeShader.SetBuffer(m_FaceTangentsKernelIdx, "uvToTriTransforms", m_UVToTriTransforms);

                // Outputs of faceTangents shader:
                m_FaceNormals = new ComputeBuffer(numFaces, 3 * sizeof(float));
                m_ComputeShader.SetBuffer(m_FaceTangentsKernelIdx, "faceNormals", m_FaceNormals);
                m_FaceTangents = new ComputeBuffer(numFaces, 3 * 2 * sizeof(float));
                m_ComputeShader.SetBuffer(m_FaceTangentsKernelIdx, "faceTangents", m_FaceTangents);
            }

            // Set up VertexTangents kernel
            {
                m_VertexTangentsKernelIdx = m_ComputeShader.FindKernel(VertexTangentsName);
                m_ComputeShader.GetKernelThreadGroupSizes(
                    m_VertexTangentsKernelIdx, out uint vertexThreadGroupSize, out uint _, out uint _);
                m_NumVertexWorkGroups = (int)Math.Ceiling((float)numVertices / vertexThreadGroupSize);

                // Constant inputs to vertexTangents shader:
                m_ComputeShader.SetInt("totalNumVertexTangents", numVertices);
                tangentFramesInfo.GetConcatenatedNormalFaces(
                    out var normalFaces, out var normalFacesStarts);
                m_VertexNormalFaces = new ComputeBuffer(normalFaces.Length, sizeof(int));
                m_VertexNormalFaces.SetData(normalFaces);
                m_ComputeShader.SetBuffer(m_VertexTangentsKernelIdx, "vertexNormalFaces", m_VertexNormalFaces);
                m_VertexNormalFacesStarts = new ComputeBuffer(normalFacesStarts.Length, sizeof(int));
                m_VertexNormalFacesStarts.SetData(normalFacesStarts);
                m_ComputeShader.SetBuffer(
                    m_VertexTangentsKernelIdx, "vertexNormalFacesStarts", m_VertexNormalFacesStarts);

                tangentFramesInfo.GetConcatenatedTangentFaces(
                    out var tangentFaces, out var tangentFacesStarts);
                m_VertexTangentFaces = new ComputeBuffer(tangentFaces.Length, sizeof(int));
                m_VertexTangentFaces.SetData(tangentFaces);
                m_ComputeShader.SetBuffer(m_VertexTangentsKernelIdx, "vertexTangentFaces", m_VertexTangentFaces);
                m_VertexTangentFacesStarts = new ComputeBuffer(tangentFacesStarts.Length, sizeof(int));
                m_VertexTangentFacesStarts.SetData(tangentFacesStarts);
                m_ComputeShader.SetBuffer(
                    m_VertexTangentsKernelIdx, "vertexTangentFacesStarts", m_VertexTangentFacesStarts);
            }

            UpdateMeshGraphicsBuffer(mesh, meshGraphicsBuffer);

            // Hook the outputs of the faceTangents kernel to the inputs of the vertexTangents kernel.
            m_ComputeShader.SetBuffer(m_VertexTangentsKernelIdx, "inFaceNormals", m_FaceNormals);
            m_ComputeShader.SetBuffer(m_VertexTangentsKernelIdx, "inFaceTangents", m_FaceTangents);

            return true;
        }

        public void Dispatch()
        {
            m_ComputeShader.Dispatch(m_FaceTangentsKernelIdx, m_NumFaceWorkGroups, 1, 1);
            m_ComputeShader.Dispatch(m_VertexTangentsKernelIdx, m_NumVertexWorkGroups, 1, 1);
        }

        public void UpdateMeshGraphicsBuffer(Mesh mesh, GraphicsBuffer meshGraphicsBuffer)
        {
            //shared
            m_ComputeShader.SetInt("vertexStride", mesh.GetVertexBufferStride(0));

            //used by face tangents kernel
            {
                m_ComputeShader.SetInt("positionOffset", mesh.GetVertexAttributeOffset(VertexAttribute.Position));
                // Dynamically changing inputs to faceTangents shader (from a different kernel):
                m_ComputeShader.SetBuffer(m_FaceTangentsKernelIdx, "inputMesh", meshGraphicsBuffer);
            }

            //used by vertex tangents kernel
            {
                m_ComputeShader.SetInt("normalOffset", mesh.GetVertexAttributeOffset(VertexAttribute.Normal));
                m_ComputeShader.SetInt("tangentOffset", mesh.GetVertexAttributeOffset(VertexAttribute.Tangent));

                // Outputs of vertexTangents kernel
                m_ComputeShader.SetBuffer(m_VertexTangentsKernelIdx, "outputMesh", meshGraphicsBuffer);
            }
        }
    }
}
#endif
