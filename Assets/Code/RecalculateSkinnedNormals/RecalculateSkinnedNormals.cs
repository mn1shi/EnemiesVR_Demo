using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.DemoTeam.DigitalHuman;
using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class RecalculateSkinnedNormals : MonoBehaviour
{
    static class Uniforms
    {
        internal static int _AdjacencyListOffsetCount = Shader.PropertyToID("_AdjacencyListOffsetCount");
        internal static int _AdjacentTriangleIndices = Shader.PropertyToID("_AdjacentTriangleIndices");
        internal static int _TriangleCrossProd = Shader.PropertyToID("_TriangleCrossProd");
        internal static int _TriangleCrossProdRW = Shader.PropertyToID("_TriangleCrossProdRW");
        
        internal static int _PosNormalBuffer = Shader.PropertyToID("_PosNormalBuffer");
        internal static int _PosNormalBufferRW = Shader.PropertyToID("_PosNormalBufferRW");
        internal static int _IndexBuffer = Shader.PropertyToID("_IndexBuffer");
        internal static int _StridePosNormOffset = Shader.PropertyToID("_StridePosNormOffset");
        internal static int _TriangleCount = Shader.PropertyToID("_TriangleCount");
        internal static int _VertexCount = Shader.PropertyToID("_VertexCount");
        
    }

    private MeshBuffers meshBuffers;
    private MeshAdjacency meshAdjacency;
   

    private ComputeBuffer adjacentTriangleIndicesOffsetCountBuffer;
    private ComputeBuffer adjacentTriangleIndicesBuffer;
    private ComputeBuffer crossProdPerTriangleBuffer;
    private ComputeBuffer indexBuffer;
    
    private ComputeShader recalculateNormalsCS;
    private int kCalculateCrossProductPerTriangle;
    private int kRecalculateNormalsKernel;

    private const int KERNEL_SIZE = 64;
    private void OnEnable()
    {
        Mesh mesh = null;
        var smr = GetComponent<SkinnedMeshRenderer>();
        if (smr && smr.sharedMesh)
        {
            mesh = smr.sharedMesh;
        }
        
        var mr = GetComponent<MeshFilter>();
        if (mesh == null)
        {
            if (mr && mr.sharedMesh)
            {
                mesh = mr.sharedMesh;
                
            }
        }

        if (mesh == null) return;

        if (!Initialize(mesh)) return;
        if (smr != null)
        {
            smr.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
        } else if (mr != null)
        {
            mesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
            
        }
        

        RenderPipelineManager.beginFrameRendering -= AfterGpuSkinningCallback;
        RenderPipelineManager.beginFrameRendering += AfterGpuSkinningCallback;
        
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginFrameRendering -= AfterGpuSkinningCallback;
        Deinitialize();
    }

    void Deinitialize()
    {
        crossProdPerTriangleBuffer.Dispose();
        adjacentTriangleIndicesOffsetCountBuffer.Dispose();
        adjacentTriangleIndicesBuffer.Dispose();
        indexBuffer.Dispose();
    }
    
    bool Initialize(Mesh mesh)
    {
        recalculateNormalsCS = Resources.Load<ComputeShader>("RecalculateSkinnedNormals");
        kCalculateCrossProductPerTriangle = recalculateNormalsCS.FindKernel("kCalculateCrossProductPerTriangle");
        kRecalculateNormalsKernel= recalculateNormalsCS.FindKernel("kRecalculateNormals");
        if (recalculateNormalsCS == null || kRecalculateNormalsKernel == -1 || kCalculateCrossProductPerTriangle == -1) return false;

        meshBuffers = new MeshBuffers(mesh);
        meshAdjacency = new MeshAdjacency(meshBuffers, false);

        crossProdPerTriangleBuffer = new ComputeBuffer(meshAdjacency.triangleCount, sizeof(float) * 3, ComputeBufferType.Structured);
        adjacentTriangleIndicesOffsetCountBuffer = new ComputeBuffer(meshAdjacency.vertexCount, sizeof(uint) * 2, ComputeBufferType.Structured);
        adjacentTriangleIndicesBuffer = new ComputeBuffer(meshAdjacency.vertexTriangles.itemCount, sizeof(uint),
            ComputeBufferType.Structured);
        indexBuffer = new ComputeBuffer(meshAdjacency.triangleCount * 3, sizeof(uint), ComputeBufferType.Structured);

        NativeArray<uint> indexOffsetArray = new NativeArray<uint>(meshAdjacency.vertexCount * 2, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        NativeArray<uint> adjacentTriangleIndicesArray = new NativeArray<uint>(meshAdjacency.vertexTriangles.itemCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        int[] indicesArray = mesh.GetIndices(0);
        //upload data
        uint currentTriangleIndexOffset = 0;
        for (int i = 0; i != meshAdjacency.vertexCount; i++)
        {
            uint triangleOffset = 0;
            foreach (int triangle in meshAdjacency.vertexTriangles[i])
            {
                adjacentTriangleIndicesArray[(int)(currentTriangleIndexOffset + triangleOffset)] = (uint)triangle;
                ++triangleOffset;
            }
            
            indexOffsetArray[i * 2] = currentTriangleIndexOffset;
            indexOffsetArray[i * 2 + 1] = triangleOffset;

            currentTriangleIndexOffset += triangleOffset;
        }

        adjacentTriangleIndicesOffsetCountBuffer.SetData(indexOffsetArray);
        adjacentTriangleIndicesBuffer.SetData(adjacentTriangleIndicesArray);
        indexBuffer.SetData(indicesArray);
        
        indexOffsetArray.Dispose();
        adjacentTriangleIndicesArray.Dispose();
        
        return true;
    }
    void AfterGpuSkinningCallback(ScriptableRenderContext scriptableRenderContext, Camera[] cameras)
    {
        if (crossProdPerTriangleBuffer == null || !crossProdPerTriangleBuffer.IsValid()) return;
        var smr = GetComponent<SkinnedMeshRenderer>();
        var mf = GetComponent<MeshFilter>();
        if (smr == null && mf == null) return; 

        Mesh skinMesh = smr ? smr.sharedMesh : mf.sharedMesh;
        int positionStream = skinMesh.GetVertexAttributeStream(VertexAttribute.Position);
        int normalStream = skinMesh.GetVertexAttributeStream(VertexAttribute.Normal);

        if (positionStream != normalStream)
        {
            Debug.LogError(
                "RecalculateSkinnedNormals requires that the skin has it's positions and normals in the same vertex buffer/stream.");
            return;
        }

        using GraphicsBuffer vertexBuffer = smr ? smr.GetVertexBuffer() : mf.sharedMesh.GetVertexBuffer(positionStream);
        if (vertexBuffer == null)
        {
            return;
        }

        int[] skinVertexBufferStrideAndOffsets =
        {
            skinMesh.GetVertexBufferStride(positionStream),
            skinMesh.GetVertexAttributeOffset(VertexAttribute.Position),
            skinMesh.GetVertexAttributeOffset(VertexAttribute.Normal)
        };


        CommandBuffer cmd = new CommandBuffer();
        cmd.name = "Recalculate Skinned Normals";
        cmd.BeginSample("Recalculate Skinned Normals");
        //calculate normal and area per triangle
        {
            cmd.SetComputeIntParams(recalculateNormalsCS, Uniforms._StridePosNormOffset, skinVertexBufferStrideAndOffsets);
            cmd.SetComputeIntParam(recalculateNormalsCS, Uniforms._VertexCount, meshAdjacency.vertexCount);
            cmd.SetComputeIntParam(recalculateNormalsCS, Uniforms._TriangleCount, meshAdjacency.triangleCount);

            cmd.SetComputeBufferParam(recalculateNormalsCS, kCalculateCrossProductPerTriangle, Uniforms._TriangleCrossProdRW, crossProdPerTriangleBuffer);
            cmd.SetComputeBufferParam(recalculateNormalsCS, kCalculateCrossProductPerTriangle, Uniforms._PosNormalBuffer, vertexBuffer);
            cmd.SetComputeBufferParam(recalculateNormalsCS, kCalculateCrossProductPerTriangle, Uniforms._IndexBuffer, indexBuffer);

            int workGroups = (meshAdjacency.triangleCount + KERNEL_SIZE - 1) / KERNEL_SIZE;
            cmd.DispatchCompute(recalculateNormalsCS, kCalculateCrossProductPerTriangle, workGroups, 1, 1);
        }


        {
            cmd.SetComputeBufferParam(recalculateNormalsCS, kRecalculateNormalsKernel, Uniforms._TriangleCrossProd, crossProdPerTriangleBuffer);
            cmd.SetComputeBufferParam(recalculateNormalsCS, kRecalculateNormalsKernel, Uniforms._AdjacentTriangleIndices, adjacentTriangleIndicesBuffer);
            cmd.SetComputeBufferParam(recalculateNormalsCS, kRecalculateNormalsKernel, Uniforms._AdjacencyListOffsetCount, adjacentTriangleIndicesOffsetCountBuffer);
            
            cmd.SetComputeBufferParam(recalculateNormalsCS, kRecalculateNormalsKernel, Uniforms._PosNormalBufferRW, vertexBuffer);

            int workGroups = (meshAdjacency.vertexCount + KERNEL_SIZE - 1) / KERNEL_SIZE;
            cmd.DispatchCompute(recalculateNormalsCS, kRecalculateNormalsKernel, workGroups, 1, 1);
        }

        
        cmd.EndSample("Recalculate Skinned Normals");

        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Release();
    }
    
}
