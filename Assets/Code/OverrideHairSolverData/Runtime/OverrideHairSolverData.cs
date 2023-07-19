using System;
using System.Collections;
using System.Collections.Generic;
using Unity.DemoTeam.DigitalHuman;
using Unity.DemoTeam.Hair;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

[ExecuteAlways]
public class OverrideHairSolverData : MonoBehaviour
{
    static class Uniforms
    {
        internal static int _MeshPositions = Shader.PropertyToID("_MeshPositions");
        internal static int _StrideOffsetMeshPositions = Shader.PropertyToID("_StrideOffsetMeshPositions");

    }

    public HairInstance hairInstanceTarget;
    public MeshFilter overrideMesh;



    private static ComputeShader s_shader;
    private static int s_overrideParticlePositionsSolverKernel = -1;
    private static int s_overrideParticlePositionsStagingKernel = -1;
    private const int KERNEL_SIZE = 64;
    private const string OVERRIDE_CONTAINER_NAME = "HairSolverOverrideAttachment";

    static bool EnsureResources()
    {
        if (s_shader == null)
        {
            s_shader = Resources.Load<ComputeShader>("OverrideHairSolverDataKernels");
            s_overrideParticlePositionsSolverKernel = -1;
            s_overrideParticlePositionsStagingKernel = -1;
        }

        if (!s_shader) return false;

        if (s_overrideParticlePositionsSolverKernel == -1)
        {
            s_overrideParticlePositionsSolverKernel = s_shader.FindKernel("kOverrideParticlePositionsSolverData");
            
            
        }
        
        if(s_overrideParticlePositionsStagingKernel == -1)
        {
            s_overrideParticlePositionsStagingKernel = s_shader.FindKernel("kOverrideParticlePositionsStaging");
        }

        return true;
    }

    private void OnEnable()
    {
        if (!EnsureResources())
            return;
        
        
        if (hairInstanceTarget && overrideMesh && overrideMesh.sharedMesh)
        {
            hairInstanceTarget.onRenderingStateChanged -= onRenderingStateChangedCallback;
            hairInstanceTarget.onRenderingStateChanged += onRenderingStateChangedCallback;
        }
        
        
    }

    private void OnDisable()
    {
        if (hairInstanceTarget)
        {
            hairInstanceTarget.onRenderingStateChanged -= onRenderingStateChangedCallback;
        }
    }

    void onRenderingStateChangedCallback(CommandBuffer cmd)
    {
        if (!EnsureResources())
            return;

        if (!hairInstanceTarget || !overrideMesh || !overrideMesh.sharedMesh) return;

        CopyMeshPositionsToHairSolvedPositions(hairInstanceTarget, overrideMesh.sharedMesh, cmd);

        for (int i = 0; i < hairInstanceTarget.strandGroupInstances.Length; ++i)
        {
            if (hairInstanceTarget.strandGroupInstances[i].sceneObjects.strandMeshRenderer != null)
            {
                hairInstanceTarget.strandGroupInstances[i].sceneObjects.strandMeshRenderer.localBounds =
                    overrideMesh.sharedMesh.bounds;
            }
        }
  
    }

    static void CopyMeshPositionsToHairSolvedPositions(HairInstance instance, Mesh meshToCopyFrom, CommandBuffer cmd)
    {
        if (instance.solverData == null || instance.solverData.Length == 0)
            return;

        
        ref HairSim.SolverData solverData = ref instance.solverData[0];

        int copyKernel = instance.GetSettingsStrands(instance.strandGroupInstances[0]).staging ? s_overrideParticlePositionsStagingKernel : s_overrideParticlePositionsSolverKernel;
        
        HairSim.BindSolverData(cmd, s_shader, copyKernel, solverData);


        var particlesPerStrand = (int)solverData.cbuffer._StrandParticleCount;
        var strandCount = (int)solverData.cbuffer._StrandCount;

        var overallParticleCount = particlesPerStrand * strandCount;
        if (overallParticleCount != meshToCopyFrom.vertexCount)
        {
            Debug.LogError(
                "solver particle count and the override mesh vertex count do not match, the solver positions will not be overridden");
            return;
        }

        if ((meshToCopyFrom.vertexBufferTarget & GraphicsBuffer.Target.Raw) == 0)
        {
            meshToCopyFrom.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
        }

        int posStream = meshToCopyFrom.GetVertexAttributeStream(VertexAttribute.Position);
        GraphicsBuffer vertexData = meshToCopyFrom.GetVertexBuffer(posStream);
        int[] posStrideOffset =
        {
            meshToCopyFrom.GetVertexBufferStride(posStream),
            meshToCopyFrom.GetVertexAttributeOffset(VertexAttribute.Position)
        };

        cmd.SetComputeIntParams(s_shader, Uniforms._StrideOffsetMeshPositions, posStrideOffset);
        cmd.SetComputeBufferParam(s_shader, copyKernel, Uniforms._MeshPositions, vertexData);

        int workItems = instance.GetSettingsStrands(instance.strandGroupInstances[0]).staging ? strandCount : overallParticleCount;
        var workGroups = (workItems + KERNEL_SIZE - 1) / KERNEL_SIZE;
        cmd.DispatchCompute(s_shader, copyKernel, workGroups, 1, 1);

        vertexData.Dispose();
    }

}