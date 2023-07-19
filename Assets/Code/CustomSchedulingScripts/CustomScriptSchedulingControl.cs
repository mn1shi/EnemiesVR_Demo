using System;
using System.Collections;
using System.Collections.Generic;
using Unity.DemoTeam.DigitalHuman;
using Unity.DemoTeam.Hair;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.PlayerLoop;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

[ExecuteAlways]
public class CustomScriptSchedulingControl : MonoBehaviour
{
    public SkinAttachmentTarget skinAttachmentTarget;
    public MeshToSDF[] sdfGeneratorsToSchedule;
    public HairInstance[] asyncScheduledHairInstances;
    
    private bool succesfullyHookedToPlayerLoop = false;
    private HairSimSyncAfterDepthNormalCustomPass customPass;
    private GameObject customPassGO;
    private GraphicsFence beforeAsyncHairSimFence;

    private void AttachToPlayerLoop()
    {
        var rootLoop = PlayerLoop.GetCurrentPlayerLoop();
        var newList = new List<PlayerLoopSystem>();
        bool PrelateUpdatePassed = false;
        for (int i = 0; i < rootLoop.subSystemList.Length; i++)
        {
            var type = rootLoop.subSystemList[i].type;

            newList.Add(rootLoop.subSystemList[i]);

            if (type == typeof(PreLateUpdate))
            {
                PrelateUpdatePassed = true;
            }

            //hook ourselves right after zivas lateUpdate
            if (type != null && type.Name == "ZivaRTPlayerManager" && PrelateUpdatePassed)
            {
                PlayerLoopSystem s = default;
                s.updateDelegate += AfterZivaSkinDeform;
                s.type = GetType();
                newList.Add(s);
                succesfullyHookedToPlayerLoop = true;
            }
        }

        if (succesfullyHookedToPlayerLoop)
        {
            rootLoop.subSystemList = newList.ToArray();
            PlayerLoop.SetPlayerLoop(rootLoop);
        }
        
    }

    private void DetachFromPlayerLoop()
    {
        var rootLoop = PlayerLoop.GetCurrentPlayerLoop();
        var newList = new List<PlayerLoopSystem>();
        for (int i = 0; i < rootLoop.subSystemList.Length; i++)
        {
            var type = rootLoop.subSystemList[i].type;
            var del = rootLoop.subSystemList[i].updateDelegate;
            if (del != AfterZivaSkinDeform)
                newList.Add(rootLoop.subSystemList[i]);
        }

        rootLoop.subSystemList = newList.ToArray();
        PlayerLoop.SetPlayerLoop(rootLoop);
        succesfullyHookedToPlayerLoop = false;
    }

    private void AfterZivaSkinDeform()
    {
        if (skinAttachmentTarget == null || !skinAttachmentTarget.isActiveAndEnabled) return;
        skinAttachmentTarget.ExecuteSkinAttachmentResolveAutomatically = false;
        skinAttachmentTarget.Resolve();
    }

    bool HasAsyncHairInstancesToSchedule()
    {
        if (asyncScheduledHairInstances == null || asyncScheduledHairInstances.Length == 0) return false;

        bool hasValidInstances = false;

        foreach (var instance in asyncScheduledHairInstances)
        {
            if (instance != null && instance.enabled && instance.gameObject.activeInHierarchy)
            {
                hasValidInstances = true;
                break;
            }
                
        }

        return hasValidInstances;

    }
    
    private void LateUpdate()
    {
        if (!succesfullyHookedToPlayerLoop)
        {
            AttachToPlayerLoop();
        }

        bool scheduleAsyncHair = HasAsyncHairInstancesToSchedule();
        
        CommandBuffer cmd = CommandBufferPool.Get("Mesh2SDF");
        
        if (sdfGeneratorsToSchedule != null)
        {
            foreach (var sdf in sdfGeneratorsToSchedule)
            {
                sdf.updateMode = MeshToSDF.UpdateMode.Explicit;
                sdf.UpdateSDF(cmd);
            }
        }

        if (scheduleAsyncHair)
        {
            beforeAsyncHairSimFence = cmd.CreateAsyncGraphicsFence();
        }
        
        Graphics.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);

        if (scheduleAsyncHair)
        {
            ScheduleAsyncHairInstances();
        }
    }

    private void OnEnable()
    {
        customPassGO = new GameObject("HairSimSyncContainer");
        customPassGO.hideFlags = HideFlags.HideAndDontSave;
        var customPassVolume = customPassGO.AddComponent<CustomPassVolume>();
        {
            customPass = (HairSimSyncAfterDepthNormalCustomPass)customPassVolume.AddPassOfType<HairSimSyncAfterDepthNormalCustomPass>();
            customPassVolume.injectionPoint = CustomPassInjectionPoint.AfterOpaqueDepthAndNormal;
            customPassVolume.isGlobal = true;
        }

        if (asyncScheduledHairInstances != null)
        {
            foreach (var instance in asyncScheduledHairInstances)
            {
                if (instance)
                {
                    instance.settingsSystem.updateMode = HairInstance.SettingsSystem.UpdateMode.External;
                }
            }
        }

        if (skinAttachmentTarget == null) return;
        skinAttachmentTarget.ExecuteSkinAttachmentResolveAutomatically = false;
        AttachToPlayerLoop();
    }

    private void OnDisable()
    {
        CoreUtils.Destroy(customPassGO);
        
        DetachFromPlayerLoop();
        if (skinAttachmentTarget == null) return;
        skinAttachmentTarget.ExecuteSkinAttachmentResolveAutomatically = true;
        
    }

    
    void ScheduleAsyncHairInstances()
    {
        CommandBuffer cmd = CommandBufferPool.Get("AsyncHair");
        cmd.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

        bool hairInstancesScheduled = false;
        cmd.WaitOnAsyncGraphicsFence(beforeAsyncHairSimFence);
        foreach (var instance in asyncScheduledHairInstances)
        {
            if (instance != null && instance.enabled && instance.gameObject.activeInHierarchy)
            {
                instance.settingsSystem.updateMode = HairInstance.SettingsSystem.UpdateMode.External;
                instance.DispatchUpdate(cmd, CommandBufferExecutionFlags.AsyncCompute, Time.deltaTime);
                hairInstancesScheduled = true;
            }
        }

        if (hairInstancesScheduled)
        {
            customPass.afterHairSimulationFenceSubmitted = true;
            customPass.afterHairSimulationFence = cmd.CreateAsyncGraphicsFence();
            Graphics.ExecuteCommandBufferAsync(cmd, ComputeQueueType.Default);
        }

        CommandBufferPool.Release(cmd);
    }
    
    //custom pass for syncing
    public class HairSimSyncAfterDepthNormalCustomPass : CustomPass
    {
        public bool afterHairSimulationFenceSubmitted = false;
        public GraphicsFence afterHairSimulationFence;
        protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
        {
            base.name = "HairSimSyncAfterDepthNormalCustomPass";
        }

        protected override void Execute(CustomPassContext context)
        {
            if (afterHairSimulationFenceSubmitted)
            {
                context.cmd.WaitOnAsyncGraphicsFence(afterHairSimulationFence);
            }

            afterHairSimulationFenceSubmitted = false;
        }
    }
}