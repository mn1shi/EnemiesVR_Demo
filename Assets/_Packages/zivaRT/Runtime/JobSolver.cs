#define DETACHED_RESTCOPY
#define MOTION_VECTORS

using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.ZivaRTPlayer
{
    internal enum SchedulingMode
    {
        MainThread,
        SingleWorkerThread,
        MultipleWorkerThreads
    }

    internal class JobSolver : Solver
    {
        const int k_AccumulateDisplacementsJobBatchSize = 32;
        const int k_SkinningJobBatchSize = 16;
        const int k_RemapJobBatchSize = 64;
        const int k_FaceTangentsJobBatchSize = 16;
        const int k_VertexTangentsJobBatchSize = 16;

#if MOTION_VECTORS
        const int k_CalculateMotionVectorsJobBatchSize = 64;
#endif

        // Controls how the jobs will be scheduled.
        // MainThread means the job will run immediately on the main thread.
        // SingleWorkerThread means the job will be scheduled to run on a single worker thread.
        // MultipleWorkerThread means the job will be scheduled to run in parallel on multiple workers.
        ZivaRTRig m_Rig;

        SchedulingMode m_SchedulingMode = SchedulingMode.MultipleWorkerThreads;

        // When working on a single thread, there's a faster algorithm we should use for accumulating
        // the patch displacements into the final vertex buffer. 
        // It's only implemented for CorrectiveType.TensorSkin.
        // Enabling this when using multithreading or modes other than TensorSkin 
        // will produce incorrect results and race conditions.
        bool m_AccumulateDisplacementsInPatchCorrectivesJob =>
            m_SchedulingMode != SchedulingMode.MultipleWorkerThreads &&
            m_Rig.m_CorrectiveType == CorrectiveType.TensorSkin;

        int[] m_VertexIndexMap;

        JobHandle m_RecomputeTangentsHandle = new JobHandle();
        JobHandle m_MotionVectorsReadyHandle = new JobHandle();
        JobHandle m_RemapHandle = new JobHandle();
        RelativeTransformsCalculator m_RelativeTransformsCalculator;

        struct InputBuffers
        {
            public NativeArray<float3> RestShape;
            public NativeArray<float> CurrentPose;

            public void Initialize(ZivaRTRig zivaAsset)
            {
                ReleaseBuffers();

                int numVertices = zivaAsset.m_Character.NumVertices;
                this.RestShape = new NativeArray<float3>(numVertices, Allocator.Persistent);
                this.RestShape.Reinterpret<float>(sizeof(float) * 3)
                    .CopyFrom(zivaAsset.m_Character.RestShape);

                int numPoseDimensions = zivaAsset.m_Character.PoseVectorSize;
                this.CurrentPose = new NativeArray<float>(numPoseDimensions, Allocator.Persistent);
            }

            public void SetPose(NativeArray<float> pose) { CurrentPose.CopyFrom(pose); }

            public void ReleaseBuffers()
            {
                if (RestShape.IsCreated)
                    RestShape.Dispose();
                if (CurrentPose.IsCreated)
                    CurrentPose.Dispose();
            }
        }
        InputBuffers m_InputBuffers;

        VertexBufferInterface m_VertexBufferInterface = new VertexBufferInterface();

        CopyVerticesJob m_CopyRestShapeJob;
        JobHandle m_CopyRestShapeJobHandle;

        ComputePatchCorrectivesJob m_JobPatches;
        AccumulateDisplacementsJob m_JobAccumDisplacements;
        LinearBlendSkinningJob m_SkinningJob;

        RemapVerticesJob m_RemapJob;

        const int k_NumBoundsJobs = 4;

#if MOTION_VECTORS
        bool m_CalculateMotionVectors;
        ComputeMovecsJob m_ComputeMovecsJob;
        FirstTimeComputeMovecsJob m_FirstTimeComputeMovecsJob;
        NativeArray<float3> m_PreviousPositions;
#endif

        ComputeFaceTangentFramesJob m_FaceTangentsJob;
        ComputeVertexTangentFramesJob m_VertexTangentsJob;

        // Markers for profiling pieces of the jobs.
        // Currently must be created statically in managed C#, then passed to the Job object. :(
        static readonly ComputePatchCorrectivesJob.Profile k_PatchProfiling =
            ComputePatchCorrectivesJob.Profile.Init();

        public JobSolver(SchedulingMode schedulingMode) { m_SchedulingMode = schedulingMode; }

        public override NativeArray<float3x4> RelativeTransforms
        {
            get
            {
                return m_RelativeTransformsCalculator.RelativeTransforms;
            }
        }

        // All containers in a Job must be initialized or we get a validation error - "All containers must be valid when scheduling a job."
        // Workaround is to allocate native arrays with size 0
        static public NativeArray<T> AllocateConditionally<T>(bool condition, int size)
            where T : struct
        {
            return new NativeArray<T>(
                condition ? size : 0, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
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
            if (!base.Init(rig, targetMesh, vertexIndexMap, tangentFramesInfo, recomputeTangentFrames,
                calculateMotionVectors, shaderData))
                return false;

            Assert.AreEqual(vertexIndexMap.Length, targetMesh.vertexCount);

            m_Rig = rig;
            m_VertexIndexMap = vertexIndexMap;

            m_InputBuffers.Initialize(m_Rig);

            m_CopyRestShapeJob.Initialize(m_Rig.m_Character.NumVertices);
            m_CopyRestShapeJob.Input = m_InputBuffers.RestShape;
#if DETACHED_RESTCOPY
            ScheduleRestShapeCopyJob(); // Solve() expects this to already have been scheduled.
#endif

            m_JobPatches.Initialize(m_Rig, m_AccumulateDisplacementsInPatchCorrectivesJob);
            m_JobAccumDisplacements.Initialize(
                rig, m_JobPatches.DisplacementsStarts, m_AccumulateDisplacementsInPatchCorrectivesJob);
            m_SkinningJob.Initialize(m_Rig.m_Skinning.SkinningWeights);

            m_RemapJob.Initialize(m_VertexIndexMap);

#if MOTION_VECTORS
            this.m_CalculateMotionVectors = calculateMotionVectors;
            if (calculateMotionVectors)
            {
                m_PreviousPositions = new NativeArray<float3>(targetMesh.vertexCount, Allocator.Persistent);
            }
            m_IsFirstTime = true;
#endif

            m_FaceTangentsJob.Initialize(targetMesh.triangles, tangentFramesInfo);
            m_VertexTangentsJob.Initialize(tangentFramesInfo);
            // this call can modify the mesh
            m_VertexBufferInterface.Initialize(targetMesh, recomputeTangentFrames, calculateMotionVectors);

            // Assign profiling markers to jobs.
            m_JobPatches.Profiling = k_PatchProfiling;

            m_RelativeTransformsCalculator = new RelativeTransformsCalculator(m_Rig);

            return true;
        }

        internal struct JobFuture<T>
            where T : struct
        {
            public JobHandle dependencies; // Jobs that must finish before the data can be accessed
            public NativeArray<T> data;

            public static implicit operator NativeArray<T>(JobFuture<T> future)
            {
                future.dependencies.Complete();
                return future.data;
            }
        }

        public override void StartAsyncSolve(
            JobFuture<float> currentPose,
            JobFuture<float> jointWorldTransforms,
            RecomputeTangentFrames recomputeTangentFrames,
            bool doCorrectives,
            bool doSkinning)
        {
            // Uncomment this if you need to debug the source of native memory leaks:
            // NativeLeakDetection.Mode = NativeLeakDetectionMode.EnabledWithStackTrace;

            Profiler.BeginSample("ScheduleCorrectives");
            JobFuture<float3> preSkinningShape = ScheduleCorrectivesJobs(currentPose, doCorrectives);
            Profiler.EndSample();

            Profiler.BeginSample("ScheduleSkinning");
            JobFuture<float3> skinnedShape = ScheduleSkinningJobs(jointWorldTransforms, preSkinningShape, doSkinning);
            Profiler.EndSample();

            Profiler.BeginSample("ScheduleVertexRemap");
            m_RemapJob.SrcVertices = skinnedShape.data;
            m_RemapJob.DstVertices = m_VertexBufferInterface.PositionsStreamData;
            int numDstVertices = m_RemapJob.NumDstVertices;
            m_RemapHandle = CustomizableScheduleJobFor(
                m_SchedulingMode, m_RemapJob, numDstVertices, k_RemapJobBatchSize, skinnedShape.dependencies);
            Profiler.EndSample();

            m_MotionVectorsReadyHandle = new JobHandle();
#if MOTION_VECTORS
            if (m_CalculateMotionVectors)
            {
                ComputeMovecsJobContext context;
                context.DeltaTime = Application.isPlaying ? Time.deltaTime : 1f / 30f;
                context.CurrentPositions = m_VertexBufferInterface.PositionsStreamData;
                context.MotionVectors = m_VertexBufferInterface.MovecsStreamData;
                context.PreviousPositions = m_PreviousPositions;
                // First time will just copy current positions to previous positions and set motion vectors to 0
                if (m_IsFirstTime)
                {
                    m_FirstTimeComputeMovecsJob.Context = context;                   
                    m_MotionVectorsReadyHandle = CustomizableScheduleJobFor(
                        m_SchedulingMode, m_FirstTimeComputeMovecsJob, numDstVertices, k_CalculateMotionVectorsJobBatchSize,
                        m_RemapHandle);
                    m_IsFirstTime = false;
                }
                else
                {
                    m_ComputeMovecsJob.Context = context;
                    m_MotionVectorsReadyHandle = CustomizableScheduleJobFor(
                        m_SchedulingMode, m_ComputeMovecsJob, numDstVertices, k_CalculateMotionVectorsJobBatchSize,
                        m_RemapHandle);                  
                }
            }
#endif

            m_RecomputeTangentsHandle = new JobHandle();
            if (recomputeTangentFrames != RecomputeTangentFrames.None)
            {
                Profiler.BeginSample("ScheduleTangentFrameRecalc");

                bool recalcTangents = (recomputeTangentFrames == RecomputeTangentFrames.NormalsAndTangents);

                // Kick off face tangents job
                m_FaceTangentsJob.Vertices = m_RemapJob.DstVertices;
                m_FaceTangentsJob.CalculateTangents = recalcTangents;

                int numTriangles = m_FaceTangentsJob.NumTriangles;
                var faceNormalsHandle = CustomizableScheduleJobFor(
                    m_SchedulingMode, m_FaceTangentsJob, numTriangles, k_FaceTangentsJobBatchSize, m_RemapHandle);

                // Kick off vertex tangents job
                m_VertexTangentsJob.FaceNormals = m_FaceTangentsJob.FaceNormals;
                m_VertexTangentsJob.CalculateTangents = recalcTangents;
                m_VertexTangentsJob.FaceTangents = m_FaceTangentsJob.FaceTangents;

                // Store the results into our local vertex buffers.
                m_VertexTangentsJob.VertexNormals = m_VertexBufferInterface.NormalsStreamData;
                if (recalcTangents)
                {
                    m_VertexTangentsJob.VertexTangents = m_VertexBufferInterface.TangentsStreamData;
                }

                int numVertices = m_VertexTangentsJob.NumVertices;
                m_RecomputeTangentsHandle = CustomizableScheduleJobFor(
                    m_SchedulingMode, m_VertexTangentsJob, numVertices, k_VertexTangentsJobBatchSize,
                    faceNormalsHandle);

                Profiler.EndSample();
            }

            // When all the jobs that consume the skinned positions are done, it's safe to re-copy
            // the rest shape into the accumulation buffer.
            // We don't need these results until next frame, so schedule this job last.
#if DETACHED_RESTCOPY
            var restShapeCopyDeps = m_RemapHandle;

            // Reset rest data which we'll need next frame
            ScheduleRestShapeCopyJob(restShapeCopyDeps);
#endif
        }

        unsafe void ToVector3Array(Vector3[] arr, VertexBufferInterface.DataStream ds)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = *ds.GetPtrAtIndex<float3>(i);
            }
        }

        unsafe void ToVector4Array(Vector4[] arr, VertexBufferInterface.DataStream ds)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = *ds.GetPtrAtIndex<float4>(i);
            }
        }

        public override void GetPositions(Vector3[] positions)
        {
            ToVector3Array(positions, m_VertexBufferInterface.PositionsStreamData);
        }

        public override void GetNormals(Vector3[] normals)
        {
            if (m_VertexBufferInterface.NormalsStreamData.Data.IsCreated)
            {
                ToVector3Array(normals, m_VertexBufferInterface.NormalsStreamData);
            }
            else
            {
                Debug.LogError("Normals not available because 'Recalculate Tangent Frames' option was not set in the ZivaRT component.");
            }
        }

        public override void GetTangents(Vector4[] tangents)
        {
            if (m_VertexBufferInterface.TangentsStreamData.Data.IsCreated)
            {
                ToVector4Array(tangents, m_VertexBufferInterface.TangentsStreamData);
            }
            else
            {
                Debug.LogError("Tangents not available because 'Recalculate Tangent Frames' option was not set in the ZivaRT component.");
            }
        }

        public override void GetMotionVectors(Vector3[] motionVectors)
        {
            if (m_VertexBufferInterface.MovecsStreamData.Data.IsCreated)
            {
                ToVector3Array(motionVectors, m_VertexBufferInterface.MovecsStreamData);               
            }
            else
            {
                Debug.LogError("Motion vectors not available because 'Calculate Motion Vectors' option was not set in the ZivaRT component.");
            }
        }

        public override void WaitOnAsyncSolve()
        {
            // Get results back from the jobs and forward them to Unity as needed.

            // First synchronize on the completion of the vertex positions data.
            // 'remapHandle' moves vertex positions into the correct order.
            m_RemapHandle.Complete();
            m_VertexBufferInterface.CommitChanges(VertexBufferInterface.AttributeStreams.Position);

            // Finish computing all remaining vertex attributes, as needed.
            m_RecomputeTangentsHandle.Complete();
            m_MotionVectorsReadyHandle.Complete();
            m_VertexBufferInterface.CommitChanges(VertexBufferInterface.AttributeStreams.NormalTangentMovecs);
        }

        JobHandle ScheduleRestShapeCopyJob(JobHandle dependencies = default)
        {
            // Kick off job to copy the rest shape so it can then be deformed.
            m_CopyRestShapeJobHandle = CustomizableScheduleJob(m_SchedulingMode, m_CopyRestShapeJob, dependencies);
            return m_CopyRestShapeJobHandle;
        }

        JobFuture<float3> ScheduleCorrectivesJobs(JobFuture<float> currentPose, bool applyCorrectives)
        {
            // When the returned JobHandle completes, preSkinningShape will contain the mesh
            // vertices with correctives applied according to the solver's current settings,
            // ready to be passed into the LinearBlendSkinning job.

            // `applyCorrectives` determines whether or not we actually deform the mesh's rest
            // shape with correctives. It may seem strange to call
            // ScheduleCorrectivesJobs(applyCorrectives=false, ...), but if we want to debug
            // the skinning step by turning off correctives, we still need to make sure that the
            // necessary jobs run to prepare the mesh for the LinearBlendSkinning job.
#if DETACHED_RESTCOPY
            var copyHandle = m_CopyRestShapeJobHandle;
#else
            var copyHandle = ScheduleRestShapeCopyJob();
#endif

            if (applyCorrectives)
            {
                // First kick off the job to compute correctives for all patches
                Profiler.BeginSample("CopyPoseVector");
                m_InputBuffers.SetPose(currentPose);
                Profiler.EndSample();
                m_JobPatches.CurrentPose = m_InputBuffers.CurrentPose;
                m_JobPatches.Shape = m_CopyRestShapeJob.Output;
                int numPatches = m_JobPatches.NumPatches;
                JobHandle patchesHandle = CustomizableScheduleJobFor(
                    m_SchedulingMode, m_JobPatches, numPatches, 1, copyHandle);

                // Then kick off the job to accumulate all patches into the pre-skinning shape
                m_JobAccumDisplacements.Displacements = m_JobPatches.Displacements;
                m_JobAccumDisplacements.Shape = m_CopyRestShapeJob.Output;
                Profiler.BeginSample("JobHandle.CombineDependencies()-AccumDisplacements");
                JobHandle accumDeps = JobHandle.CombineDependencies(copyHandle, patchesHandle);
                Profiler.EndSample();
                int numVertices = m_JobAccumDisplacements.NumShapeVertices;
                JobHandle accumHandle = accumDeps;

                // Only run the AccumulateDisplacementsJob if we are running in parallel with multiple
                // worker threads. The single threaded version runs significantly faster with a different
                // memory access behavior compared to what the AccumulateDisplacementsJob does, but the
                // single threaded version cannot be trivially used in parallel due to synchronization.
                // When running on one thread, the accumulation of displacements is done in the
                // ComputePatchCorrectivesJob.
                if (!m_AccumulateDisplacementsInPatchCorrectivesJob)
                {
                    accumHandle = CustomizableScheduleJobFor<AccumulateDisplacementsJob>(
                        m_SchedulingMode, m_JobAccumDisplacements, numVertices,
                        k_AccumulateDisplacementsJobBatchSize, accumDeps);
                }

                return new JobFuture<float3>
                {
                    dependencies = accumHandle,
                    data = m_JobAccumDisplacements.Shape,
                };
            }
            else
            {
                // The copy-rest-shape job is the only one that needs to run, and it's output should
                // flow directly into the skinning job.
                return new JobFuture<float3>
                {
                    dependencies = copyHandle,
                    data = m_CopyRestShapeJob.Output,
                };
            }
        }

        JobFuture<float3> ScheduleSkinningJobs(JobFuture<float> jointWorldTransforms,
                                                       JobFuture<float3> preSkinningShape,
                                                       bool doSkinning)
        {
            if (doSkinning)
            {
                Profiler.BeginSample("WorldToRelativeBoneTransforms");
                m_SkinningJob.BoneTransforms = m_RelativeTransformsCalculator.WorldToRelative(jointWorldTransforms);
                Profiler.EndSample();
                m_SkinningJob.Vertices = preSkinningShape.data;
                int numVertices = m_SkinningJob.NumVertices;
                JobHandle skinningHandle = CustomizableScheduleJobFor(
                    m_SchedulingMode, m_SkinningJob, numVertices, k_SkinningJobBatchSize,
                    preSkinningShape.dependencies);

                return new JobFuture<float3>
                {
                    dependencies = skinningHandle,
                    data = m_SkinningJob.Vertices,
                };
            }
            else
            {
                // If not skinning, just forward the shape buffer onwards.

                // Flip x-axis (skinning does this for us, it gets baked into the joint transforms).
                // Not optimized, as turning off skinning is debug only.
                preSkinningShape.dependencies.Complete();
                var shape = preSkinningShape.data;
                for (int i = 0; i < shape.Length; ++i)
                {
                    float3 vtx = shape[i];
                    vtx.x *= -1;
                    shape[i] = vtx;
                }

                return preSkinningShape;
            }
        }

        // Schedules an IJobFor job to run, but allows controlling how exactly it will be scheduled.
        // For SingleWorkerThread and MultipleWorkerThreads it returns correct JobHandles so that they
        // can be used as dependencies.
        // For MainThread it returns a stub JobHandle value that has no meaning, but also is not used.
        static JobHandle CustomizableScheduleJobFor<Job>(
            SchedulingMode mode,
            Job job,
            int arrayLength,
            int innerLoopBatchCount,
            JobHandle dependency)
            where Job : struct, IJobFor
        {
            switch (mode)
            {
                case SchedulingMode.MainThread:
                {
                    dependency.Complete();
                    job.Run(arrayLength);
                    return default;
                }
                case SchedulingMode.SingleWorkerThread:
                {
                    return job.Schedule(arrayLength, dependency);
                }
                case SchedulingMode.MultipleWorkerThreads:
                {
                    return job.ScheduleParallel(arrayLength, innerLoopBatchCount, dependency);
                }
            }
            return default;
        }

        // Same as above, but for jobs that implement the IJob interface.
        static JobHandle CustomizableScheduleJob<Job>(
            SchedulingMode mode,
            Job job,
            JobHandle dependency)
            where Job : struct, IJob
        {
            switch (mode)
            {
                case SchedulingMode.MainThread:
                {
                    dependency.Complete();
                    job.Run();
                    return default;
                }
                case SchedulingMode.SingleWorkerThread:
                case SchedulingMode.MultipleWorkerThreads:
                {
                    return job.Schedule(dependency);
                }
            }
            return default;
        }

        public override void Dispose()
        {
            m_RemapHandle.Complete();
            m_RecomputeTangentsHandle.Complete();
            m_MotionVectorsReadyHandle.Complete();

#if DETACHED_RESTCOPY
            m_CopyRestShapeJobHandle.Complete();
#endif

            m_InputBuffers.ReleaseBuffers();

            m_VertexBufferInterface.Dispose();

            m_CopyRestShapeJob.ReleaseBuffers();
            m_JobPatches.ReleaseBuffers();
            m_JobAccumDisplacements.ReleaseBuffers();
            m_SkinningJob.ReleaseBuffers();

            m_RemapJob.ReleaseBuffers();

#if MOTION_VECTORS
            if (m_PreviousPositions.IsCreated)
            {
                m_PreviousPositions.Dispose();
            }
#endif

            m_FaceTangentsJob.ReleaseBuffers();
            m_VertexTangentsJob.ReleaseBuffers();
            if (m_RelativeTransformsCalculator != null)
            { 
                m_RelativeTransformsCalculator.Dispose();
            }
        }
    }
}
