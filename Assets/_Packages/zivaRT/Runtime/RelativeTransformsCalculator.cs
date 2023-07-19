using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{
    class RelativeTransformsCalculator : IDisposable
    {
        [BurstCompile]
        struct UpdateJointTransformsJob : IJobParallelFor
        {
            public NativeArray<float3x4> RelativeTransforms;
            [ReadOnly] public NativeArray<float3x4> RestPoseInverse;

            public void Execute(int index)
            {
                RelativeTransforms[index] = math.mul(
                    RelativeTransforms[index], new float4x4(
                        new float4(RestPoseInverse[index].c0, 0),
                        new float4(RestPoseInverse[index].c1, 0),
                        new float4(RestPoseInverse[index].c2, 0),
                        new float4(RestPoseInverse[index].c3, 1)
                    ));
            }
        }

        public RelativeTransformsCalculator(ZivaRTRig rig)
        {
            // Allocate scratch space for computing joint transforms for skinning.
            m_RelativeTransforms =
                new NativeArray<float3x4>(rig.m_Character.NumJoints, Allocator.Persistent);

            // Convert rest pose bone transform inverses into float3x4 for later convenience.
            m_RestPoseInverse =
                new NativeArray<float3x4>(rig.m_Character.NumJoints, Allocator.Persistent);
            m_RestPoseInverse.Reinterpret<float>(3 * 4 * sizeof(float))
                .CopyFrom(rig.m_Skinning.RestPoseInverse);
        }

        public NativeArray<float3x4> WorldToRelative(NativeArray<float> worldTransformsFlattened)
        {
            // Convert bone transforms to be relative-to-rest-pose
            // Temporarily storing world transforms in mRelativeTransforms is for convenience/performance.
            m_RelativeTransforms.Reinterpret<float>(3 * 4 * sizeof(float))
                .CopyFrom(worldTransformsFlattened);

            new UpdateJointTransformsJob
            {
                RelativeTransforms = m_RelativeTransforms,
                RestPoseInverse = m_RestPoseInverse
            }.Run(m_RelativeTransforms.Length);

            return m_RelativeTransforms;
        }

        public void Dispose()
        {
            if (m_RelativeTransforms.IsCreated)
                m_RelativeTransforms.Dispose();

            if (m_RestPoseInverse.IsCreated)
                m_RestPoseInverse.Dispose();
        }

        NativeArray<float3x4> m_RestPoseInverse;
        NativeArray<float3x4> m_RelativeTransforms;

        public NativeArray<float3x4> RelativeTransforms
        {
            get
            {
                return m_RelativeTransforms;
            }
        }
    }
}
