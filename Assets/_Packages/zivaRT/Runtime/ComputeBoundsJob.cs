using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{
    // Job to compute min and max of skinned mesh 
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct ComputeBoundsJob : IJob
    {
        [ReadOnly]
        public NativeArray<float3> Extents;   // extents and centers of rest pose bounding boxes
        public NativeArray<float3> Centers;
        public NativeArray<int> BoneIndices;
        public NativeArray<float3x4> RelativeTransforms;  // bone transforms
        int m_BoneCount;

        // Output
        // m_MinMax[0] is the minimum corner of the bounding box
        // m_MinMax[1] is the maximum corner of the bounding box
        [WriteOnly]
        [NativeFixedLength(2)]
        NativeArray<float3> m_MinMax;
        public NativeArray<float3> MinMax
        {
            get
            {
                return m_MinMax;
            }
        }
       
        public void Initialize(int boneCount)
        {
            m_BoneCount = boneCount;
            Extents = new NativeArray<float3>(m_BoneCount, Allocator.Persistent);
            Centers = new NativeArray<float3>(m_BoneCount, Allocator.Persistent);
            BoneIndices = new NativeArray<int>(m_BoneCount, Allocator.Persistent);
            m_MinMax = new NativeArray<float3>(2, Allocator.Persistent);
        }

        public void ReleaseBuffers()
        {
            if (Extents.IsCreated)
            { 
                Extents.Dispose();
            }
            if (Centers.IsCreated)
            {
                Centers.Dispose();
            }
            if (BoneIndices.IsCreated)
            {
                BoneIndices.Dispose();
            }
            if (m_MinMax.IsCreated)
            {
                m_MinMax.Dispose();
            }
        }

        public void Execute() 
        {
            float3 min = new float3(float.MaxValue, float.MaxValue, float.MaxValue);
            float3 max = new float3(float.MinValue, float.MinValue, float.MinValue);
            for (int boneIndex = 0; boneIndex < m_BoneCount; boneIndex++)
            {
                int transformIndex = BoneIndices[boneIndex];
                
                // transform the center by the bone transform
                float3 center = RelativeTransforms[transformIndex].c3;
                center += Centers[boneIndex].x * RelativeTransforms[transformIndex].c0;
                center += Centers[boneIndex].y * RelativeTransforms[transformIndex].c1;
                center += Centers[boneIndex].z * RelativeTransforms[transformIndex].c2;

                // rotate the extents by the bone transform, this mimics what core c++ code is doing for SkinnedMeshRenderer
                float3 extents = math.abs(Extents[boneIndex].x * RelativeTransforms[transformIndex].c0);
                extents += math.abs(Extents[boneIndex].y * RelativeTransforms[transformIndex].c1);
                extents += math.abs(Extents[boneIndex].z * RelativeTransforms[transformIndex].c2);
                min = math.min(min, center - extents);
                max = math.max(max, center + extents);
                
            }
            m_MinMax[0] = min;
            m_MinMax[1] = max;
        }
    }
}
