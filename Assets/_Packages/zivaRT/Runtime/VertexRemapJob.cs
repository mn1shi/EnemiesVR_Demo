using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{
    // Copy vertex positions from srcVertices into dstVertices according to an index map from
    // destination indices to source indices.
    // Parallelized over the vertices of the destination array.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct RemapVerticesJob : IJobFor
    {
        // Input
        [ReadOnly]
        public NativeArray<float3> SrcVertices;

        // Output
        [WriteOnly]
        public VertexBufferInterface.DataStream DstVertices;

        // Static data
        // For each element of dstVertices, stores the index in srcVertices that will be copied.
        [ReadOnly]
        NativeArray<int> m_IndexMap;

        public unsafe void Execute(int index)
        {
            float3* pDst = DstVertices.GetPtrAtIndex<float3>(index);
            *pDst = SrcVertices[m_IndexMap[index]];
        }

        // The number of vertices in the destination array.
        // This is the number of parallel iterations the job should execute.
        public int NumDstVertices { get { return m_IndexMap.Length; } }

        // Set up the job with the map from indices of dstVertices to indices of srcVertices.
        // NOTE: Every source index in the index map must be less than the size of the srcVertices
        // array fed into this job.
        public void Initialize(int[] dstVertexToSrcVertexMap)
        {
            ReleaseBuffers();
            this.m_IndexMap = new NativeArray<int>(dstVertexToSrcVertexMap, Allocator.Persistent);
        }

        public void ReleaseBuffers()
        {
            if (m_IndexMap.IsCreated)
                m_IndexMap.Dispose();
        }
    }
}
