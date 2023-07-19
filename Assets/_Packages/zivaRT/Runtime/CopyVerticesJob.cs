using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{
    // Job to copy a vertex buffer.
    // NOTE: Not parallelized. Use this when the overhead of ParallelFor isn't worth it.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct CopyVerticesJob : IJob
    {
        [ReadOnly]
        public NativeArray<float3> Input;

        [WriteOnly]
        NativeArray<float3> m_Output;
        public NativeArray<float3> Output { get { return m_Output; } }

        public void Execute() { Input.CopyTo(m_Output); }

        // Initialize the job by telling it the size fo the vertex buffer it will be copying.
        // This is needed in order to pre-allocate and manage the memory of the output buffer.
        public void Initialize(int numVertices)
        {
            ReleaseBuffers();
            this.m_Output = new NativeArray<float3>(numVertices, Allocator.Persistent);
        }

        public void ReleaseBuffers()
        {
            if (m_Output.IsCreated)
                m_Output.Dispose();
        }
    }

}
