using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{

    internal struct ComputeMovecsJobContext
    {
        public float DeltaTime;

        [NativeDisableContainerSafetyRestriction]
        [NoAlias]
        [ReadOnly]
        public VertexBufferInterface.DataStream CurrentPositions;

        public NativeArray<float3> PreviousPositions;

        [NativeDisableContainerSafetyRestriction]
        [NoAlias]
        [WriteOnly]
        public VertexBufferInterface.DataStream MotionVectors;
    }

    // Stash previous positions and compute motion vectors
    // Parallelized over the vertices of the vertex array.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct ComputeMovecsJob : IJobFor
    {
        public ComputeMovecsJobContext Context;

        public unsafe void Execute(int index)
        {
            float3 prevPos = Context.PreviousPositions[index];
            float3 currPos = *Context.CurrentPositions.GetPtrAtIndex<float3>(index);      
            float3* pMovec = Context.MotionVectors.GetPtrAtIndex<float3>(index);
            *pMovec = currPos - prevPos;
            Context.PreviousPositions[index] = currPos;
        }
    }

    // When running for the 1st time we don't know previous positions
    // Initialize previous positions and set compute motion vectors to 0
    // Parallelized over the vertices of the vertex array.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct FirstTimeComputeMovecsJob : IJobFor
    {
        public ComputeMovecsJobContext Context;

        public unsafe void Execute(int index)
        {
            float3 currPos = *Context.CurrentPositions.GetPtrAtIndex<float3>(index);
            float3* pMovec = Context.MotionVectors.GetPtrAtIndex<float3>(index);
            *pMovec = float3.zero;
            Context.PreviousPositions[index] = currPos;
        }
    }
}
