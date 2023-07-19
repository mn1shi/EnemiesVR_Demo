using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.ZivaRTPlayer
{
    // Skin mesh vertices according to skeletal bone transforms.
    // NOTE: Does not deform mesh normals.
    // Parallel-for over vertices.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct LinearBlendSkinningJob : IJobFor
    {
        // Input: Transforms of all bones, relative to the rest pose bone transforms.
        [ReadOnly]
        public NativeArray<float3x4> BoneTransforms;

        // Input-Output: Mesh vertex positions
        // Should be fed the rest shape of the mesh, this is then deformed by the bone transforms.
        public NativeArray<float3> Vertices;

        // Weights of the bones that influence each vertex.
        // Stored as a sparse matrix where boneInfluences and weights are concatenated arrays of
        // corresponding bone-weight pairs, and boneInfluencesStarts tell us where the sub-arrays
        // for each vertex begin and end.
        [ReadOnly]
        NativeArray<float> m_Weights;
        [ReadOnly]
        NativeArray<int> m_BoneInfluences;
        [ReadOnly]
        NativeArray<int> m_BoneInfluencesStarts;

        // Number of vertices in the mesh being skinned.
        // This is the number of iterations in the parallel-for loop.
        public int NumVertices { get { return m_BoneInfluencesStarts.Length - 1; } }

        public void Execute(int index)
        {
            int start = m_BoneInfluencesStarts[index];
            int end = m_BoneInfluencesStarts[index + 1];

            float3x4 blendedTransform = float3x4.zero;
            for (int i = start; i < end; ++i)
            {
                float weight = m_Weights[i];
                int boneIdx = m_BoneInfluences[i];
                float3x4 boneTransform = BoneTransforms[boneIdx];
                blendedTransform += weight * boneTransform;
            }

            // vertex = blendedTransform * vertex
            float3 position = Vertices[index];
            float3 result = blendedTransform.c3; // Start with translation component
            result += position.x * blendedTransform.c0;
            result += position.y * blendedTransform.c1;
            result += position.z * blendedTransform.c2;
            Vertices[index] = result;
        }

        // SparseMatrix of skinning weights: columns are vertices, rows are bones.
        public void Initialize(Unity.ZivaRTPlayer.SparseMatrix skinningWeights)
        {
            ReleaseBuffers();

            this.m_Weights = new NativeArray<float>(skinningWeights.W, Allocator.Persistent);
            this.m_BoneInfluences = new NativeArray<int>(skinningWeights.RowIndices, Allocator.Persistent);
            this.m_BoneInfluencesStarts = new NativeArray<int>(skinningWeights.ColStarts, Allocator.Persistent);
        }

        public void ReleaseBuffers()
        {
            if (m_Weights.IsCreated)
                m_Weights.Dispose();
            if (m_BoneInfluences.IsCreated)
                m_BoneInfluences.Dispose();
            if (m_BoneInfluencesStarts.IsCreated)
                m_BoneInfluencesStarts.Dispose();
        }
    }
}
