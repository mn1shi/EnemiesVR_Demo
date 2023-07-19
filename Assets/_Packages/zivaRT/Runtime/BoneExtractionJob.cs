using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Jobs;

namespace Unity.ZivaRTPlayer
{
    [BurstCompile]
    struct CalculateJointMatricesJob : IJobParallelForTransform
    {
        public NativeArray<float4x4> JointLocalMatrices;

        public void Execute(int index, TransformAccess transform)
        {
            JointLocalMatrices[index] =
                float4x4.TRS(transform.localPosition, transform.localRotation, transform.localScale);
        }
    }
    [BurstCompile]
    struct BoneExtractionJob : IJob
    {
        [WriteOnly]
        public NativeArray<float> Pose;

        [WriteOnly]
        public NativeArray<float> Bones;

        [ReadOnly]
        public float MeshRescale;

        [ReadOnly]
        public NativeArray<int> BoneIndexMap;

        [ReadOnly]
        public int NumExtraParameters;

        [ReadOnly]
        public int NumJoints;

        [ReadOnly]
        public float4x4 ParentWorldToLocalMatrix;

        [ReadOnly]
        public float4x4 TransformRoot;

        [ReadOnly]
        public NativeArray<int> ParentIndices;

        [DeallocateOnJobCompletion]
        public NativeArray<float4x4> JointLocalMatrices;
        [DeallocateOnJobCompletion]
        public NativeArray<float4x4> JointLocalToWorldMatrices;

        public void Execute()
        {
            for (int i = 0; i < JointLocalMatrices.Length; i++)
            {
                if (ParentIndices[i] > -1)
                    JointLocalToWorldMatrices[i] =
                        math.mul(JointLocalToWorldMatrices[ParentIndices[i]], JointLocalMatrices[i]);
                else
                    JointLocalToWorldMatrices[i] = TransformRoot;
            }

            unsafe
            {
                float* posePtr = &((float*)Pose.GetUnsafePtr())[NumExtraParameters];
                float* bonePtr = (float*)Bones.GetUnsafePtr();

                for (int b = 0; b < NumJoints; b++)
                {
                    var unityBoneIndex = BoneIndexMap[b];
                    if (unityBoneIndex == -1)
                    {
                        // Skip over the current Ziva joint, because we don't have a corresponding Unity bone.
                        bonePtr += sizeof(float3);
                        posePtr += sizeof(float3);
                        continue;
                    }

                    var localMatrix = float4x4.identity;

                    if (b > 0)
                    {
                        localMatrix = JointLocalMatrices[unityBoneIndex];
                    }

                    // Construct local transform of the bone and copy it into the pose vector.

                    // The bones are in Unity space (left-handed coordinates).
                    // Ziva needs to receive transforms in right-handed coordinates (x-coordinate flipped).
                    // var unityToZiva = Matrix4x4.Scale(new Vector3(-1, 1, 1));
                    // Conveniently, this matrix is its own inverse.
                    // var zivaToUnity = unityToZiva;

                    // We need the equivalent localMatrix transform in Ziva space:
                    // localMatrix = unityToZiva * localMatrix * zivaToUnity
                    // It's cheaper to just negate the appropriate entries of localMatrix as we copy.

                    localMatrix.c0.yz = -localMatrix.c0.yz;
                    localMatrix.c1.x = -localMatrix.c1.x;
                    localMatrix.c2.x = -localMatrix.c2.x;
                    localMatrix.c3.x = -localMatrix.c3.x;

                    UnsafeUtility.MemCpyStride(posePtr, sizeof(float3), &localMatrix, sizeof(float4), sizeof(float3),
                        4);
                    posePtr += sizeof(float3);

                    // Construct the local-to-world bone transforms that will be used to skin the mesh
                    // after the corrective shapes are applied.

                    // We don't actually want the full world-space transforms of the bones, we want the
                    // transforms relative to the parent space of the root joint (which might not be world-space).
                    var boneMatrix = math.mul(ParentWorldToLocalMatrix, JointLocalToWorldMatrices[unityBoneIndex]);
                    boneMatrix = math.mul(boneMatrix, float4x4.Scale(new float3(MeshRescale, MeshRescale, MeshRescale)));
                    boneMatrix.c0 = -boneMatrix.c0;

                    // The bone matrices will be applied to the mesh after the correctives.
                    // The correctives are done in Ziva space, so convert back to Unity space as part of the
                    // skinning transform.

                    // Copy matrix in column-major order, flipping the x-axis column.
                    // This is equivalent to boneMatrix * zivaToUnity.

                    UnsafeUtility.MemCpyStride(bonePtr, sizeof(float3), &boneMatrix, sizeof(float4),
                        sizeof(float3), 4);
                    bonePtr += sizeof(float3);
                }
            }
        }
    }
}
