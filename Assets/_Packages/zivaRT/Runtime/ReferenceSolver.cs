using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.ZivaRTPlayer
{
    internal class ReferenceSolver : Solver
    {
        struct Workspace
        {
            public float[] LocalWeightedPose;
            public float[] Phi;
            public float[] SubspaceCoeff;
            public float[] Displacements;
        }

        ZivaRTRig m_Rig;

        float[] m_Shape;
        int[] m_VertexIndexMap;
        MeshTangentFramesInfo m_TangentFramesInfo;

        float[] m_RelativeTransforms;

        Vector3[] m_Verts;
        int[] m_Triangles;
        Vector3[] m_Normals;
        Vector4[] m_Tangents;

        ComputeTangentFrames m_ComputeTangents;

        Workspace[] m_Workspaces;

        public override bool Init(
            ZivaRTRig rig,
            Mesh targetMesh,
            int[] vertexIndexMap,
            MeshTangentFramesInfo tangentFramesInfo,
            RecomputeTangentFrames recomputeTangentFrames,
            bool calculateMotionVectors,
            ZivaShaderData shaderData)
        {
            if (!base.Init(rig, targetMesh, vertexIndexMap, tangentFramesInfo, recomputeTangentFrames, calculateMotionVectors, shaderData))
                return false;

            if (calculateMotionVectors)
                Debug.LogWarning($"calculateMotionVectors not implemented by reference solver");

            Assert.AreEqual(vertexIndexMap.Length, targetMesh.vertexCount);

            m_Rig = rig;
            m_VertexIndexMap = vertexIndexMap;
            m_TangentFramesInfo = tangentFramesInfo;

            m_Shape = new float[m_Rig.m_Character.RestShape.Length];
            m_RelativeTransforms = new float[rig.m_Skinning.RestPoseInverse.Length];

            // NOTE: These calls to access the vertices and triangles indices are actually very expensive.
            // So only do them once, and cache the resulting arrays.
            m_Verts = targetMesh.vertices;
            m_Triangles = targetMesh.triangles;
            m_Normals = targetMesh.normals;
            m_Tangents = targetMesh.tangents;

            m_ComputeTangents = new ComputeTangentFrames(m_Triangles.Length / 3);

            m_Workspaces = AllocateWorkspaces(rig);

            return true;
        }

        public override void StartAsyncSolve(
            JobSolver.JobFuture<float> currentPose,
            JobSolver.JobFuture<float> jointWorldTransforms,
            RecomputeTangentFrames recomputeTangentFrames,
            bool doCorrectives,
            bool doSkinning)
        {
            Array.Copy(m_Rig.m_Character.RestShape, m_Shape, m_Shape.Length);

            if (doCorrectives)
                Correctives(currentPose);

            if (doSkinning)
                Skinning(jointWorldTransforms);
            else
                ToUnitySpace(ref m_Shape);

            Profiler.BeginSample("ZivaVertRemap");
            for (int a = 0; a < m_VertexIndexMap.Length; a++)
            {
                var zivaIdx = m_VertexIndexMap[a];
                m_Verts[a].x = m_Shape[zivaIdx * 3];
                m_Verts[a].y = m_Shape[zivaIdx * 3 + 1];
                m_Verts[a].z = m_Shape[zivaIdx * 3 + 2];
            }
            Profiler.EndSample();

            m_Mesh.SetVertices(m_Verts);
            m_Mesh.RecalculateBounds();

            if (recomputeTangentFrames == RecomputeTangentFrames.NormalsAndTangents)
            {
                m_ComputeTangents.Compute(
                    m_Verts, m_Triangles, m_TangentFramesInfo, ref m_Normals, ref m_Tangents);
                m_Mesh.SetNormals(m_Normals);
                m_Mesh.SetTangents(m_Tangents);
            }
            else if (recomputeTangentFrames == RecomputeTangentFrames.NormalsOnly)
            {
                m_ComputeTangents.Compute(m_Verts, m_Triangles, m_TangentFramesInfo, ref m_Normals);
                m_Mesh.SetNormals(m_Normals);
            }
        }

        public override void WaitOnAsyncSolve()
        {  
            // In the ReferenceSolver all the work is done already.
        }

        public override void GetPositions(Vector3[] positions)
        {
            m_Verts.CopyTo(positions, 0);
        }

        public override void GetNormals(Vector3[] normals)
        {
            m_Normals.CopyTo(normals, 0);
        }

        public override void GetTangents(Vector4[] tangents)
        {
            m_Tangents.CopyTo(tangents, 0);
        }

        public override void GetMotionVectors(Vector3[] motionVectors)
        {
            Debug.LogError("Motion vectors not available when using the Mono solver.");
        }

        static Workspace[] AllocateWorkspaces(ZivaRTRig rig)
        {
            var ws = new Workspace[rig.m_Patches.Length];
            for (int p = 0; p < rig.m_Patches.Length; p++)
            {
                var patch = rig.m_Patches[p];

                // TODO: temporary fix to support zero kernel centers. Should be fixed properly.
                if (patch.hasZeroKernelCenters())
                    continue;

                int subspaceCoeffsSize, displacementsSize;
                if (rig.m_CorrectiveType == CorrectiveType.TensorSkin)
                {
                    subspaceCoeffsSize = patch.RbfCoeffs.Rows;
                    displacementsSize = 3 * patch.ReducedBasis.LeadingDimension;
                }
                else if (rig.m_CorrectiveType == CorrectiveType.EigenSkin)
                {
                    subspaceCoeffsSize = patch.RbfCoeffs.Rows;
                    displacementsSize = patch.ReducedBasis.LeadingDimension;
                }
                else // FullSpace
                {
                    subspaceCoeffsSize = 0;
                    displacementsSize = patch.RbfCoeffs.Rows;
                }

                ws[p] = new Workspace()
                {
                    LocalWeightedPose = new float[patch.KernelCenters.Cols],
                    Phi = new float[patch.KernelCenters.Rows + 1],
                    SubspaceCoeff = new float[subspaceCoeffsSize],
                    Displacements = new float[displacementsSize]
                };
            }

            return ws;
        }

        void Correctives(NativeArray<float> currentPose)
        {
            var correctiveType = m_Rig.m_CorrectiveType;
            for (int p = 0; p < m_Rig.m_Patches.Length; p++)
            {
                var patch = m_Rig.m_Patches[p];
                var workspace = m_Workspaces[p];

                // TODO: temporary fix to support zero kernel centers. Should be fixed properly.
                if (patch.hasZeroKernelCenters())
                    continue;

                restrictedWeightedPose(patch, currentPose, workspace.LocalWeightedPose);
                kernelFunctions(patch, workspace.LocalWeightedPose, workspace.Phi);
                rbfInterpolate(
                    patch, workspace.Phi,
                    correctiveType != CorrectiveType.FullSpace ? workspace.SubspaceCoeff
                    : workspace.Displacements);

                // NOTE: accumulateDisplacements function is not thread-safe if parallelizing patches
                if (correctiveType == CorrectiveType.TensorSkin)
                {
                    expandTensorSkinning(patch, workspace.SubspaceCoeff, workspace.Displacements);
                    accumulateDisplacementsNonInterleaved(patch, workspace.Displacements, m_Shape);
                }
                else if (correctiveType == CorrectiveType.EigenSkin)
                {
                    expandEigenSkinning(patch, workspace.SubspaceCoeff, workspace.Displacements);
                    accumulateDisplacements(patch, workspace.Displacements, m_Shape);
                }
                else
                {
                    Assert.IsTrue(correctiveType == CorrectiveType.FullSpace);
                    accumulateDisplacements(patch, workspace.Displacements, m_Shape);
                }
            }
        }

        static void restrictedWeightedPose(Patch patch, NativeArray<float> currentPose, float[] localWeightedPose)
        {
            var indices = patch.PoseIndices;
            var shift = patch.PoseShift;
            var scale = patch.PoseScale;

            Assert.AreEqual(indices.Length, shift.Length);
            Assert.AreEqual(indices.Length, scale.Length);
            Assert.AreEqual(indices.Length, localWeightedPose.Length);

            for (int i = 0; i < indices.Length; i++)
                localWeightedPose[i] = scale[i] * currentPose[indices[i]] + shift[i];
        }

        static void kernelFunctions(Patch patch, float[] pose, float[] phi)
        {
            var poseVectorDim = patch.KernelCenters.Cols;
            var numKernels = patch.KernelCenters.Rows;
            var lda = patch.KernelCenters.LeadingDimension;
            var kernelCenters = patch.KernelCenters.Values;

            Assert.AreEqual(poseVectorDim, pose.Length);

            for (int k = 0; k < phi.Length; ++k)
                phi[k] = 0.0f;

            for (int i = 0; i != poseVectorDim; ++i)
            {
                var poseStartIndex = lda * i;
                for (int k = 0; k != numKernels; ++k)
                {
                    float dx = pose[i] - patch.KernelScale[i] * kernelCenters[k + poseStartIndex];
                    phi[k] += dx * dx;
                }
            }

            for (int k = 0; k != numKernels; ++k)
                phi[k] = patch.ScalePerKernel[k] * Mathf.Sqrt(phi[k]);

            phi[numKernels] = patch.ScalePerKernel[numKernels];
        }

        static void rbfInterpolate(Patch patch, float[] phi, float[] out_array)
        {
            var mat = patch.RbfCoeffs;
            var lda = mat.LeadingDimension;
            naiveGEMV(mat.Rows, mat.Cols, lda, mat.Values, phi, out_array);
            diagonalProduct(patch.ScalePerRBFCoeff, out_array);
        }

        static void expandEigenSkinning(Patch patch, float[] subspaceCoeff, float[] out_array)
        {
            var mat = patch.ReducedBasis;
            var lda = mat.LeadingDimension;
            naiveGEMV(mat.Rows, mat.Cols, lda, mat.Values, subspaceCoeff, out_array);
            diagonalProduct(patch.ScalePerVertex, out_array);
        }

        static void expandTensorSkinning(Patch patch, float[] subspaceCoeff, float[] out_array)
        {
            var mat = patch.ReducedBasis;
            var lda = mat.LeadingDimension;
            naiveGemv3(mat.Rows, mat.Cols, lda, mat.Values, subspaceCoeff, lda, out_array);
            diagonalProduct3(lda, patch.ScalePerVertex, out_array);
        }

        static void accumulateDisplacements(Patch patch, float[] displacements, float[] shape)
        {
            var vertexIndices = patch.Vertices;
            for (int a = 0; a < vertexIndices.Length; a++)
            {
                uint vtx = vertexIndices[a] * 3;
                shape[vtx + 0] += displacements[3 * a + 0];
                shape[vtx + 1] += displacements[3 * a + 1];
                shape[vtx + 2] += displacements[3 * a + 2];
            }
        }

        static void accumulateDisplacementsNonInterleaved(
            Patch patch,
            float[] displacements,
            float[] shape)
        {
            var vertexIndices = patch.Vertices;

            var lda = patch.ReducedBasis.LeadingDimension;
            var xStart = lda * 0;
            var yStart = lda * 1;
            var zStart = lda * 2;

            for (int a = 0; a < vertexIndices.Length; a++)
            {
                uint vtx = vertexIndices[a] * 3;
                shape[vtx + 0] += displacements[xStart + a];
                shape[vtx + 1] += displacements[yStart + a];
                shape[vtx + 2] += displacements[zStart + a];
            }
        }

        static void naiveGEMV(int rows, int cols, int lda, sbyte[] matrix, float[] in_array, float[] out_array)
        {
            for (int a = 0; a < out_array.Length; a++)
                out_array[a] = 0.0f;
            for (int col = 0; col != cols; ++col)
            {
                for (int row = 0; row != rows; ++row)
                    out_array[row] += matrix[row + col * lda] * in_array[col];
            }
        }
        static void naiveGEMV(int rows, int cols, int lda, short[] matrix, float[] in_array, float[] out_array)
        {
            for (int a = 0; a < out_array.Length; a++)
                out_array[a] = 0.0f;
            for (int col = 0; col != cols; ++col)
            {
                for (int row = 0; row != rows; ++row)
                    out_array[row] += matrix[row + col * lda] * in_array[col];
            }
        }

        static void naiveGemv3(
            int rows,
            int cols,
            int lda,
            sbyte[] matrix,
            float[] in_array,
            int ldaOut,
            float[] out_array)
        {
            for (int a = 0; a < out_array.Length; a++)
                out_array[a] = 0.0f;
            for (int m = 0; m != cols; ++m)
            {
                for (int v = 0; v != rows; ++v)
                {
                    for (int d = 0; d != 3; ++d)
                        out_array[d * ldaOut + v] += in_array[3 * m + d] * matrix[m * lda + v];
                }
            }
        }

        static void diagonalProduct(float[] scale, float[] v)
        {
            for (int i = 0; i != Math.Min(scale.Length, v.Length); ++i)
                v[i] *= scale[i];
        }

        static void diagonalProduct3(int lda, float[] scale, float[] v)
        {
            Assert.IsTrue(lda >= scale.Length);
            for (int i = 0; i != scale.Length; ++i)
            {
                for (int d = 0; d != 3; ++d)
                    v[d * lda + i] *= scale[i];
            }
        }

        void Skinning(NativeArray<float> worldTransforms)
        {
            worldToRelative(m_Rig.m_Skinning.RestPoseInverse, worldTransforms, m_RelativeTransforms);
            linearBlendSkinning(m_Rig, m_RelativeTransforms, m_Shape);
        }

        static void worldToRelative(
            float[] restPoseInverse,
            NativeArray<float> worldTransforms,
            float[] relativeTransforms)
        {
            for (int a = 0; a < restPoseInverse.Length / 12; a++)
            {
                var idx = a * 12;

                for (int row = 0; row < 3; ++row)
                {
                    for (int col = 0; col != 4; ++col)
                    {
                        float rij = 0;
                        for (int k = 0; k != 3; ++k)
                            rij += worldTransforms[row + 3 * k + idx] * restPoseInverse[k + 3 * col + idx];
                        relativeTransforms[row + 3 * col + idx] = rij;
                    }
                    relativeTransforms[row + 3 * 3 + idx] += worldTransforms[row + 3 * 3 + idx];
                }
            }
        }

        static void linearBlendSkinning(
            ZivaRTRig character,
            float[] relativeTransforms,
            float[] shape)
        {
            int[] colStarts = character.m_Skinning.SkinningWeights.ColStarts;
            int[] indices = character.m_Skinning.SkinningWeights.RowIndices;
            float[] weights = character.m_Skinning.SkinningWeights.W;
            var numVerts = character.m_Skinning.SkinningWeights.NumCols;

            var weightedTransform = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var xworld = new float[] { 0, 0, 0 };

            for (int vert = 0; vert < numVerts; ++vert)
            {
                var start = colStarts[vert];
                var numActiveJoints = colStarts[vert + 1] - colStarts[vert];

                for (int i = 0; i < 12; ++i)
                    weightedTransform[i] = 0.0f;

                for (int j = 0; j != numActiveJoints; ++j)
                {
                    var startIndex = 12 * indices[start + j];
                    var weight = weights[start + j];
                    for (int i = 0; i != 12; ++i)
                        weightedTransform[i] += weight * relativeTransforms[startIndex + i];
                }

                // Do the 3x4 transform from corrected rest position to skinned positions
                xworld[0] = weightedTransform[9];
                xworld[1] = weightedTransform[10];
                xworld[2] = weightedTransform[11];

                var vertIndex = 3 * vert;
                for (int i = 0; i != 3; ++i)
                {
                    for (int k = 0; k != 3; ++k)
                        xworld[i] += weightedTransform[i + 3 * k] * shape[vertIndex + k];
                }
                for (int i = 0; i != 3; ++i)
                    shape[vertIndex + i] = xworld[i];
            }
        }

        void ToUnitySpace(ref float[] shape)
        {
            int numVertices = shape.Length / 3;
            // Invert x-component of each vertex
            for (int v = 0; v < numVertices; ++v)
                shape[3 * v + 0] *= -1.0f;
        }

        public override void Dispose() { }
    }
}
