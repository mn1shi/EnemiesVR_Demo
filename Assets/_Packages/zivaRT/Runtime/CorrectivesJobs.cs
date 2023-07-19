using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;

namespace Unity.ZivaRTPlayer
{
    // Helper class for consistently calculating the size/structure of the workspace buffers needed
    // by the ZivaRT correctives computations.
    internal struct PatchWorkspaceSizes
    {
        public int PhiSize;
        public int SubspaceCoeffsSize;
        public int DisplacementsSize;

        // Distance between spatial components of each vertex within the displacmeents buffer.
        // The displacements buffer might be interleaved XYZXYZXYZ..., in which case stride = 1.
        // Or it might be XXXX...YYYY....ZZZZ..., in which case stride tells you the distance from
        // the X sub-array to the Y sub-array (and from Y to Z).
        public int DisplacementsStride;

        public static PatchWorkspaceSizes Calc(CorrectiveType correctiveType, Patch patch)
        {
            int numKernels = patch.KernelCenters.Rows;
            int phiSize = numKernels + 1;
            int rbfCoeffsNumRows = patch.RbfCoeffs.Rows;
            int reducedBasisLDA = patch.ReducedBasis.LeadingDimension;

            if (correctiveType == CorrectiveType.TensorSkin)
            {
                return new PatchWorkspaceSizes
                {
                    PhiSize = phiSize,
                    SubspaceCoeffsSize = rbfCoeffsNumRows,
                    DisplacementsSize = 3 * reducedBasisLDA,
                    DisplacementsStride = reducedBasisLDA,
                };
            }
            else if (correctiveType == CorrectiveType.EigenSkin)
            {
                return new PatchWorkspaceSizes
                {
                    PhiSize = phiSize,
                    SubspaceCoeffsSize = rbfCoeffsNumRows,
                    DisplacementsSize = reducedBasisLDA,
                    DisplacementsStride = 1,
                };
            }
            else // FullSpace
            {
                return new PatchWorkspaceSizes
                {
                    PhiSize = phiSize,
                    SubspaceCoeffsSize = 0,
                    DisplacementsSize = rbfCoeffsNumRows,
                    DisplacementsStride = 1,
                };
            }
        }
    }

    // Compute corrective displacements for every Patch of a ZivaRT character asset.
    // ParallelFor is over patches.
    // NOTE: Each iteration of the parallel-for is writing to multiple entries of the output
    // displacements buffer, which is non-standard IJobFor behavior. It means we have to
    // be extra careful that each iteration writes to separate sub-arrays of displacements, ie:
    // the displacements sub-array of the patch.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct ComputePatchCorrectivesJob : IJobFor
    {
        const int k_InvalidIndex = Int32.MinValue;
        [ReadOnly]
        public NativeArray<int> PatchDisplacementStrides;
        [ReadOnly]
        public NativeArray<int> DisplacementToShapeVertexIndices;
        [ReadOnly]
        public NativeArray<int> PatchDisplacementCounts;

        // Disabling safety here is required for the single threaded execution to work.
        // The reason is because in the single threaded mode, patch displacements need to do
        // scattered writes and the safety system complains that we are writing to indices we
        // shouldn't be. But since only one thread will do the write, we don't care about safety.
        // Code that accesses shape in parallel should proceed with extreme caution, since
        // the safety system will no longer warn about potential data races.
        [NativeDisableParallelForRestriction]
        public NativeArray<float3> Shape;

        // Input
        [ReadOnly]
        public NativeArray<float> CurrentPose;

        // The correctiveType determines the specifics of the algorithm used to compute
        // corrective displacements.
        CorrectiveType m_CorrectiveType;

        // Static buffers for data from all patches, concatenated together.
        [ReadOnly]
        NativeArray<ushort> m_PoseIndices;
        [ReadOnly]
        NativeArray<float> m_PoseScale;
        [ReadOnly]
        NativeArray<float> m_PoseShift;
        [ReadOnly]
        NativeArray<float> m_KernelScale;
        [ReadOnly]
        NativeArray<sbyte> m_KernelCenters;
        [ReadOnly]
        NativeArray<float> m_ScalePerKernel;
        [ReadOnly]
        NativeArray<short> m_RbfCoeffs;
        [ReadOnly]
        NativeArray<float> m_ScalePerRBFCoeff;
        [ReadOnly]
        NativeArray<sbyte> m_ReducedBasis;
        [ReadOnly]
        NativeArray<float> m_ScalePerVertex;

        // Local read-write workspaces, with space for all patches at once.
        NativeArray<float> m_LocalWeightedPose;
        NativeArray<float> m_Phi;
        NativeArray<float> m_SubspaceCoeff;

        // Struct that stores indices into all of the concatenated buffers at once.
        // Useful for storing the start indices of all sub-arrays corresponding to a single patch.
        // Also can be used to store the size of all sub-arrays corresponding to a single patch.
        struct PatchIndices
        {
            public int PoseIndices; // AKA numPoseVectorDimensions

            // The following always have the same size as poseIndices:
            public int PoseScale { get { return PoseIndices; } }
            public int PoseShift { get { return PoseIndices; } }
            public int KernelScale { get { return PoseIndices; } }

            public int KernelCenters;
            public int ScalePerKernel;
            public int RbfCoeffs;
            public int ScalePerRBFCoeff;
            public int ReducedBasis;
            public int ScalePerVertex;

            // localWeightedPose size is always the same as poseIndices size.
            public int LocalWeightedPose { get { return PoseIndices; } }

            public int Phi;
            public int SubspaceCoeff;

            public static PatchIndices Zero
            {
                get // The first set of sub-arrays will all start at index 0
                {
                    return new PatchIndices
                    {
                        PoseIndices = 0,
                        KernelCenters = 0,
                        ScalePerKernel = 0,
                        RbfCoeffs = 0,
                        ScalePerRBFCoeff = 0,
                        ReducedBasis = 0,
                        ScalePerVertex = 0,
                        Phi = 0,
                        SubspaceCoeff = 0,
                    };
                }
            }
        }
        // Indexing information for sub-arrays of all patches.
        [ReadOnly]
        NativeArray<PatchIndices> m_PatchStarts;

        // Dimensions of all of the matrices stored for each of the patches.
        struct MatrixDimensions
        {
            public int Rows;
            public int Cols;

            // Leading dimension, aka distance from the start of one column to the start of the next.
            // May be larger than the number of rows, if there is padding in the matrix data.
            public int Lda;

            // Pull out the dimension data from a given ZivaRT Matrix object.
            public static MatrixDimensions Read<T>(MatrixX<T> matrix)
            {
                return new MatrixDimensions
                {
                    Rows = matrix.Rows,
                    Cols = matrix.Cols,
                    Lda = matrix.LeadingDimension,
                };
            }
        }
        [ReadOnly]
        NativeArray<MatrixDimensions> m_KernelCenterMatrixDims;
        [ReadOnly]
        NativeArray<MatrixDimensions> m_RbfCoeffsMatrixDims;
        [ReadOnly]
        NativeArray<MatrixDimensions> m_ReducedBasisMatrixDims;

        // Start indices of sub-arrays of the displacements buffer for each patch.
        // Separate from the start indices of other data buffers so that it can be shared with other
        // code that might need to refer to it to index into the displacements array.
        [ReadOnly]
        NativeArray<int> m_DisplacementsStarts;

        // Output: displacements for all patches
        [NativeDisableParallelForRestriction] // We want to write values for an entire patch all at once
        NativeArray<float> m_Displacements;
        public NativeArray<float> Displacements { get { return m_Displacements; } }

        // Start indices for the sub-arrays of `displacments` that correspond to each patch.
        // Has length `NumPatches + 1` with the last element containing one-past-the-end of the last
        // patch's sub-array. This allows more convenient lookup of index ranges for every patch.
        public NativeArray<int>.ReadOnly DisplacementsStarts { get { return m_DisplacementsStarts.AsReadOnly(); } }

        // Markers for profiling pieces of the job.
        public struct Profile
        {
            public Profiling.ProfilerMarker PatchIndexing;
            public Profiling.ProfilerMarker RestrictedWeightedPose;
            public Profiling.ProfilerMarker KernelFunctions;
            public Profiling.ProfilerMarker Gemv_i16;
            public Profiling.ProfilerMarker Gemv_i8;
            public Profiling.ProfilerMarker DiagonalProduct;
            public Profiling.ProfilerMarker Gemv3;
            public Profiling.ProfilerMarker DiagonalProduct3;

            public static Profile Init()
            {
                return new Profile
                {
                    PatchIndexing = new Profiling.ProfilerMarker("PatchIndexing"),
                    RestrictedWeightedPose = new Profiling.ProfilerMarker("RestrictedWeightedPose"),
                    KernelFunctions = new Profiling.ProfilerMarker("KernelFunctions"),
                    Gemv_i8 = new Profiling.ProfilerMarker("Gemv_i8"),
                    Gemv_i16 = new Profiling.ProfilerMarker("Gemv_i16"),
                    DiagonalProduct = new Profiling.ProfilerMarker("DiagonalProduct"),
                    Gemv3 = new Profiling.ProfilerMarker("Gemv3"),
                    DiagonalProduct3 = new Profiling.ProfilerMarker("DiagonalProduct3"),
                };
            }

            public static void Begin(Profiling.ProfilerMarker marker)
            {
                // marker.Begin(); // Comment/uncomment to enable/disable profiling.
            }
            public static void End(Profiling.ProfilerMarker marker)
            {
                // marker.End(); // Comment/uncomment to enable/disable profiling.
            }
        }
        public Profile Profiling; // Must be initialized and passed in from managed code. :(

        static int PadSimd(int size)
        {
            // I would like to determine this value based on the supported SIMD instruction set
            // (4-floats for SSE, 8-floats for AVX), but sadly I'm not getting consistent
            // values from Burst.Intrinsics.X86.Avx.IsAvxSupported every time this function is
            // called! I don't know why; maybe sometimes the Burst compiler isn't fully enabled
            // with SIMD intrinsics turned on???
            // To work-around this problem, just use the maximum size that we might need. :(
            const int simdPackSize = 8;

            // Round up to nearest pack size
            return (size + (simdPackSize - 1)) / simdPackSize * simdPackSize;
        }

        bool m_AccumulateDisplacementsInPatchCorrectivesJob;

        // Set up all internal buffers with concatenated data from all Patches of the given Ziva asset.
        public void Initialize(ZivaRTRig zivaAsset, bool accumulateDisplacementsInPatchCorrectivesJob)
        {
            ReleaseBuffers();

            this.m_AccumulateDisplacementsInPatchCorrectivesJob = accumulateDisplacementsInPatchCorrectivesJob;
            this.m_CorrectiveType = zivaAsset.m_CorrectiveType;

            int numPatches = zivaAsset.m_Patches.Length;

            this.m_PatchStarts = new NativeArray<PatchIndices>(numPatches + 1, Allocator.Persistent);
            this.m_DisplacementsStarts = new NativeArray<int>(numPatches + 1, Allocator.Persistent);
            PatchDisplacementStrides = JobSolver.AllocateConditionally<int>(
                accumulateDisplacementsInPatchCorrectivesJob, zivaAsset.m_Patches.Length);

            // We need to merge all of the data from all patches into a single set of buffers,
            // with indexing information so we can look-up sub-arrays for a specific patch later.
            // Go through every patch to figuure out how much space is needed for each sub-array.
            this.m_PatchStarts[0] = PatchIndices.Zero;
            this.m_DisplacementsStarts[0] = 0;
            for (int p = 0; p < numPatches; ++p)
            {
                var currPatch = zivaAsset.m_Patches[p];
                var currPatchStarts = this.m_PatchStarts[p];

                var workspaceSizes = PatchWorkspaceSizes.Calc(zivaAsset.m_CorrectiveType, currPatch);

                this.m_PatchStarts[p + 1] = new PatchIndices
                {
                    PoseIndices = currPatchStarts.PoseIndices + currPatch.PoseIndices.Length,
                    KernelCenters = currPatchStarts.KernelCenters + currPatch.KernelCenters.Values.Length,
                    ScalePerKernel = currPatchStarts.ScalePerKernel + PadSimd(currPatch.ScalePerKernel.Length),
                    RbfCoeffs = currPatchStarts.RbfCoeffs + currPatch.RbfCoeffs.Values.Length,
                    ScalePerRBFCoeff = currPatchStarts.ScalePerRBFCoeff + currPatch.ScalePerRBFCoeff.Length,
                    ReducedBasis = currPatchStarts.ReducedBasis + currPatch.ReducedBasis.Values.Length,
                    ScalePerVertex = currPatchStarts.ScalePerVertex + PadSimd(currPatch.ScalePerVertex.Length),
                    Phi = currPatchStarts.Phi + PadSimd(workspaceSizes.PhiSize),
                    SubspaceCoeff = currPatchStarts.SubspaceCoeff + PadSimd(workspaceSizes.SubspaceCoeffsSize),
                };

                this.m_DisplacementsStarts[p + 1] = this.m_DisplacementsStarts[p] + workspaceSizes.DisplacementsSize;

                if (accumulateDisplacementsInPatchCorrectivesJob)
                {
                    PatchDisplacementStrides[p] = workspaceSizes.DisplacementsStride;
                }
            }

            // Divide by 3 to get the number of displacement verts. This may include padding.
            var numDisplacements = m_DisplacementsStarts[zivaAsset.m_Patches.Length] / 3;
            DisplacementToShapeVertexIndices = JobSolver.AllocateConditionally<int>(
                accumulateDisplacementsInPatchCorrectivesJob, numDisplacements);
            PatchDisplacementCounts = JobSolver.AllocateConditionally<int>(
                accumulateDisplacementsInPatchCorrectivesJob, zivaAsset.m_Patches.Length);

            // If single threaded then we need these NativeArrays to be actually allocated
            if (accumulateDisplacementsInPatchCorrectivesJob)
            {
                for (int i = 0; i < DisplacementToShapeVertexIndices.Length; ++i)
                {
                    // A negative value indicates this displacement does not have a shape vertex.
                    // This can happen due to padding between patches.
                    DisplacementToShapeVertexIndices[i] = k_InvalidIndex;
                }

                for (int i = 0; i < PatchDisplacementCounts.Length; ++i)
                {
                    PatchDisplacementCounts[i] = zivaAsset.m_Patches[i].Vertices.Length;
                }

                for (int p = 0; p < zivaAsset.m_Patches.Length; ++p)
                {
                    Patch patch = zivaAsset.m_Patches[p];
                    int patchVertexCount = patch.Vertices.Length;
                    int patchStartIndex = m_DisplacementsStarts[p] / 3;
                    for (int patchVertex = 0; patchVertex < patchVertexCount; ++patchVertex)
                    {
                        int displacementIndex = patchStartIndex + patchVertex;
                        int shapeVertexIndex = (int)patch.Vertices[patchVertex];

                        // Check that this displacement hasn't had a shape vertex assigned.
                        Assert.IsTrue(DisplacementToShapeVertexIndices[displacementIndex] == k_InvalidIndex);
                        DisplacementToShapeVertexIndices[displacementIndex] = shapeVertexIndex;
                    }
                }
            }

            // Allocate space to hold all of the data from all of the patches.
            // Because of the way we constructed the patchStarts data, the last entries in that array
            // contain the total size of each buffer (aka: one-past-the-end index).
            var patchTotals = this.m_PatchStarts[this.m_PatchStarts.Length - 1];
            this.m_PoseIndices = new NativeArray<ushort>(patchTotals.PoseIndices, Allocator.Persistent);
            this.m_PoseScale = new NativeArray<float>(patchTotals.PoseScale, Allocator.Persistent);
            this.m_PoseShift = new NativeArray<float>(patchTotals.PoseShift, Allocator.Persistent);
            this.m_KernelScale = new NativeArray<float>(patchTotals.KernelScale, Allocator.Persistent);
            this.m_KernelCenters = new NativeArray<sbyte>(patchTotals.KernelCenters, Allocator.Persistent);
            this.m_ScalePerKernel = new NativeArray<float>(patchTotals.ScalePerKernel, Allocator.Persistent);
            this.m_RbfCoeffs = new NativeArray<short>(patchTotals.RbfCoeffs, Allocator.Persistent);
            this.m_ScalePerRBFCoeff = new NativeArray<float>(patchTotals.ScalePerRBFCoeff, Allocator.Persistent);
            this.m_ReducedBasis = new NativeArray<sbyte>(patchTotals.ReducedBasis, Allocator.Persistent);
            this.m_ScalePerVertex = new NativeArray<float>(patchTotals.ScalePerVertex, Allocator.Persistent);
            this.m_LocalWeightedPose = new NativeArray<float>(patchTotals.LocalWeightedPose, Allocator.Persistent);
            this.m_Phi = new NativeArray<float>(patchTotals.Phi, Allocator.Persistent);
            this.m_SubspaceCoeff = new NativeArray<float>(patchTotals.SubspaceCoeff, Allocator.Persistent);
            this.m_Displacements = new NativeArray<float>(m_DisplacementsStarts[numPatches], Allocator.Persistent);

            // Copy data from each patch into the concatenated buffers.
            for (int p = 0; p < numPatches; ++p)
            {
                var patch = zivaAsset.m_Patches[p];
                // TODO: temporary fix to support zero kernel centers. Should be fixed properly.
                if (patch.hasZeroKernelCenters()) continue;
                var patchStarts = this.m_PatchStarts[p];
                var patchSizes = CalcPatchArraySizes(patchStarts, this.m_PatchStarts[p + 1]);
                this.m_PoseIndices.GetSubArray(patchStarts.PoseIndices, patchSizes.PoseIndices)
                    .CopyFrom(patch.PoseIndices);
                this.m_PoseScale.GetSubArray(patchStarts.PoseScale, patchSizes.PoseScale)
                    .CopyFrom(patch.PoseScale);
                this.m_PoseShift.GetSubArray(patchStarts.PoseShift, patchSizes.PoseShift)
                    .CopyFrom(patch.PoseShift);
                this.m_KernelScale.GetSubArray(patchStarts.KernelScale, patchSizes.KernelScale)
                    .CopyFrom(patch.KernelScale);
                this.m_KernelCenters.GetSubArray(patchStarts.KernelCenters, patchSizes.KernelCenters)
                    .CopyFrom(patch.KernelCenters.Values);
                this.m_ScalePerKernel.GetSubArray(patchStarts.ScalePerKernel, patch.ScalePerKernel.Length)
                    .CopyFrom(patch.ScalePerKernel);
                this.m_RbfCoeffs.GetSubArray(patchStarts.RbfCoeffs, patchSizes.RbfCoeffs)
                    .CopyFrom(patch.RbfCoeffs.Values);
                this.m_ScalePerRBFCoeff.GetSubArray(patchStarts.ScalePerRBFCoeff, patchSizes.ScalePerRBFCoeff)
                    .CopyFrom(patch.ScalePerRBFCoeff);
                this.m_ReducedBasis.GetSubArray(patchStarts.ReducedBasis, patchSizes.ReducedBasis)
                    .CopyFrom(patch.ReducedBasis.Values);
                this.m_ScalePerVertex.GetSubArray(patchStarts.ScalePerVertex, patch.ScalePerVertex.Length)
                    .CopyFrom(patch.ScalePerVertex);
            }

            // Copy dimensions of all our matrices.
            this.m_KernelCenterMatrixDims = new NativeArray<MatrixDimensions>(numPatches, Allocator.Persistent);
            this.m_RbfCoeffsMatrixDims = new NativeArray<MatrixDimensions>(numPatches, Allocator.Persistent);
            this.m_ReducedBasisMatrixDims = new NativeArray<MatrixDimensions>(numPatches, Allocator.Persistent);
            for (int p = 0; p < numPatches; ++p)
            {
                var patch = zivaAsset.m_Patches[p];
                this.m_KernelCenterMatrixDims[p] = MatrixDimensions.Read(patch.KernelCenters);
                this.m_RbfCoeffsMatrixDims[p] = MatrixDimensions.Read(patch.RbfCoeffs);
                this.m_ReducedBasisMatrixDims[p] = MatrixDimensions.Read(patch.ReducedBasis);
            }

            // Do some data validation checks to make sure everything will work at runtime.
            int simdPackSize = PadSimd(1);
            for (int p = 0; p < numPatches; ++p)
            {
                var patchSizes = CalcPatchArraySizes(this.m_PatchStarts[p], this.m_PatchStarts[p + 1]);

                // Vectorized KernelFunctions evaluation requires SIMD alignment of kernelCenters,
                // scalePerKernel, and phi.
                int kernelCentersLDA = this.m_KernelCenterMatrixDims[p].Lda;
                if (kernelCentersLDA % simdPackSize != 0)
                {
                    UnityEngine.Debug.LogError(string.Format(
                        "KernelCenters matrix LDA must be aligned for SIMD. Patch #{0}, LDA = {1}", p,
                        kernelCentersLDA));
                }
                if (patchSizes.ScalePerKernel % simdPackSize != 0)
                {
                    UnityEngine.Debug.LogError(
                        string.Format("ScalePerKernel must be padded for SIMD. Patch #{0}, Size = {1}", p,
                                      patchSizes.ScalePerKernel));
                }
                // Phi should have at least max(simdAligned(numKernels), numKernels+1) elements.
                int numKernels = this.m_KernelCenterMatrixDims[p].Rows;
                if (patchSizes.Phi < math.max(PadSimd(numKernels), numKernels + 1))
                {
                    UnityEngine.Debug.LogError(string.Format(
                        "Phi vector must be aligned for SIMD. Patch #{0}, Size = {1}, numKernels = {2}", p,
                        patchSizes.Phi, numKernels));
                }

                // Vectorized RbfInterpolate and EigenSkinning requires SIMD alignment of
                // rbfCoeffs and subspaceCoeffs.
                int rbfCoeffsLDA = this.m_RbfCoeffsMatrixDims[p].Lda;
                if (rbfCoeffsLDA % simdPackSize != 0)
                {
                    UnityEngine.Debug.LogError(
                        string.Format("RBFCoeffs matrix LDA must be aligned for SIMD. Patch #{0}, LDA = {1}",
                                      p, rbfCoeffsLDA));
                }
                if (patchSizes.SubspaceCoeff % simdPackSize != 0)
                {
                    UnityEngine.Debug.LogError(
                        string.Format("SubspaceCoeff must be padded for SIMD. Patch #{0}, Size = {1}", p,
                                      patchSizes.SubspaceCoeff));
                }

                if (m_CorrectiveType == CorrectiveType.TensorSkin)
                {
                    // We are using the LDA of the reducedBasisMatrix as the LDA for our
                    // displacements matrix when we do tensor skinning. Our SIMD implementation
                    // requires this LDA to be padded to a SIMD boundary. It also requires the
                    // scalePerVertex array to be padded in the same way.
                    int reducedBasisLDA = this.m_ReducedBasisMatrixDims[p].Lda;
                    if (reducedBasisLDA % simdPackSize != 0)
                    {
                        UnityEngine.Debug.LogError(string.Format(
                            "ReducedBasis matrix LDA must be aligned for SIMD. Patch #{0}, LDA = {1}", p,
                            reducedBasisLDA));
                    }
                    if (patchSizes.ScalePerVertex % simdPackSize != 0)
                    {
                        UnityEngine.Debug.LogError(
                            string.Format("ScalePerVertex array not padded for SIMD. Patch #{0}, Length = {1}",
                                          p, patchSizes.ScalePerVertex));
                    }
                }
            }
        }

        static void SafeDispose<T>(ref NativeArray<T> array)
            where T : struct
        {
            if (array.IsCreated)
                array.Dispose();
        }
        public void ReleaseBuffers()
        {
            SafeDispose(ref m_PoseIndices);
            SafeDispose(ref m_PoseScale);
            SafeDispose(ref m_PoseShift);
            SafeDispose(ref m_KernelScale);
            SafeDispose(ref m_KernelCenters);
            SafeDispose(ref m_ScalePerKernel);
            SafeDispose(ref m_RbfCoeffs);
            SafeDispose(ref m_ScalePerRBFCoeff);
            SafeDispose(ref m_ReducedBasis);
            SafeDispose(ref m_ScalePerVertex);

            SafeDispose(ref m_LocalWeightedPose);
            SafeDispose(ref m_Phi);
            SafeDispose(ref m_SubspaceCoeff);

            SafeDispose(ref m_PatchStarts);

            SafeDispose(ref m_KernelCenterMatrixDims);
            SafeDispose(ref m_RbfCoeffsMatrixDims);
            SafeDispose(ref m_ReducedBasisMatrixDims);

            SafeDispose(ref m_DisplacementsStarts);
            SafeDispose(ref m_Displacements);
            SafeDispose(ref DisplacementToShapeVertexIndices);
            SafeDispose(ref PatchDisplacementCounts);
            SafeDispose(ref PatchDisplacementStrides);
        }

        // The number of patchs in the Ziva character asset.
        // This is the number of iterations to be executed by the ParallelFor.
        public int NumPatches { get { return m_KernelCenterMatrixDims.Length; } }

        public void Execute(int index)
        {
            Profile.Begin(Profiling.PatchIndexing);

            // Create sub-arrays containing the data relevant to the current patch.
            var patchStarts = this.m_PatchStarts[index];
            var patchSizes = CalcPatchArraySizes(patchStarts, this.m_PatchStarts[index + 1]);

            var poseIndices = this.m_PoseIndices.GetSubArray(patchStarts.PoseIndices, patchSizes.PoseIndices);
            var poseScale = this.m_PoseScale.GetSubArray(patchStarts.PoseScale, patchSizes.PoseScale);
            var poseShift = this.m_PoseShift.GetSubArray(patchStarts.PoseShift, patchSizes.PoseShift);
            var kernelScale = this.m_KernelScale.GetSubArray(patchStarts.KernelScale, patchSizes.KernelScale);
            var kernelCenters = this.m_KernelCenters.GetSubArray(patchStarts.KernelCenters, patchSizes.KernelCenters);
            var scalePerKernel =
                this.m_ScalePerKernel.GetSubArray(patchStarts.ScalePerKernel, patchSizes.ScalePerKernel);
            var rbfCoeffs = this.m_RbfCoeffs.GetSubArray(patchStarts.RbfCoeffs, patchSizes.RbfCoeffs);
            var scalePerRBFCoeff =
                this.m_ScalePerRBFCoeff.GetSubArray(patchStarts.ScalePerRBFCoeff, patchSizes.ScalePerRBFCoeff);
            var reducedBasis = this.m_ReducedBasis.GetSubArray(patchStarts.ReducedBasis, patchSizes.ReducedBasis);
            var scalePerVertex =
                this.m_ScalePerVertex.GetSubArray(patchStarts.ScalePerVertex, patchSizes.ScalePerVertex);

            var localWeightedPose =
                this.m_LocalWeightedPose.GetSubArray(patchStarts.LocalWeightedPose, patchSizes.LocalWeightedPose);
            var phi = this.m_Phi.GetSubArray(patchStarts.Phi, patchSizes.Phi);

            // Not all correctiveTypes use subspaceCoeffs, so its size might be zero.
            // Don't build its subarray until later when we know we need it.

            var kernelCentersMatrixDims = this.m_KernelCenterMatrixDims[index];
            int numKernels = kernelCentersMatrixDims.Rows;
            var rbfCoeffsMatrixDims = this.m_RbfCoeffsMatrixDims[index];
            var reducedBasisMatrixDims = this.m_ReducedBasisMatrixDims[index];

            int displacementsStart = m_DisplacementsStarts[index];
            int displacementsSize = m_DisplacementsStarts[index + 1] - displacementsStart;
            var displacements = this.m_Displacements.GetSubArray(displacementsStart, displacementsSize);

            Profile.End(Profiling.PatchIndexing);

            RestrictedWeightedPose(
                localWeightedPose, this.CurrentPose, poseIndices, poseScale, poseShift);
            KernelFunctions(
                phi, localWeightedPose, kernelCenters, numKernels, kernelCentersMatrixDims.Lda, kernelScale,
                scalePerKernel);

            if (m_CorrectiveType == CorrectiveType.TensorSkin)
            {
                var subspaceCoeffs =
                    this.m_SubspaceCoeff.GetSubArray(patchStarts.SubspaceCoeff, patchSizes.SubspaceCoeff);

                RbfInterpolate(subspaceCoeffs, phi, rbfCoeffs, rbfCoeffsMatrixDims, scalePerRBFCoeff);
                int displacementsLDA = reducedBasisMatrixDims.Lda;
                ExpandTensorSkinning(
                    displacements, displacementsLDA, reducedBasis, reducedBasisMatrixDims, subspaceCoeffs,
                    scalePerVertex);

                if (m_AccumulateDisplacementsInPatchCorrectivesJob)
                {
                    // Normally, the accumulation of displacements would be done with a separate job
                    // but when running on one thread, this leads to worse performance due to the worse
                    // cache behavior of how the AccumulateDisplacementsJob accesses the displacements.
                    // Because of this, if we are running with only one thread we sum the displacements
                    // right here instead of kicking an entirely separate job. There are two problems
                    // with the AccumulateDisplacementsJob when it comes to single threaded perf:
                    //
                    // 1. AccumulateDisplacementsJob is parallelized over the output shape vertices,
                    //    which is fine for avoiding synchronization on the writes to the output shape
                    //    vertices, but this forces random reads of the displacements due to each
                    //    shape vertex having multiple displacement contributions which may come from
                    //    many different patches. Since the displacement count is usually significantly
                    //    larger than the shape vertex count (at least in typical face assets), we've basically
                    //    opted to have more cache misses which ends up being a net loss in performance
                    //    in the single threaded case.
                    //
                    // 2. The use of the AccumulateDisplacementsJob forces two passes over the displacements
                    //    which is a significant cost. ComputePatchCorrectivesJob will process patches at a time
                    //    and instead of finishing all of the work for that patch right there while the displacment
                    //    data for the patch is hot, we process other patches and kick out the displacement data
                    //    from earlier patches before we can even get to running the AccumulateDisplacementsJob.
                    //    By accumulating right here, we take advantage of the hot displacement data and gain
                    //    more speed than if we were to use a job instead.
                    AccumulateDisplacementsSingleThreaded(
                        DisplacementsStarts[index], PatchDisplacementStrides[index], PatchDisplacementCounts[index],
                        this.m_Displacements, DisplacementToShapeVertexIndices, Shape);
                }
            }
            else if (m_CorrectiveType == CorrectiveType.EigenSkin)
            {
                var subspaceCoeffs =
                    this.m_SubspaceCoeff.GetSubArray(patchStarts.SubspaceCoeff, patchSizes.SubspaceCoeff);

                RbfInterpolate(subspaceCoeffs, phi, rbfCoeffs, rbfCoeffsMatrixDims, scalePerRBFCoeff);
                ExpandEigenSkinning(
                    displacements, reducedBasis, reducedBasisMatrixDims, subspaceCoeffs, scalePerVertex);
            }
            else // FullSpace
            {
                RbfInterpolate(displacements, phi, rbfCoeffs, rbfCoeffsMatrixDims, scalePerRBFCoeff);
            }
        }

        static void AccumulateDisplacementsSingleThreaded(
            int patchStart, int patchStride, int patchDisplacementCount, NativeArray<float> displacements,
            NativeArray<int> displacementToShapeVertexIndices, NativeArray<float3> shapeVertices)
        {
            NativeArray<float> displacementX =
                displacements.GetSubArray(patchStart + (0 * patchStride), patchDisplacementCount);
            NativeArray<float> displacementY =
                displacements.GetSubArray(patchStart + (1 * patchStride), patchDisplacementCount);
            NativeArray<float> displacementZ =
                displacements.GetSubArray(patchStart + (2 * patchStride), patchDisplacementCount);
            int patchDisplacementStartIndex = patchStart / 3;

            for (int i = 0; i < patchDisplacementCount; ++i)
            {
                shapeVertices[displacementToShapeVertexIndices[patchDisplacementStartIndex + i]] +=
                    new float3 { x = displacementX[i], y = displacementY[i], z = displacementZ[i] };
            }
        }

        // Given the start indices of two adjacent patches (aka: patchStarts[i] and patchStarts[i+1]),
        // compute the sizes of the sub-arrays for the first patch. This just the difference between
        // start indices of the two given patches.
        static PatchIndices CalcPatchArraySizes(PatchIndices starts, PatchIndices ends)
        {
            return new PatchIndices
            {
                PoseIndices = ends.PoseIndices - starts.PoseIndices,
                KernelCenters = ends.KernelCenters - starts.KernelCenters,
                ScalePerKernel = ends.ScalePerKernel - starts.ScalePerKernel,
                RbfCoeffs = ends.RbfCoeffs - starts.RbfCoeffs,
                ScalePerRBFCoeff = ends.ScalePerRBFCoeff - starts.ScalePerRBFCoeff,
                ReducedBasis = ends.ReducedBasis - starts.ReducedBasis,
                ScalePerVertex = ends.ScalePerVertex - starts.ScalePerVertex,
                Phi = ends.Phi - starts.Phi,
                SubspaceCoeff = ends.SubspaceCoeff - starts.SubspaceCoeff,
            };
        }

        void RestrictedWeightedPose(
            NativeArray<float> localWeightedPose,
            NativeArray<float> currentPose,
            NativeArray<ushort> poseIndices,
            NativeArray<float> poseScale,
            NativeArray<float> poseShift)
        {
            Profile.Begin(Profiling.RestrictedWeightedPose);

            int count = poseIndices.Length;
            for (int i = 0; i < count; i++)
            {
                localWeightedPose[i] = poseScale[i] * currentPose[poseIndices[i]] + poseShift[i];
            }

            Profile.End(Profiling.RestrictedWeightedPose);
        }

        void KernelFunctions(
            NativeArray<float> phi,
            NativeArray<float> localWeightedPose,
            NativeArray<sbyte> kernelCenters,
            int numKernels,
            int kernelCentersLDA,
            NativeArray<float> kernelScale,
            NativeArray<float> scalePerKernel)
        {
            Profile.Begin(Profiling.KernelFunctions);

            // Zero out the phi vector that we will accumulate into.
            int count = phi.Length;
            for (int k = 0; k < count; ++k)
                phi[k] = 0.0f;

            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                KernelFunctions_AVX2(
                    phi, localWeightedPose, kernelCenters, numKernels, kernelCentersLDA, kernelScale,
                    scalePerKernel);
            }
            else
            {
                KernelFunctions_float4(
                    phi, localWeightedPose, kernelCenters, numKernels, kernelCentersLDA, kernelScale,
                    scalePerKernel);
            }

            // Set the constant term (aka "bias" term).
            phi[numKernels] = scalePerKernel[numKernels];

            Profile.End(Profiling.KernelFunctions);
        }

        static void KernelFunctions_float4(
            NativeArray<float> phi,
            NativeArray<float> localWeightedPose,
            NativeArray<sbyte> kernelCenters,
            int numKernels,
            int kernelCentersLDA,
            NativeArray<float> kernelScale,
            NativeArray<float> scalePerKernel)
        {
            int poseDimensions = localWeightedPose.Length;

            int i = 0;

#if true // Fuse 4 columns
            for (; i <= poseDimensions - 4; i += 4)
            {
                var kernelCentersCol1 = kernelCenters.GetSubArray((i + 0) * kernelCentersLDA, kernelCentersLDA);
                var kernelCentersCol2 = kernelCenters.GetSubArray((i + 1) * kernelCentersLDA, kernelCentersLDA);
                var kernelCentersCol3 = kernelCenters.GetSubArray((i + 2) * kernelCentersLDA, kernelCentersLDA);
                var kernelCentersCol4 = kernelCenters.GetSubArray((i + 3) * kernelCentersLDA, kernelCentersLDA);

                float pose1 = localWeightedPose[i + 0];
                float pose2 = localWeightedPose[i + 1];
                float pose3 = localWeightedPose[i + 2];
                float pose4 = localWeightedPose[i + 3];

                float scale1 = kernelScale[i + 0];
                float scale2 = kernelScale[i + 1];
                float scale3 = kernelScale[i + 2];
                float scale4 = kernelScale[i + 3];

                for (int k = 0; k < numKernels; k += 4)
                {
                    var kernelCenters1 = ConvertSbytesToFloat4(kernelCentersCol1, k);
                    var kernelCenters2 = ConvertSbytesToFloat4(kernelCentersCol2, k);
                    var kernelCenters3 = ConvertSbytesToFloat4(kernelCentersCol3, k);
                    var kernelCenters4 = ConvertSbytesToFloat4(kernelCentersCol4, k);

                    var phi_k = phi.ReinterpretLoad<float4>(k);

                    var dx1 = pose1 - scale1 * kernelCenters1;
                    var dx2 = pose2 - scale2 * kernelCenters2;
                    var dx3 = pose3 - scale3 * kernelCenters3;
                    var dx4 = pose4 - scale4 * kernelCenters4;

                    phi_k += dx1 * dx1;
                    phi_k += dx2 * dx2;
                    phi_k += dx3 * dx3;
                    phi_k += dx4 * dx4;

                    phi.ReinterpretStore(k, phi_k);
                }
            }
#endif
            for (; i != poseDimensions; ++i)
            {
                var kernelCentersCol = kernelCenters.GetSubArray(i * kernelCentersLDA, kernelCentersLDA);

                float pose_i = localWeightedPose[i];
                float scale_i = kernelScale[i];

                for (int k = 0; k < numKernels; k += 4)
                {
                    var kernelCenters_k = ConvertSbytesToFloat4(kernelCentersCol, k);
                    var phi_k = phi.ReinterpretLoad<float4>(k);
                    var dx = pose_i - scale_i * kernelCenters_k;
                    phi_k += dx * dx;
                    phi.ReinterpretStore(k, phi_k);
                }
            }

            float4 tiny = 5e-38f;
            for (int k = 0; k < numKernels; k += 4)
            {
                var phi_k = phi.ReinterpretLoad<float4>(k);
                var scale = scalePerKernel.ReinterpretLoad<float4>(k);

                // phi_k = scale * math.sqrt(phi_k);

                // rsqrt(x) = 1/sqrt(x); rsqrt is a faster instruction than sqrt
                // sqrt(x) ~= x/sqrt(x), modulo a few Newton iterations
                // We are okay with the loss of precision that this introduces.
                var rSquared = phi_k + tiny; // Perturb to avoid divide-by-zero
                phi_k = scale * rSquared * math.rsqrt(rSquared);

                phi.ReinterpretStore(k, phi_k);
            }
        }

        static void KernelFunctions_AVX2(
            NativeArray<float> phi,
            NativeArray<float> localWeightedPose,
            NativeArray<sbyte> kernelCenters,
            int numKernels,
            int kernelCentersLDA,
            NativeArray<float> kernelScale,
            NativeArray<float> scalePerKernel)
        {
            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                int poseDimensions = localWeightedPose.Length;

                int i = 0;

#if true // Fuse 4 columns
                for (; i <= poseDimensions - 4; i += 4)
                {
                    var kernelCentersCol1 = kernelCenters.GetSubArray((i + 0) * kernelCentersLDA, kernelCentersLDA);
                    var kernelCentersCol2 = kernelCenters.GetSubArray((i + 1) * kernelCentersLDA, kernelCentersLDA);
                    var kernelCentersCol3 = kernelCenters.GetSubArray((i + 2) * kernelCentersLDA, kernelCentersLDA);
                    var kernelCentersCol4 = kernelCenters.GetSubArray((i + 3) * kernelCentersLDA, kernelCentersLDA);

                    var pose1 = new Burst.Intrinsics.v256(localWeightedPose[i + 0]);
                    var pose2 = new Burst.Intrinsics.v256(localWeightedPose[i + 1]);
                    var pose3 = new Burst.Intrinsics.v256(localWeightedPose[i + 2]);
                    var pose4 = new Burst.Intrinsics.v256(localWeightedPose[i + 3]);

                    var scale1 = new Burst.Intrinsics.v256(kernelScale[i + 0]);
                    var scale2 = new Burst.Intrinsics.v256(kernelScale[i + 1]);
                    var scale3 = new Burst.Intrinsics.v256(kernelScale[i + 2]);
                    var scale4 = new Burst.Intrinsics.v256(kernelScale[i + 3]);

                    for (int k = 0; k < numKernels; k += 8)
                    {
                        var kernelCenters1 = ConvertSBytesToFloats256(kernelCentersCol1, k);
                        var kernelCenters2 = ConvertSBytesToFloats256(kernelCentersCol2, k);
                        var kernelCenters3 = ConvertSBytesToFloats256(kernelCentersCol3, k);
                        var kernelCenters4 = ConvertSBytesToFloats256(kernelCentersCol4, k);

                        var phi_k = phi.ReinterpretLoad<Burst.Intrinsics.v256>(k);

                        var dx1 = Burst.Intrinsics.X86.Fma.mm256_fmsub_ps(scale1, kernelCenters1, pose1);
                        var dx2 = Burst.Intrinsics.X86.Fma.mm256_fmsub_ps(scale2, kernelCenters2, pose2);
                        var dx3 = Burst.Intrinsics.X86.Fma.mm256_fmsub_ps(scale3, kernelCenters3, pose3);
                        var dx4 = Burst.Intrinsics.X86.Fma.mm256_fmsub_ps(scale4, kernelCenters4, pose4);

                        phi_k = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(dx1, dx1, phi_k);
                        phi_k = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(dx2, dx2, phi_k);
                        phi_k = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(dx3, dx3, phi_k);
                        phi_k = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(dx4, dx4, phi_k);

                        phi.ReinterpretStore(k, phi_k);
                    }
                }
#endif
                for (; i != poseDimensions; ++i)
                {
                    var kernelCentersCol = kernelCenters.GetSubArray(i * kernelCentersLDA, kernelCentersLDA);

                    var pose_i = new Burst.Intrinsics.v256(localWeightedPose[i]);
                    var scale_i = new Burst.Intrinsics.v256(kernelScale[i]);

                    for (int k = 0; k < numKernels; k += 8)
                    {
                        var kernelCenters_k = ConvertSBytesToFloats256(kernelCentersCol, k);
                        var phi_k = phi.ReinterpretLoad<Burst.Intrinsics.v256>(k);
                        var dx = Burst.Intrinsics.X86.Fma.mm256_fmsub_ps(scale_i, kernelCenters_k, pose_i);
                        phi_k = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(dx, dx, phi_k);
                        phi.ReinterpretStore(k, phi_k);
                    }
                }

                var tiny = new Burst.Intrinsics.v256(5e-38f);
                for (int k = 0; k < numKernels; k += 8)
                {
                    var phi_k = phi.ReinterpretLoad<Burst.Intrinsics.v256>(k);
                    var scale = scalePerKernel.ReinterpretLoad<Burst.Intrinsics.v256>(k);

                    // phi_k = scale * math.sqrt(phi_k);

                    // rsqrt(x) = 1/sqrt(x); rsqrt is a faster instruction than sqrt
                    // sqrt(x) ~= x/sqrt(x), modulo a few Newton iterations
                    // We are okay with the loss of precision that this introduces.
                    // Perturb to avoid divide-by-zero:
                    var rSquared = Burst.Intrinsics.X86.Avx.mm256_add_ps(phi_k, tiny);
                    var invSqrt = Burst.Intrinsics.X86.Avx.mm256_rsqrt_ps(rSquared);
                    var rSquaredScaled = Burst.Intrinsics.X86.Avx.mm256_mul_ps(scale, rSquared);
                    phi_k = Burst.Intrinsics.X86.Avx.mm256_mul_ps(rSquaredScaled, invSqrt);

                    phi.ReinterpretStore(k, phi_k);
                }
            }
            else
            {
                throw new Exception("AVX2 not supported!");
            }
        }

        void RbfInterpolate(
            NativeArray<float> output,
            NativeArray<float> phi,
            NativeArray<short> rbfCoeffs,
            MatrixDimensions rbfCoeffsDims,
            NativeArray<float> scalePerRBFCoeff)
        {
            Gemv(rbfCoeffsDims.Rows, rbfCoeffsDims.Cols, rbfCoeffsDims.Lda, rbfCoeffs, phi, output);
            DiagonalProduct(scalePerRBFCoeff, output);
        }

        void ExpandTensorSkinning(
            NativeArray<float> displacements,
            int displacementsLDA,
            NativeArray<sbyte> reducedBasis,
            MatrixDimensions reducedBasisDims,
            NativeArray<float> subspaceCoeffs,
            NativeArray<float> scalePerVertex)
        {
            Gemv3(
                reducedBasisDims.Rows, reducedBasisDims.Cols, reducedBasisDims.Lda, reducedBasis, subspaceCoeffs,
                displacementsLDA, displacements);
            DiagonalProduct3(displacementsLDA, scalePerVertex, displacements);
        }

        void ExpandEigenSkinning(
            NativeArray<float> displacements,
            NativeArray<sbyte> reducedBasis,
            MatrixDimensions reducedBasisDims,
            NativeArray<float> subspaceCoeffs,
            NativeArray<float> scalePerVertex)
        {
            NaiveGEMV(
                reducedBasisDims.Rows, reducedBasisDims.Cols, reducedBasisDims.Lda, reducedBasis, subspaceCoeffs,
                displacements);
            DiagonalProduct(scalePerVertex, displacements);
        }

        void NaiveGEMV(
            int rows,
            int cols,
            int lda,
            NativeArray<sbyte> matrix,
            NativeArray<float> in_array,
            NativeArray<float> out_array)
        {
            Profile.Begin(Profiling.Gemv_i8);

            int count = out_array.Length;
            for (int a = 0; a < count; a++)
                out_array[a] = 0.0f;

            for (int col = 0; col != cols; ++col)
            {
                for (int row = 0; row != rows; ++row)
                {
                    out_array[row] += matrix[row + col * lda] * in_array[col];
                }
            }

            Profile.End(Profiling.Gemv_i8);
        }

        void Gemv(
            int rows,
            int cols,
            int lda,
            NativeArray<short> matrix,
            NativeArray<float> in_array,
            NativeArray<float> out_array)
        {
            Profile.Begin(Profiling.Gemv_i16);

            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                Gemv_AVX2(rows, cols, lda, matrix, in_array, out_array);
            }
            else
            {
                Gemv_float4(rows, cols, lda, matrix, in_array, out_array);
            }

            Profile.End(Profiling.Gemv_i16);
        }

        static float4 ConvertShortsToFloat4(NativeArray<short> shorts, int startIndex)
        {
            return new float4(
                shorts[startIndex + 0], shorts[startIndex + 1], shorts[startIndex + 2], shorts[startIndex + 3]);
        }

        static void Gemv_float4(
            int rows,
            int cols,
            int lda,
            NativeArray<short> matrix,
            NativeArray<float> in_array,
            NativeArray<float> out_array)
        {
            int c = 0;
#if true // Fuse 4 columns
            for (; c < cols - 4; c += 4)
            {
                var matrixCol1 = matrix.GetSubArray((c + 0) * lda, lda);
                var matrixCol2 = matrix.GetSubArray((c + 1) * lda, lda);
                var matrixCol3 = matrix.GetSubArray((c + 2) * lda, lda);
                var matrixCol4 = matrix.GetSubArray((c + 3) * lda, lda);

                float in1 = in_array[c + 0];
                float in2 = in_array[c + 1];
                float in3 = in_array[c + 2];
                float in4 = in_array[c + 3];

                for (int r = 0; r < rows; r += 4)
                {
                    var matrixCol_r1 = ConvertShortsToFloat4(matrixCol1, r);
                    var matrixCol_r2 = ConvertShortsToFloat4(matrixCol2, r);
                    var matrixCol_r3 = ConvertShortsToFloat4(matrixCol3, r);
                    var matrixCol_r4 = ConvertShortsToFloat4(matrixCol4, r);

                    var out_r = out_array.ReinterpretLoad<float4>(r);

                    out_r += matrixCol_r1 * in1;
                    out_r += matrixCol_r2 * in2;
                    out_r += matrixCol_r3 * in3;
                    out_r += matrixCol_r4 * in4;

                    out_array.ReinterpretStore(r, out_r);
                }
            }
#endif
            for (; c != cols; ++c)
            {
                var matrixCol = matrix.GetSubArray(c * lda, lda);
                float in_c = in_array[c];

                for (int r = 0; r < rows; r += 4)
                {
                    var matrixCol_r = ConvertShortsToFloat4(matrixCol, r);
                    var out_r = out_array.ReinterpretLoad<float4>(r);
                    out_r += matrixCol_r * in_c;
                    out_array.ReinterpretStore(r, out_r);
                }
            }
        }

        static Burst.Intrinsics.v256 ConvertShortsToFloats256(NativeArray<short> shorts, int startIndex)
        {
            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                var s = shorts.ReinterpretLoad<Burst.Intrinsics.v128>(startIndex);
                var i = Burst.Intrinsics.X86.Avx2.mm256_cvtepi16_epi32(s);
                return Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(i);
            }
            else
            {
                throw new Exception("AVX2 not supported!");
            }
        }

        static void Gemv_AVX2(
            int rows,
            int cols,
            int lda,
            NativeArray<short> matrix,
            NativeArray<float> in_array,
            NativeArray<float> out_array)
        {
            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                int c = 0;
#if true // Fuse 4 columns
                for (; c < cols - 4; c += 4)
                {
                    var matrixCol1 = matrix.GetSubArray((c + 0) * lda, lda);
                    var matrixCol2 = matrix.GetSubArray((c + 1) * lda, lda);
                    var matrixCol3 = matrix.GetSubArray((c + 2) * lda, lda);
                    var matrixCol4 = matrix.GetSubArray((c + 3) * lda, lda);

                    var in1 = new Burst.Intrinsics.v256(in_array[c + 0]);
                    var in2 = new Burst.Intrinsics.v256(in_array[c + 1]);
                    var in3 = new Burst.Intrinsics.v256(in_array[c + 2]);
                    var in4 = new Burst.Intrinsics.v256(in_array[c + 3]);

                    for (int r = 0; r < rows; r += 8)
                    {
                        var matrixCol_r1 = ConvertShortsToFloats256(matrixCol1, r);
                        var matrixCol_r2 = ConvertShortsToFloats256(matrixCol2, r);
                        var matrixCol_r3 = ConvertShortsToFloats256(matrixCol3, r);
                        var matrixCol_r4 = ConvertShortsToFloats256(matrixCol4, r);

                        var out_r = out_array.ReinterpretLoad<Burst.Intrinsics.v256>(r);

                        out_r = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(matrixCol_r1, in1, out_r);
                        out_r = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(matrixCol_r2, in2, out_r);
                        out_r = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(matrixCol_r3, in3, out_r);
                        out_r = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(matrixCol_r4, in4, out_r);

                        out_array.ReinterpretStore(r, out_r);
                    }
                }
#endif
                for (; c != cols; ++c)
                {
                    var matrixCol = matrix.GetSubArray(c * lda, lda);
                    var in_c = new Burst.Intrinsics.v256(in_array[c]);
                    for (int r = 0; r < rows; r += 8)
                    {
                        var matrixCol_r = ConvertShortsToFloats256(matrixCol, r);
                        var out_r = out_array.ReinterpretLoad<Burst.Intrinsics.v256>(r);
                        out_r = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(matrixCol_r, in_c, out_r);
                        out_array.ReinterpretStore(r, out_r);
                    }
                }
            }
            else
            {
                throw new Exception("AVX2 not supported!");
            }
        }

        void DiagonalProduct(NativeArray<float> scale, NativeArray<float> v)
        {
            Profile.Begin(Profiling.DiagonalProduct);

            int count = Math.Min(scale.Length, v.Length);
            for (int i = 0; i != count; ++i)
                v[i] *= scale[i];

            Profile.End(Profiling.DiagonalProduct);
        }

        void Gemv3(
            int rows,
            int cols,
            int lda,
            NativeArray<sbyte> matrix,
            NativeArray<float> in_matrix3,
            int ldaOut,
            NativeArray<float> out_matrix3)
        {
            Profile.Begin(Profiling.Gemv3);

            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(matrix, in_matrix3);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(in_matrix3, out_matrix3);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(matrix, out_matrix3);

            int count = out_matrix3.Length;
            for (int a = 0; a < count; a++)
                out_matrix3[a] = 0.0f;

            var outX = out_matrix3.GetSubArray(0, ldaOut);
            var outY = out_matrix3.GetSubArray(ldaOut, ldaOut);
            var outZ = out_matrix3.GetSubArray(ldaOut * 2, ldaOut);

            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(outX, outY);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(outX, outZ);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(outY, outZ);

            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                Gemv3_AVX2(rows, cols, lda, matrix, in_matrix3, outX, outY, outZ);
            }
            else
            {
                Gemv3(rows, cols, lda, matrix, in_matrix3, outX, outY, outZ);
            }

            Profile.End(Profiling.Gemv3);
        }

        static float4 ConvertSbytesToFloat4(NativeArray<sbyte> bytes, int startIdx)
        {
            return new float4(
                bytes[startIdx + 0], bytes[startIdx + 1], bytes[startIdx + 2], bytes[startIdx + 3]);
        }

        static void Gemv3(
            int rows,
            int cols,
            int lda,
            NativeArray<sbyte> matrix,
            NativeArray<float> in_matrix3,
            NativeArray<float> outX,
            NativeArray<float> outY,
            NativeArray<float> outZ)
        {
            int c = 0;

            for (; c < cols; ++c)
            {
                var matrixCol = matrix.GetSubArray(c * lda, lda);

                // in_matrix3 is row-major
                float3 inRow = in_matrix3.ReinterpretLoad<float3>(c * 3);

                for (int r = 0; r < rows; r += 4)
                {
                    float4 matrixCoeff = ConvertSbytesToFloat4(matrixCol, r);

                    float4 x = outX.ReinterpretLoad<float4>(r);
                    float4 y = outY.ReinterpretLoad<float4>(r);
                    float4 z = outZ.ReinterpretLoad<float4>(r);

                    x += inRow.x * matrixCoeff;
                    y += inRow.y * matrixCoeff;
                    z += inRow.z * matrixCoeff;

                    outX.ReinterpretStore(r, x);
                    outY.ReinterpretStore(r, y);
                    outZ.ReinterpretStore(r, z);
                }
            }
        }

        static Burst.Intrinsics.v256 ConvertSBytesToFloats256(NativeArray<sbyte> bytes, int startIndex)
        {
            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                var b = new Burst.Intrinsics.v128(bytes.ReinterpretLoad<ulong>(startIndex), 0);
                var i = Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(b);
                return Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(i);
            }
            else
            {
                throw new Exception("AVX2 not supported!");
            }
        }

        static void Gemv3_AVX2(
            int rows,
            int cols,
            int lda,
            NativeArray<sbyte> matrix,
            NativeArray<float> in_matrix3,
            NativeArray<float> outX,
            NativeArray<float> outY,
            NativeArray<float> outZ)
        {
            if (Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                int c = 0;

#if true // Fuse four columns at a time
                for (; c <= cols - 4; c += 4)
                {
                    var matrixCol1 = matrix.GetSubArray(c * lda, lda);
                    var matrixCol2 = matrix.GetSubArray((c + 1) * lda, lda);
                    var matrixCol3 = matrix.GetSubArray((c + 2) * lda, lda);
                    var matrixCol4 = matrix.GetSubArray((c + 3) * lda, lda);

                    // in_matrix3 is row-major
                    float3 inRow1 = in_matrix3.ReinterpretLoad<float3>(c * 3);
                    var inX1 = new Burst.Intrinsics.v256(inRow1.x);
                    var inY1 = new Burst.Intrinsics.v256(inRow1.y);
                    var inZ1 = new Burst.Intrinsics.v256(inRow1.z);

                    float3 inRow2 = in_matrix3.ReinterpretLoad<float3>((c + 1) * 3);
                    var inX2 = new Burst.Intrinsics.v256(inRow2.x);
                    var inY2 = new Burst.Intrinsics.v256(inRow2.y);
                    var inZ2 = new Burst.Intrinsics.v256(inRow2.z);

                    float3 inRow3 = in_matrix3.ReinterpretLoad<float3>((c + 2) * 3);
                    var inX3 = new Burst.Intrinsics.v256(inRow3.x);
                    var inY3 = new Burst.Intrinsics.v256(inRow3.y);
                    var inZ3 = new Burst.Intrinsics.v256(inRow3.z);

                    float3 inRow4 = in_matrix3.ReinterpretLoad<float3>((c + 3) * 3);
                    var inX4 = new Burst.Intrinsics.v256(inRow4.x);
                    var inY4 = new Burst.Intrinsics.v256(inRow4.y);
                    var inZ4 = new Burst.Intrinsics.v256(inRow4.z);

                    for (int r = 0; r < rows; r += 8)
                    {
                        var matrixCoeff1 = ConvertSBytesToFloats256(matrixCol1, r);
                        var matrixCoeff2 = ConvertSBytesToFloats256(matrixCol2, r);
                        var matrixCoeff3 = ConvertSBytesToFloats256(matrixCol3, r);
                        var matrixCoeff4 = ConvertSBytesToFloats256(matrixCol4, r);

                        var x = outX.ReinterpretLoad<Burst.Intrinsics.v256>(r);
                        var y = outY.ReinterpretLoad<Burst.Intrinsics.v256>(r);
                        var z = outZ.ReinterpretLoad<Burst.Intrinsics.v256>(r);

                        x = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inX1, matrixCoeff1, x);
                        y = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inY1, matrixCoeff1, y);
                        z = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inZ1, matrixCoeff1, z);

                        x = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inX2, matrixCoeff2, x);
                        y = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inY2, matrixCoeff2, y);
                        z = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inZ2, matrixCoeff2, z);

                        x = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inX3, matrixCoeff3, x);
                        y = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inY3, matrixCoeff3, y);
                        z = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inZ3, matrixCoeff3, z);

                        x = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inX4, matrixCoeff4, x);
                        y = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inY4, matrixCoeff4, y);
                        z = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inZ4, matrixCoeff4, z);

                        outX.ReinterpretStore(r, x);
                        outY.ReinterpretStore(r, y);
                        outZ.ReinterpretStore(r, z);
                    }
                }
#endif

                for (; c < cols; ++c)
                {
                    var matrixCol = matrix.GetSubArray(c * lda, lda);

                    // in_matrix3 is row-major
                    float3 inRow = in_matrix3.ReinterpretLoad<float3>(c * 3);
                    var inX = new Burst.Intrinsics.v256(inRow.x);
                    var inY = new Burst.Intrinsics.v256(inRow.y);
                    var inZ = new Burst.Intrinsics.v256(inRow.z);

                    for (int r = 0; r < rows; r += 8)
                    {
                        var matrixCoeff = ConvertSBytesToFloats256(matrixCol, r);

                        var x = outX.ReinterpretLoad<Burst.Intrinsics.v256>(r);
                        var y = outY.ReinterpretLoad<Burst.Intrinsics.v256>(r);
                        var z = outZ.ReinterpretLoad<Burst.Intrinsics.v256>(r);

                        x = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inX, matrixCoeff, x);
                        y = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inY, matrixCoeff, y);
                        z = Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(inZ, matrixCoeff, z);

                        outX.ReinterpretStore(r, x);
                        outY.ReinterpretStore(r, y);
                        outZ.ReinterpretStore(r, z);
                    }
                }
            }
            else
            {
                throw new Exception("AVX2 not supported!");
            }
        }

        void DiagonalProduct3(int lda, NativeArray<float> scale, NativeArray<float> v)
        {
            Profile.Begin(Profiling.DiagonalProduct3);

            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(scale, v);

            var x = v.GetSubArray(0, lda);
            var y = v.GetSubArray(lda, lda);
            var z = v.GetSubArray(lda * 2, lda);

            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(x, y);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(x, z);
            Unity.Burst.CompilerServices.Aliasing.ExpectNotAliased(y, z);

            if (Burst.Intrinsics.X86.Avx.IsAvxSupported)
            {
                DiagonalProduct3_AVX(scale, x, y, z);
            }
            else
            {
                DiagonalProduct3(scale, x, y, z);
            }

            Profile.End(Profiling.DiagonalProduct3);
        }

        static void DiagonalProduct3(
            NativeArray<float> scale,
            NativeArray<float> x,
            NativeArray<float> y,
            NativeArray<float> z)
        {
            int count = Math.Min(scale.Length, x.Length);
            for (int i = 0; i < count; i += 4)
            {
                float4 s = scale.ReinterpretLoad<float4>(i);

                float4 x_i = x.ReinterpretLoad<float4>(i);
                float4 y_i = y.ReinterpretLoad<float4>(i);
                float4 z_i = z.ReinterpretLoad<float4>(i);

                x_i *= s;
                y_i *= s;
                z_i *= s;

                x.ReinterpretStore(i, x_i);
                y.ReinterpretStore(i, y_i);
                z.ReinterpretStore(i, z_i);
            }
        }

        static void DiagonalProduct3_AVX(
            NativeArray<float> scale,
            NativeArray<float> x,
            NativeArray<float> y,
            NativeArray<float> z)
        {
            if (Burst.Intrinsics.X86.Avx.IsAvxSupported)
            {
                int count = Math.Min(scale.Length, x.Length);
                for (int i = 0; i < count; i += 8)
                {
                    var s = scale.ReinterpretLoad<Burst.Intrinsics.v256>(i);

                    var x_i = x.ReinterpretLoad<Burst.Intrinsics.v256>(i);
                    var y_i = y.ReinterpretLoad<Burst.Intrinsics.v256>(i);
                    var z_i = z.ReinterpretLoad<Burst.Intrinsics.v256>(i);

                    x_i = Burst.Intrinsics.X86.Avx.mm256_mul_ps(x_i, s);
                    y_i = Burst.Intrinsics.X86.Avx.mm256_mul_ps(y_i, s);
                    z_i = Burst.Intrinsics.X86.Avx.mm256_mul_ps(z_i, s);

                    x.ReinterpretStore(i, x_i);
                    y.ReinterpretStore(i, y_i);
                    z.ReinterpretStore(i, z_i);
                }
            }
            else
            {
                throw new Exception("AVX not supported!");
            }
        }
    }

    // Job to accumulate the displacements computed for all patches onto a single shape buffer.
    // ParallelFor over the vertex positions in the shape array.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    struct AccumulateDisplacementsJob : IJobFor
    {
        // Input:
        // Displacements computed by all patches, unified into a single array.
        // The specific structure of the displacements data depends on the correctiveType of the
        // ZRT asset. It may be interleaved (XYZXYZXYZ...) or non-interleaved (XXX...YYY...ZZZ) with
        // arbitrary padding.
        // The Initialization() step sets up the indexing information in displacementInfluences and
        // influencesStarts to properly handle the different possible ordering and strides of the
        // displacement data.
        [ReadOnly]
        public NativeArray<float> Displacements;

        // Input-Output: Shape into which the displacements are accumulated.
        // Receives the "rest shape" as input, then applies corrective displacements on output.
        public NativeArray<float3> Shape;

        // Flat array of displacement influence indices for all vertices.
        // A vertex can have an arbitrary number of influences.
        // Use influencesStarts to determine the (contiguous) set of displacement indices for a
        // vertex. Then look up the actual current values of those influences in the displacements
        // array.
        // NOTE: Because the displacements may be non-interleaved, we need both the start index of
        // the first component AND the stride from there to the subsequent components.
        struct VertexDisplacementIndices
        {
            public int Start;
            public int Stride;
        }
        [ReadOnly]
        NativeArray<VertexDisplacementIndices> m_DisplacementInfluences;

        // Start indices of the sub-arrays of displacementInfluences for each vertex.
        // NOTE: Should have length numVertices + 1, with the first entry == 0 and the
        // last entry == displacementInfluences.Length
        [ReadOnly]
        NativeArray<int> m_InfluencesStarts;

        // Accumulate all displacements for a single vertex.
        public void Execute(int index)
        {
            int influencesStartIdx = m_InfluencesStarts[index];
            int influencesEndIdx = m_InfluencesStarts[index + 1];

            for (int i = influencesStartIdx; i < influencesEndIdx; ++i)
            {
                var displacementIdx = m_DisplacementInfluences[i];
                int start = displacementIdx.Start;
                int stride = displacementIdx.Stride;

                float3 displacement;
                displacement.x = Displacements[start + stride * 0];
                displacement.y = Displacements[start + stride * 1];
                displacement.z = Displacements[start + stride * 2];
                Shape[index] += displacement;
            }
        }

        // Number of vertices in the shape being accumulated onto.
        // This is the number of ParallelFor iterations to be invoked.
        public int NumShapeVertices { get { return m_InfluencesStarts.Length - 1; } }

        public void ReleaseBuffers()
        {
            if (m_DisplacementInfluences.IsCreated)
                m_DisplacementInfluences.Dispose();
            if (m_InfluencesStarts.IsCreated)
                m_InfluencesStarts.Dispose();
        }

        // For each vertex in the final shape, determine the set of patch-vertices that influence
        // them and store the indices into the full concatenated displacements buffer where they
        // live.
        // patchDisplacementStarts should come from the ComputePatchCorrectivesJob that is
        // producing the displacements buffer data. It tells us where the sub-arrays corresponding
        // to each patch start and end. From there we compute more-specific indices of the vertices
        // within those patches that displace each specific shape vertex.
        public void Initialize(
            ZivaRTRig zivaAsset,
            NativeArray<int>.ReadOnly patchDisplacementStarts,
            bool accumulateDisplacementsInPatchCorrectivesJob)
        {
            ReleaseBuffers();

            // We have a list of patches and the shape vertices they influence.
            // What we need is a list of the patch-vertices that influence each shape vertex.
            var vertexInfluenceLists = PatchInfluences.InvertPatchVertexMap(zivaAsset);

            // From the list of shape vertex influences, construct the indices for indexing into the
            // final flattened/concatenated list of influences.
            var influencesStarts = PatchInfluences.CalcInfluenceStartIndices(vertexInfluenceLists);
            this.m_InfluencesStarts = new NativeArray<int>(influencesStarts, Allocator.Persistent);

            // For the next step, it will be convenient to have information about the size/stride of
            // the displacements for each patch ready to query.
            var patchWorkspaceSizes = new PatchWorkspaceSizes[zivaAsset.m_Patches.Length];
            for (int p = 0; p < patchWorkspaceSizes.Length; ++p)
            {
                patchWorkspaceSizes[p] = PatchWorkspaceSizes.Calc(zivaAsset.m_CorrectiveType, zivaAsset.m_Patches[p]);
            }

            // Convert our patch-vertex index pairs into indices into the flattened/concatenated array
            // of all displacements from all patches.
            var numShapeVertices = vertexInfluenceLists.Length;
            int totalDisplacementInfluences = influencesStarts[numShapeVertices];
            this.m_DisplacementInfluences = JobSolver.AllocateConditionally<VertexDisplacementIndices>(!accumulateDisplacementsInPatchCorrectivesJob, totalDisplacementInfluences);

            if (!accumulateDisplacementsInPatchCorrectivesJob)
            {
                for (int shapeVertex = 0; shapeVertex < numShapeVertices; ++shapeVertex)
                {
                    int influencesStartIdx = influencesStarts[shapeVertex];
                    var vertexInfluences = vertexInfluenceLists[shapeVertex];
                    for (int i = 0; i < vertexInfluences.Count; ++i)
                    {
                        var vertexInfluence = vertexInfluences[i];

                        int patchStartIdx = patchDisplacementStarts[vertexInfluence.Patch];

                        int displacementIdx, displacementsStride;
                        if (zivaAsset.m_CorrectiveType == CorrectiveType.TensorSkin)
                        {
                            // The displacement data is non-interleaved XXX...YYY...ZZZ...
                            // We need the index of the X-component, and then the stride from there to the
                            // Y- and Z-components.
                            // The displacement data may have padding between the component sub-arrays, so take
                            // that into account in the stride (PatchWorkspaceSizes does this for us).
                            int xOffset = vertexInfluence.Vertex;
                            displacementIdx = patchStartIdx + xOffset;

                            displacementsStride = patchWorkspaceSizes[vertexInfluence.Patch].DisplacementsStride;
                        }
                        else
                        {
                            // The displacement data is interleaved XYZXYZXYZ... so the index we want is the
                            // start of the patch vertex's XYZ 3-vector.
                            int patchDisplacementOffset = 3 * vertexInfluence.Vertex;
                            displacementIdx = patchStartIdx + patchDisplacementOffset;

                            displacementsStride = 1; // Each 3-vector is contiguous
                        }

                        m_DisplacementInfluences[influencesStartIdx + i] = new VertexDisplacementIndices
                        {
                            Start = displacementIdx,
                            Stride = displacementsStride,
                        };
                    }
                }
            }
        }
    }
}
