#if UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX || UNITY_IOS || UNITY_TVOS
#define UNITY_APPLE
#endif

#if (UNITY_2021_2_OR_NEWER && !UNITY_APPLE) || UNITY_2022_2_OR_NEWER
using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.ZivaRTPlayer
{
    internal struct KernelFunctionsKernel
    {
        static readonly string k_Name = "KernelFunctions";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_Pose;

        ComputeBuffer m_KernelPatchMap;

        ComputeBuffer m_PatchKernelFunctionStartIndices;
        ComputeBuffer m_PatchPhiStartIndices;
        public ComputeBuffer PhiStartIndices { get { return m_PatchPhiStartIndices; } }

        ComputeBuffer m_PoseIndices;
        ComputeBuffer m_PoseShift;
        ComputeBuffer m_PoseScale;

        ComputeBuffer m_KernelScale;
        ComputeBuffer m_KernelCenters;
        ComputeBuffer m_PatchKernelCentersStrides;
        ComputeBuffer m_ScalePerKernel;

        ComputeBuffer m_Phi;
        public ComputeBuffer Phi { get { return m_Phi; } }

        public void Release()
        {
            m_Pose?.Release();
            m_KernelPatchMap?.Release();
            m_PatchKernelFunctionStartIndices?.Release();
            m_PatchPhiStartIndices?.Release();
            m_PoseIndices?.Release();
            m_PoseShift?.Release();
            m_PoseScale?.Release();
            m_KernelScale?.Release();
            m_KernelCenters?.Release();
            m_PatchKernelCentersStrides?.Release();
            m_ScalePerKernel?.Release();
            m_Phi?.Release();
        }

        // Packed set of indices to be sent to the GPU
        [StructLayout(LayoutKind.Sequential, Pack = 0)]
        struct KernelFunctionIndices
        {
            public uint PoseIndices; // Not the same as `pose` due to data compression!
            public uint Pose;
            public uint KernelCenters;
            public uint ScalePerKernel;
            public uint KernelNum; // No corresponding array, tracks total number of kernel centers

            public uint PoseScale { get { return Pose; } }
            public uint PoseShift { get { return Pose; } }
            public uint KernelScale { get { return Pose; } }

            public static KernelFunctionIndices Zero
            {
                get
                {
                    return new KernelFunctionIndices
                    {
                        PoseIndices = 0,
                        Pose = 0,
                        KernelCenters = 0,
                        ScalePerKernel = 0,
                        KernelNum = 0,
                    };
                }
            }
        }

        // A ZivaRT patch with extra data members that override its contents.
        // eg: The poseIndices array here overrides the poseIndices array inside the Patch.
        struct PatchWithOverrides
        {
            public Patch Patch;
            public uint[] PoseIndices;
            public MatrixX<int> KernelCenters;
        }

        class ConcatenatedPatchData
        {
            public KernelFunctionIndices[] PatchStarts;
            KernelFunctionIndices m_PatchTotals { get { return PatchStarts[PatchStarts.Length - 1]; } }

            public uint TotalKernels() { return m_PatchTotals.KernelNum; }

            public uint[] PoseIndices;
            public float[] PoseScale;
            public float[] PoseShift;
            public float[] KernelScale;
            public int[] KernelCenters;
            public float[] ScalePerKernel;

            public uint[] KernelCentersStrides;

            public uint[] KernelToPatchMap;

            public uint[] PhiStarts;
            public uint TotalPhiEntries { get { return PhiStarts[PhiStarts.Length - 1]; } }

            int NumPatches { get { return PhiStarts.Length - 1; } }

            void InitPatchIndices(PatchWithOverrides[] patches)
            {
                this.PatchStarts = new KernelFunctionIndices[patches.Length + 1];
                this.PhiStarts = new uint[patches.Length + 1];

                this.PatchStarts[0] = KernelFunctionIndices.Zero;
                this.PhiStarts[0] = 0;
                for (int i = 0; i < patches.Length; ++i)
                {
                    var prev = this.PatchStarts[i];
                    this.PatchStarts[i + 1] = new KernelFunctionIndices
                    {
                        PoseIndices = prev.PoseIndices + (uint)patches[i].PoseIndices.Length,
                        Pose = prev.Pose + (uint)patches[i].Patch.PoseScale.Length,
                        KernelCenters = prev.KernelCenters + (uint)patches[i].KernelCenters.Values.Length,
                        ScalePerKernel = prev.ScalePerKernel + (uint)patches[i].Patch.ScalePerKernel.Length,
                        KernelNum = prev.KernelNum + (uint)patches[i].Patch.NumKernels,
                    };

                    this.PhiStarts[i + 1] = this.PhiStarts[i] + (uint)patches[i].Patch.NumKernels + 1;
                }
            }

            public static ConcatenatedPatchData Build(PatchWithOverrides[] patches)
            {
                var data = new ConcatenatedPatchData();

                // Initialize indexing data so we know the structure of the concatentated arrays.
                data.InitPatchIndices(patches);

                // Allocate enough space for the total number of elements of each concatenated array.
                int numPatches = patches.Length;
                var totals = data.m_PatchTotals;

                data.PoseIndices = new uint[totals.PoseIndices];
                data.PoseScale = new float[totals.PoseScale];
                data.PoseShift = new float[totals.PoseShift];
                data.KernelScale = new float[totals.KernelScale];
                data.KernelCenters = new int[totals.KernelCenters];
                data.ScalePerKernel = new float[totals.ScalePerKernel];

                data.KernelCentersStrides = new uint[numPatches];

                data.KernelToPatchMap = new uint[totals.KernelNum];

                // Copy data into the concatenated arrays.
                for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx)
                {
                    var patchData = patches[patchIdx];
                    var patchStarts = data.PatchStarts[patchIdx];

                    patchData.PoseIndices.CopyTo(data.PoseIndices, patchStarts.PoseIndices);
                    patchData.Patch.PoseScale.CopyTo(data.PoseScale, patchStarts.PoseScale);
                    patchData.Patch.PoseShift.CopyTo(data.PoseShift, patchStarts.PoseShift);
                    patchData.Patch.KernelScale.CopyTo(data.KernelScale, patchStarts.KernelScale);
                    patchData.KernelCenters.Values.CopyTo(data.KernelCenters, patchStarts.KernelCenters);
                    patchData.Patch.ScalePerKernel.CopyTo(data.ScalePerKernel, patchStarts.ScalePerKernel);

                    data.KernelCentersStrides[patchIdx] = (uint)patchData.KernelCenters.LeadingDimension;

                    // Fill the block of kernelToPatchMap corresponding to the current patch.
                    int numKernels = patchData.Patch.NumKernels;
                    for (int k = 0; k < numKernels; ++k)
                    {
                        data.KernelToPatchMap[patchStarts.KernelNum + k] = (uint)patchIdx;
                    }
                }

                return data;
            }

            // Copy scalePerKernel into an array with the structure of the phi vector.
            // Importantly, this sets the constant/bias terms of the phi vector to be equal to
            // scalePerKernel, which is what they need to be for correct evaluation.
            // This function handles any differences in structure/padding between scalePerKernel
            // and phi.
            public float[] DefaultPhiValues()
            {
                var phi = new float[this.TotalPhiEntries];
                for (int patchIdx = 0; patchIdx < this.NumPatches; ++patchIdx)
                {
                    uint phiStart = this.PhiStarts[patchIdx];
                    uint phiEnd = this.PhiStarts[patchIdx + 1];
                    uint numEntries = phiEnd - phiStart;

                    var patchStarts = this.PatchStarts[patchIdx];
                    uint scalePerKernelStart = patchStarts.ScalePerKernel;

                    for (int i = 0; i < numEntries; ++i)
                    {
                        phi[phiStart + i] = this.ScalePerKernel[scalePerKernelStart + i];
                    }
                }
                return phi;
            }
        }

        static PatchWithOverrides[] OverridePatchData(ZivaRTRig rig)
        {
            var patchesWithOverrides = new PatchWithOverrides[rig.m_Patches.Length];
            for (int patchIdx = 0; patchIdx < rig.m_Patches.Length; ++patchIdx)
            {
                var patch = rig.m_Patches[patchIdx];

                // Our kernelCenters data is compressed into sbytes, but GPU code can only work
                // with 32-bit integers. We'll decompress the data into floats in the shader, but
                // to make sure everything is consistent (eg: endianness), explicitly pack the data
                // into int32s here.
                var packedKernelCenters = GPUCompressionUtils.PackMatrixRows(patch.KernelCenters);

                // Our poseIndices data is compressed into ushorts, but GPU code can only work
                // with 32-bit integers. To make sure everything is consistent, explicitly pack the
                // data into uint32s here.
                var packedPoseIndices = GPUCompressionUtils.PackUshortArray(patch.PoseIndices);

                patchesWithOverrides[patchIdx] = new PatchWithOverrides
                {
                    Patch = patch,
                    KernelCenters = packedKernelCenters,
                    PoseIndices = packedPoseIndices,
                };
            }
            return patchesWithOverrides;
        }

        public bool Init(ComputeShader zrtComputeShader, ZivaRTRig rig)
        {
            Release();

            m_ComputeShader = zrtComputeShader;
            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);

            // Reshape/reformat data as needed to work with our compute shader.
            // We store the reformatted data in "patch override" structs.
            var patchesWithOverrides = OverridePatchData(rig);

            // Concatenate together all of the data arrays we need to send to the GPU.
            var concatData = ConcatenatedPatchData.Build(patchesWithOverrides);

            // Each patch's phi-vector has numKernels+1 entries. However, we only want to spawn
            // numKernels threads, since the extra constant term does not change.
            int numPatches = rig.m_Patches.Length;
            uint totalKernels = concatData.TotalKernels();
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)totalKernels / threadGroupSize);

            // Static inputs:
            m_ComputeShader.SetInt("totalNumKernels", (int)totalKernels);

            m_KernelPatchMap = new ComputeBuffer(concatData.KernelToPatchMap.Length, sizeof(uint));
            m_KernelPatchMap.SetData(concatData.KernelToPatchMap);
            m_ComputeShader.SetBuffer(m_KernelIdx, "kernelPatchMap", m_KernelPatchMap);

            m_PatchKernelFunctionStartIndices =
                new ComputeBuffer(numPatches + 1, Marshal.SizeOf<KernelFunctionIndices>());
            m_PatchKernelFunctionStartIndices.SetData(concatData.PatchStarts);
            m_ComputeShader.SetBuffer(
                m_KernelIdx, "patchKernelFunctionStartIndices", m_PatchKernelFunctionStartIndices);

            m_PatchPhiStartIndices = new ComputeBuffer(numPatches + 1, sizeof(uint));
            m_PatchPhiStartIndices.SetData(concatData.PhiStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchPhiStartIndices", m_PatchPhiStartIndices);

            m_PoseIndices = new ComputeBuffer(concatData.PoseIndices.Length, sizeof(uint));
            m_PoseIndices.SetData(concatData.PoseIndices);
            m_ComputeShader.SetBuffer(m_KernelIdx, "poseIndices", m_PoseIndices);

            m_PoseScale = new ComputeBuffer(concatData.PoseScale.Length, sizeof(float));
            m_PoseScale.SetData(concatData.PoseScale);
            m_ComputeShader.SetBuffer(m_KernelIdx, "poseScale", m_PoseScale);

            m_PoseShift = new ComputeBuffer(concatData.PoseShift.Length, sizeof(float));
            m_PoseShift.SetData(concatData.PoseShift);
            m_ComputeShader.SetBuffer(m_KernelIdx, "poseShift", m_PoseShift);

            m_KernelScale = new ComputeBuffer(concatData.KernelScale.Length, sizeof(float));
            m_KernelScale.SetData(concatData.KernelScale);
            m_ComputeShader.SetBuffer(m_KernelIdx, "kernelScale", m_KernelScale);

            m_KernelCenters = new ComputeBuffer(concatData.KernelCenters.Length, sizeof(int));
            m_KernelCenters.SetData(concatData.KernelCenters);
            m_ComputeShader.SetBuffer(m_KernelIdx, "kernelCenters", m_KernelCenters);

            m_PatchKernelCentersStrides = new ComputeBuffer(numPatches, sizeof(uint));
            m_PatchKernelCentersStrides.SetData(concatData.KernelCentersStrides);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchKernelCentersStrides", m_PatchKernelCentersStrides);

            m_ScalePerKernel = new ComputeBuffer(concatData.ScalePerKernel.Length, sizeof(float));
            m_ScalePerKernel.SetData(concatData.ScalePerKernel);
            m_ComputeShader.SetBuffer(m_KernelIdx, "scalePerKernel", m_ScalePerKernel);

            // Outputs:
            m_Phi = new ComputeBuffer((int)concatData.TotalPhiEntries, sizeof(float));
            // This is important! Set default value of phi = scalePerKernel.
            // The constant/bias terms of phi will never change from this value, so the shader
            // doesn't set them!
            m_Phi.SetData(concatData.DefaultPhiValues());
            m_ComputeShader.SetBuffer(m_KernelIdx, "phi", m_Phi);

            // Dynamic Inputs:
            m_Pose = new ComputeBuffer(rig.m_Character.PoseVectorSize, sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "pose", m_Pose);

            return true;
        }

        public void UpdatePose(NativeArray<float> newPose)
        {
            Assert.IsTrue(m_Pose.IsValid());
            m_Pose.SetData(newPose);
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }

    struct RBFInterpolateKernel
    {
        static readonly string k_Name = "RBFInterpolate";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_RbfPatchMap;

        ComputeBuffer m_PatchRBFCoeffsStartIndices;
        ComputeBuffer m_PatchRBFOutStartIndices;

        ComputeBuffer m_RbfCoeffs;
        ComputeBuffer m_PatchRBFCoeffsStrides;
        ComputeBuffer m_ScalePerRBFCoeff;

        ComputeBuffer m_RbfOut;
        public ComputeBuffer RBFOut { get { return m_RbfOut; } }
        public ComputeBuffer RBFOutStartIndices { get { return m_PatchRBFOutStartIndices; } }

        public void Release()
        {
            m_RbfPatchMap?.Release();
            m_PatchRBFCoeffsStartIndices?.Release();
            m_PatchRBFOutStartIndices?.Release();
            m_RbfCoeffs?.Release();
            m_PatchRBFCoeffsStrides?.Release();
            m_ScalePerRBFCoeff?.Release();
            m_RbfOut?.Release();
        }

        struct PatchWithOverrides
        {
            public Patch Patch;
            public MatrixX<int> RbfCoeffs;
        }

        class ConcatenatedPatchData
        {
            public uint[] RbfCoeffsStarts;
            public uint[] RbfOutStarts;

            public uint TotalRBFOutDims { get { return RbfOutStarts[RbfOutStarts.Length - 1]; } }

            public int[] RbfCoeffs;
            public uint[] RbfCoeffsStrides;

            public float[] ScalePerRBFCoeff;

            public uint[] RbfPatchMap;

            public static ConcatenatedPatchData Build(PatchWithOverrides[] patches)
            {
                int numPatches = patches.Length;

                var rbfCoeffsStarts = new uint[numPatches + 1];
                var rbfOutStarts = new uint[numPatches + 1];

                // Calculate start indices with prefix sum
                rbfCoeffsStarts[0] = 0;
                rbfOutStarts[0] = 0;
                for (int i = 0; i < numPatches; ++i)
                {
                    rbfCoeffsStarts[i + 1] = rbfCoeffsStarts[i] + (uint)patches[i].RbfCoeffs.Values.Length;
                    int numRBFOutDims = patches[i].Patch.RbfCoeffs.Rows;
                    rbfOutStarts[i + 1] = rbfOutStarts[i] + (uint)numRBFOutDims;
                }

                // Allocate space for concatenated arrays
                uint totalRBFCoeffsSize = rbfCoeffsStarts[numPatches];
                var rbfCoeffs = new int[totalRBFCoeffsSize];
                var rbfCoeffsStrides = new uint[numPatches];

                uint totalRBFOutSize = rbfOutStarts[numPatches];
                var rbfPatchMap = new uint[totalRBFOutSize];
                var scalePerRBFCoeff = new float[totalRBFOutSize];

                // Copy data into concatenated arrays
                for (int i = 0; i < numPatches; ++i)
                {
                    patches[i].RbfCoeffs.Values.CopyTo(rbfCoeffs, rbfCoeffsStarts[i]);
                    rbfCoeffsStrides[i] = (uint)patches[i].RbfCoeffs.LeadingDimension;
                    Assert.AreEqual(rbfCoeffsStrides[i], patches[i].RbfCoeffs.Rows);

                    uint rbfOutStart = rbfOutStarts[i];
                    uint rbfOutEnd = rbfOutStarts[i + 1];

                    uint numRBFOutDims = rbfOutEnd - rbfOutStart;
                    Assert.AreEqual(numRBFOutDims, patches[i].Patch.RbfCoeffs.Rows);
                    // NOTE: Eliminate any padding at the end of scalePerRBFCoeff to ensure correct indexing!
                    Array.Copy(patches[i].Patch.ScalePerRBFCoeff, 0, scalePerRBFCoeff, rbfOutStart, numRBFOutDims);

                    // Fill block of rbfPatchMap to point to current patch:
                    for (uint d = rbfOutStart; d < rbfOutEnd; ++d)
                    {
                        rbfPatchMap[d] = (uint)i;
                    }
                }

                return new ConcatenatedPatchData
                {
                    RbfCoeffsStarts = rbfCoeffsStarts,
                    RbfOutStarts = rbfOutStarts,
                    RbfCoeffs = rbfCoeffs,
                    RbfCoeffsStrides = rbfCoeffsStrides,
                    ScalePerRBFCoeff = scalePerRBFCoeff,
                    RbfPatchMap = rbfPatchMap,
                };
            }
        }

        static PatchWithOverrides[] OverridePatchData(ZivaRTRig rig)
        {
            var patchesWithOverrides = new PatchWithOverrides[rig.m_Patches.Length];
            for (int patchIdx = 0; patchIdx < rig.m_Patches.Length; ++patchIdx)
            {
                var patch = rig.m_Patches[patchIdx];

                // rbfCoeffs is compressed into 16-bit shorts, but GPU code can only work with
                // 32-bit integers. We'll decompress the data into floats in the shader, but to
                // make sure everything is consistent (eg: endianness), explicitly pack the
                // data into int32s here.
                var packedRBFCoeffs = GPUCompressionUtils.PackMatrixRows(patch.RbfCoeffs);

                patchesWithOverrides[patchIdx] = new PatchWithOverrides
                {
                    Patch = patch,
                    RbfCoeffs = packedRBFCoeffs,
                };
            }
            return patchesWithOverrides;
        }

        public bool Init(
            ComputeShader zrtComputeShader,
            ZivaRTRig rig,
            ComputeBuffer phiIn,
            ComputeBuffer patchPhiInStartIndices)
        {
            Release();

            m_ComputeShader = zrtComputeShader;
            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);

            int numPatches = rig.m_Patches.Length;

            // Reformat/restructure patch data as needed for the GPU.
            var patchesWithOverrides = OverridePatchData(rig);
            // Concatenate data together into single arrays to be sent to the GPU.
            var concatData = ConcatenatedPatchData.Build(patchesWithOverrides);

            uint totalRBFOutDims = concatData.TotalRBFOutDims;
            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)totalRBFOutDims / threadGroupSize);

            // Inputs:
            m_ComputeShader.SetInt("totalNumRBFOutDims", (int)totalRBFOutDims);

            Assert.AreEqual(concatData.RbfPatchMap.Length, totalRBFOutDims);
            m_RbfPatchMap = new ComputeBuffer((int)totalRBFOutDims, sizeof(uint));
            m_RbfPatchMap.SetData(concatData.RbfPatchMap);
            m_ComputeShader.SetBuffer(m_KernelIdx, "rbfPatchMap", m_RbfPatchMap);

            m_PatchRBFCoeffsStartIndices = new ComputeBuffer(numPatches + 1, sizeof(uint));
            m_PatchRBFCoeffsStartIndices.SetData(concatData.RbfCoeffsStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchRBFCoeffsStartIndices", m_PatchRBFCoeffsStartIndices);

            m_PatchRBFOutStartIndices = new ComputeBuffer(numPatches + 1, sizeof(uint));
            m_PatchRBFOutStartIndices.SetData(concatData.RbfOutStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchRBFOutStartIndices", m_PatchRBFOutStartIndices);

            m_RbfCoeffs = new ComputeBuffer(concatData.RbfCoeffs.Length, sizeof(int));
            m_RbfCoeffs.SetData(concatData.RbfCoeffs);
            m_ComputeShader.SetBuffer(m_KernelIdx, "rbfCoeffs", m_RbfCoeffs);

            m_PatchRBFCoeffsStrides = new ComputeBuffer(numPatches, sizeof(uint));
            m_PatchRBFCoeffsStrides.SetData(concatData.RbfCoeffsStrides);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchRBFCoeffsStrides", m_PatchRBFCoeffsStrides);

            Assert.AreEqual(concatData.ScalePerRBFCoeff.Length, totalRBFOutDims);
            m_ScalePerRBFCoeff = new ComputeBuffer((int)totalRBFOutDims, sizeof(float));
            m_ScalePerRBFCoeff.SetData(concatData.ScalePerRBFCoeff);
            m_ComputeShader.SetBuffer(m_KernelIdx, "scalePerRBFCoeff", m_ScalePerRBFCoeff);

            // Output
            m_RbfOut = new ComputeBuffer((int)totalRBFOutDims, sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "rbfOut", m_RbfOut);

            // Input buffers shared with other kernels:
            Assert.AreEqual(patchPhiInStartIndices.count, numPatches + 1);
            Assert.AreEqual(patchPhiInStartIndices.stride, sizeof(uint));
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchPhiInStartIndices", patchPhiInStartIndices);

            // phiIn.count should be >= patchPhiInStartIndices[-1] here.
            Assert.AreEqual(phiIn.stride, sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "phiIn", phiIn);

            return true;
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }

    struct TensorSkinningKernel
    {
        static readonly string k_Name = "TensorSkinning";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_DisplacementPatchMap;
        ComputeBuffer m_PatchTSStartIndices;

        ComputeBuffer m_ReducedBasis;
        ComputeBuffer m_PatchReducedBasisStrides;
        ComputeBuffer m_ScalePerVertex;

        ComputeBuffer m_DisplacementsOut;
        public ComputeBuffer Displacements { get { return m_DisplacementsOut; } }

        public void Release()
        {
            m_DisplacementPatchMap?.Release();
            m_PatchTSStartIndices?.Release();
            m_ReducedBasis?.Release();
            m_PatchReducedBasisStrides?.Release();
            m_ScalePerVertex?.Release();
            m_DisplacementsOut?.Release();
        }

        // Packed set of indices to be sent to the GPU
        [StructLayout(LayoutKind.Sequential, Pack = 0)]
        struct TensorSkinningIndices
        {
            public uint ReducedBasis;
            public uint Displacements;

            public static TensorSkinningIndices Zero
            {
                get
                {
                    return new TensorSkinningIndices
                    {
                        ReducedBasis = 0,
                        Displacements = 0,
                    };
                }
            }
        }

        struct PatchWithOverrides
        {
            public Patch Patch;
            public MatrixX<int> ReducedBasis;
        }

        class ConcatenatedPatchData
        {
            public TensorSkinningIndices[] PatchStarts;

            public uint[] SubspaceCoeffsStarts; // For validation/error-checking

            public int[] ReducedBasis;
            public float[] ScalePerVertex;

            public uint[] ReducedBasisStrides;
            public uint[] DisplacementPatchMap;

            public TensorSkinningIndices Totals { get { return PatchStarts[PatchStarts.Length - 1]; } }
            public uint TotalSubspaceCoeffs { get { return SubspaceCoeffsStarts[SubspaceCoeffsStarts.Length - 1]; } }

            public static ConcatenatedPatchData Build(PatchWithOverrides[] patches)
            {
                int numPatches = patches.Length;

                // Build start indices as prefix sum
                var patchStarts = new TensorSkinningIndices[numPatches + 1];
                var subspaceCoeffsStarts = new uint[numPatches + 1];
                patchStarts[0] = TensorSkinningIndices.Zero;
                subspaceCoeffsStarts[0] = 0;
                for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx)
                {
                    var patch = patches[patchIdx];
                    var prev = patchStarts[patchIdx];
                    patchStarts[patchIdx + 1] = new TensorSkinningIndices
                    {
                        ReducedBasis = prev.ReducedBasis + (uint)patch.ReducedBasis.Values.Length,
                        Displacements = prev.Displacements + (uint)patch.Patch.ReducedBasis.Rows,
                    };
                    subspaceCoeffsStarts[patchIdx + 1] =
                        subspaceCoeffsStarts[patchIdx] + (uint)patch.Patch.ReducedBasis.Cols;
                }

                // Allocate enough space for the concatenated arrays.
                var totals = patchStarts[numPatches];
                var reducedBasis = new int[totals.ReducedBasis];
                var scalePerVertex = new float[totals.Displacements];
                var reducedBasisStrides = new uint[numPatches];
                var displacementPatchMap = new uint[totals.Displacements];

                // Copy data into concatenated arrays.
                for (int patchIdx = 0; patchIdx < numPatches; ++patchIdx)
                {
                    var starts = patchStarts[patchIdx];
                    var ends = patchStarts[patchIdx + 1];
                    var patch = patches[patchIdx];

                    patch.ReducedBasis.Values.CopyTo(reducedBasis, starts.ReducedBasis);

                    // Make sure to eliminate any padding in the scalePerVertex data:
                    uint displacementsSize = ends.Displacements - starts.Displacements;
                    Array.Copy(
                        patch.Patch.ScalePerVertex, 0, scalePerVertex, starts.Displacements, displacementsSize);

                    reducedBasisStrides[patchIdx] = (uint)patch.ReducedBasis.LeadingDimension;

                    // Fill block of displacementPatchMap to point to current patch:
                    for (uint d = starts.Displacements; d < ends.Displacements; ++d)
                    {
                        displacementPatchMap[d] = (uint)patchIdx;
                    }
                }

                return new ConcatenatedPatchData
                {
                    PatchStarts = patchStarts,
                    SubspaceCoeffsStarts = subspaceCoeffsStarts,
                    ReducedBasis = reducedBasis,
                    ScalePerVertex = scalePerVertex,
                    ReducedBasisStrides = reducedBasisStrides,
                    DisplacementPatchMap = displacementPatchMap,
                };
            }
        }

        static PatchWithOverrides[] OverridePatchData(Patch[] patches)
        {
            var patchesWithOverrides = new PatchWithOverrides[patches.Length];
            for (int patchIdx = 0; patchIdx < patches.Length; ++patchIdx)
            {
                var patch = patches[patchIdx];

                // Our reducedBasis data is compressed into sbytes, but GPU code can only work
                // with 32-bit integers.
                var packedReducedBasis = GPUCompressionUtils.PackMatrixRows(patch.ReducedBasis);

                patchesWithOverrides[patchIdx] = new PatchWithOverrides
                {
                    Patch = patch,
                    ReducedBasis = packedReducedBasis,
                };
            }
            return patchesWithOverrides;
        }

        static uint[] ExtractDisplacementsIndices(TensorSkinningIndices[] indices)
        {
            var displacements = new uint[indices.Length];
            for (int i = 0; i < indices.Length; ++i)
            {
                displacements[i] = indices[i].Displacements;
            }
            return displacements;
        }

        public bool Init(
            ComputeShader zrtComputeShader,
            ZivaRTRig rig,
            ComputeBuffer inSubspaceCoeffs,
            ComputeBuffer inSubspaceStartIndices,
            out uint[] displacementsStartIndices)
        {
            Release();

            displacementsStartIndices = null;

            m_ComputeShader = zrtComputeShader;
            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);

            var patchOverrides = OverridePatchData(rig.m_Patches);

            var concatData = ConcatenatedPatchData.Build(patchOverrides);

            int numPatches = rig.m_Patches.Length;
            uint totalNumDisplacements = concatData.Totals.Displacements;

            m_ComputeShader.GetKernelThreadGroupSizes(
                m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)totalNumDisplacements / threadGroupSize);

            // Inputs:
            m_ComputeShader.SetInt("totalDisplacements", (int)totalNumDisplacements);

            m_DisplacementPatchMap = new ComputeBuffer((int)totalNumDisplacements, sizeof(uint));
            m_DisplacementPatchMap.SetData(concatData.DisplacementPatchMap);
            m_ComputeShader.SetBuffer(m_KernelIdx, "displacementPatchMap", m_DisplacementPatchMap);

            m_PatchTSStartIndices = new ComputeBuffer(numPatches + 1, Marshal.SizeOf<TensorSkinningIndices>());
            m_PatchTSStartIndices.SetData(concatData.PatchStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchTSStartIndices", m_PatchTSStartIndices);

            m_ReducedBasis = new ComputeBuffer(concatData.ReducedBasis.Length, sizeof(int));
            m_ReducedBasis.SetData(concatData.ReducedBasis);
            m_ComputeShader.SetBuffer(m_KernelIdx, "reducedBasis", m_ReducedBasis);

            m_PatchReducedBasisStrides = new ComputeBuffer(numPatches, sizeof(uint));
            m_PatchReducedBasisStrides.SetData(concatData.ReducedBasisStrides);
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchReducedBasisStrides", m_PatchReducedBasisStrides);

            m_ScalePerVertex = new ComputeBuffer(concatData.ScalePerVertex.Length, sizeof(float));
            m_ScalePerVertex.SetData(concatData.ScalePerVertex);
            m_ComputeShader.SetBuffer(m_KernelIdx, "scalePerVertex", m_ScalePerVertex);

            // Outputs:
            m_DisplacementsOut = new ComputeBuffer((int)concatData.Totals.Displacements, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "displacementsOut", m_DisplacementsOut);

            displacementsStartIndices = ExtractDisplacementsIndices(concatData.PatchStarts);

            // Inputs from other kernels:

            // NOTE: This buffer may have been created as a float buffer, we are reinterpreting
            // here as a buffer of float3s.
            Assert.AreEqual(
                inSubspaceCoeffs.count * inSubspaceCoeffs.stride / sizeof(float), concatData.TotalSubspaceCoeffs * 3);
            m_ComputeShader.SetBuffer(m_KernelIdx, "subspaceCoeffs", inSubspaceCoeffs);

            Assert.AreEqual(inSubspaceStartIndices.count, numPatches + 1);
            Assert.AreEqual(inSubspaceStartIndices.stride, sizeof(uint));
            m_ComputeShader.SetBuffer(m_KernelIdx, "patchSubspaceStartIndices", inSubspaceStartIndices);

            return true;
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }

    struct AccumulateDisplacementsKernel
    {
        static readonly string k_Name = "AccumulateDisplacements";

        ComputeShader m_ComputeShader;
        int m_KernelIdx;
        int m_NumWorkGroups;

        ComputeBuffer m_DisplacementInfluences;
        ComputeBuffer m_InfluencesStarts;

        public void Release()
        {
            m_DisplacementInfluences?.Release();
            m_InfluencesStarts?.Release();
        }

        class ConcatenatedInfluences
        {
            public uint[] Influences;
            public uint[] InfluencesStarts;

            public static ConcatenatedInfluences Build(
                ZivaRTRig rig,
                uint[] patchDisplacementStarts)
            {
                var vertexInfluenceLists = PatchInfluences.InvertPatchVertexMap(rig);

                var influencesStarts = PatchInfluences.CalcInfluenceStartIndices(vertexInfluenceLists);

                // Build the concatenated array of displacement influences for each vertex.
                var numShapeVertices = vertexInfluenceLists.Length;
                int totalDisplacementInfluences = influencesStarts[numShapeVertices];
                var displacementInfluences = new uint[totalDisplacementInfluences];
                for (int shapeVertex = 0; shapeVertex < numShapeVertices; ++shapeVertex)
                {
                    int influencesStartIdx = influencesStarts[shapeVertex];
                    var vertexInfluences = vertexInfluenceLists[shapeVertex];
                    for (int i = 0; i < vertexInfluences.Count; ++i)
                    {
                        var vertexInfluence = vertexInfluences[i];

                        uint patchStartIdx = patchDisplacementStarts[vertexInfluence.Patch];

                        // Displacements data is stored as float3
                        uint displacementIdx = patchStartIdx + (uint)vertexInfluence.Vertex;
                        displacementInfluences[influencesStartIdx + i] = displacementIdx;
                    }
                }

                return new ConcatenatedInfluences
                {
                    Influences = displacementInfluences,
                    InfluencesStarts = Array.ConvertAll(influencesStarts, i => (uint)i),
                };
            }
        }

        public bool Init(
            ComputeShader zrtComputeShader,
            ZivaRTRig rig,
            uint[] patchDisplacementStarts,
            ComputeBuffer inDisplacments,
            ComputeBuffer inOutPositions)
        {
            Release();

            int numVertices = rig.m_Character.NumVertices;

            m_ComputeShader = zrtComputeShader;
            m_KernelIdx = m_ComputeShader.FindKernel(k_Name);
            m_ComputeShader.GetKernelThreadGroupSizes(m_KernelIdx, out uint threadGroupSize, out uint _, out uint _);
            m_NumWorkGroups = (int)Math.Ceiling((float)numVertices / threadGroupSize);

            // Build concatenated array of influences for each vertex:
            var concatData = ConcatenatedInfluences.Build(rig, patchDisplacementStarts);

            // Constant inputs to skinning shader:
            m_ComputeShader.SetInt("totalNumVertices", numVertices);

            m_DisplacementInfluences = new ComputeBuffer(concatData.Influences.Length, sizeof(uint));
            m_DisplacementInfluences.SetData(concatData.Influences);
            m_ComputeShader.SetBuffer(m_KernelIdx, "displacementInfluences", m_DisplacementInfluences);

            m_InfluencesStarts = new ComputeBuffer(numVertices + 1, sizeof(uint));
            m_InfluencesStarts.SetData(concatData.InfluencesStarts);
            m_ComputeShader.SetBuffer(m_KernelIdx, "influencesStarts", m_InfluencesStarts);

            // Inputs from other kernels:
            Assert.AreEqual(inDisplacments.stride, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "displacements", inDisplacments);

            Assert.AreEqual(inOutPositions.count, numVertices);
            Assert.AreEqual(inOutPositions.stride, 3 * sizeof(float));
            m_ComputeShader.SetBuffer(m_KernelIdx, "positions", inOutPositions);

            return true;
        }

        public void Dispatch() { m_ComputeShader.Dispatch(m_KernelIdx, m_NumWorkGroups, 1, 1); }
    }

    struct CorrectivesKernels
    {
        KernelFunctionsKernel m_KernelFunctionsKernel;
        RBFInterpolateKernel m_RbfInterpolateKernel;
        TensorSkinningKernel m_TensorSkinningKernel;
        AccumulateDisplacementsKernel m_AccumulateDisplacementsKernel;

        public void Release()
        {
            m_KernelFunctionsKernel.Release();
            m_RbfInterpolateKernel.Release();
            m_TensorSkinningKernel.Release();
            m_AccumulateDisplacementsKernel.Release();
        }

        public bool Init(ZivaRTRig rig, ComputeBuffer dstPositionBuffer, ZivaShaderData shaderData)
        {
            var computeShaderResource = shaderData.m_ZivaRT;

            if (computeShaderResource == null)
            {
                Debug.LogError("Could not load ZivaRT.compute shader");
                return false;
            }

            var computeShader = UnityEngine.Object.Instantiate(computeShaderResource);

            bool success;

            success = m_KernelFunctionsKernel.Init(computeShader, rig);
            if (!success)
                return false;

            success = m_RbfInterpolateKernel.Init(
                computeShader, rig, m_KernelFunctionsKernel.Phi, m_KernelFunctionsKernel.PhiStartIndices);
            if (!success)
                return false;

            success = m_TensorSkinningKernel.Init(
                computeShader, rig, m_RbfInterpolateKernel.RBFOut, m_RbfInterpolateKernel.RBFOutStartIndices,
                out var displacementsStartIndices);
            if (!success)
                return false;

            success = m_AccumulateDisplacementsKernel.Init(
                computeShader, rig, displacementsStartIndices, m_TensorSkinningKernel.Displacements,
                dstPositionBuffer);
            if (!success)
                return false;

            return true;
        }

        public void Dispatch(NativeArray<float> currentPose)
        {
            Profiler.BeginSample("DispatchCorrectives");

            // Send pose to GPU.
            Profiler.BeginSample("UpdatePose");
            m_KernelFunctionsKernel.UpdatePose(currentPose);
            Profiler.EndSample();

            // Dispatch kernels to compute correctives.
            m_KernelFunctionsKernel.Dispatch();
            m_RbfInterpolateKernel.Dispatch();
            m_TensorSkinningKernel.Dispatch();
            m_AccumulateDisplacementsKernel.Dispatch();

            Profiler.EndSample();
        }
    }
}
#endif
