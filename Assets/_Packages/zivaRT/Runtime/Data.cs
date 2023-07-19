using System;
using UnityEngine.Assertions;

namespace Unity.ZivaRTPlayer
{
    [Serializable]
    internal struct CharacterComponent
    {
        public float[] RestShape;
        public float[] RestExtraParameters;
        public float[] RestLocalTransforms;
        public float[] RestWorldTransforms;
        public string[] JointNames;

        public int NumJoints { get { return JointNames.Length; } }
        public int NumExtraParameters { get { return RestExtraParameters.Length; } }
        public int PoseVectorSize { get { return NumExtraParameters + NumJoints * 12; } }
        public int NumVertices { get { return RestShape.Length / 3; } }
    }

    [Serializable]
    internal struct MatrixX<T>
    {
        public T[] Values;
        public int Rows;
        public int Cols;

        public int LeadingDimension { get { return Cols > 0 ? Values.Length / Cols : 0; } }
        public T Get(int row, int col)
        {
            Assert.IsTrue(row < Rows);
            Assert.IsTrue(col < Cols);
            return Values[col * LeadingDimension + row];
        }
        public void Set(int row, int col, T value)
        {
            Assert.IsTrue(row < Rows);
            Assert.IsTrue(col < Cols);
            Values[col * LeadingDimension + row] = value;
        }
    }

    [Serializable]
    internal enum CorrectiveType
    {
        TensorSkin,
        EigenSkin,
        FullSpace
    }

    [Serializable]
    internal class Patch
    {
        public uint[] Vertices;

        public ushort[] PoseIndices;
        public float[] PoseShift;
        public float[] PoseScale;
        public float[] KernelScale;
        public MatrixX<sbyte> KernelCenters;
        public float[] ScalePerKernel;

        public MatrixX<short> RbfCoeffs;
        public float[] ScalePerRBFCoeff;

        public MatrixX<sbyte> ReducedBasis;
        public float[] ScalePerVertex;

        public int NumKernels { get { return KernelCenters.Rows; } }
        public int PoseDimensions { get { return KernelCenters.Cols; } }

        public bool hasZeroKernelCenters() { return KernelCenters.Rows == 0; }
    }

    [Serializable]
    internal struct SparseMatrix
    {
        public int NumCols;
        public int NumRows;
        public int[] RowIndices;
        public int[] ColStarts;
        public float[] W;
    }

    [Serializable]
    internal struct Skinning
    {
        public float[] RestPoseInverse;
        public SparseMatrix SkinningWeights;
    }
}
