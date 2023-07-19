namespace Unity.ZivaRTPlayer
{
    internal class GPUCompressionUtils
    {
        public static int PackSBytes(sbyte byte0, sbyte byte1, sbyte byte2, sbyte byte3)
        {
            // Pack the 4 values into a 32-bit integer.
            // CAUTION! These are signed values, so we have to be careful to remove
            // the sign-extension bits when we expand from 8-bit to 32-bit int.
            int packed = ((int)byte0) & 0xFF;
            packed |= (((int)byte1) & 0xFF) << 8;
            packed |= (((int)byte2) & 0xFF) << 16;
            packed |= (((int)byte3) & 0xFF) << 24;
            return packed;
        }

        public static int PackShorts(short short0, short short1)
        {
            int packed = ((int)short0) & 0xFFFF;
            packed |= (((int)short1) & 0xFFFF) << 16;
            return packed;
        }

        public static uint PackUShorts(ushort short0, ushort short1)
        {
            uint packed = short0;
            packed |= (uint)short1 << 16;
            return packed;
        }

        public static uint[] PackUshortArray(ushort[] shortIndices)
        {
            int numPackedIndices = (shortIndices.Length + 1) / 2;
            var packedIndices = new uint[numPackedIndices];
            for (int i = 0; i < shortIndices.Length; i += 2)
            {
                ushort s0 = shortIndices[i];
                ushort s1 = i + 1 < shortIndices.Length ? shortIndices[i + 1] : (ushort)0;

                packedIndices[i / 2] = PackUShorts(s0, s1);
            }
            return packedIndices;
        }

        // Pack each group of 4 values along each row into a 32-bit int in the result.
        // Pad the number of columns before compression so it's a multiple of 4.
        // The end result is a compressed matrix of size [rows x ceil(cols/4)].
        // NOTE: Does not preserve any padding in the leading dimension!
        public static MatrixX<int> PackMatrixRows(MatrixX<sbyte> byteMatrix)
        {
            int numPackedRows = byteMatrix.Rows;
            int numPackedCols = (byteMatrix.Cols + 3) / 4; // cols/4 rounded up
            var packedMatrix = new MatrixX<int>
            {
                Rows = numPackedRows,
                Cols = numPackedCols,
                Values = new int[numPackedRows * numPackedCols]
            };

            for (int packedCol = 0; packedCol < numPackedCols; ++packedCol)
            {
                for (int row = 0; row < byteMatrix.Rows; ++row)
                {
                    int origCol = packedCol * 4;

                    // Load 4 values from the current row
                    sbyte byte0 = byteMatrix.Get(row, origCol + 0);
                    sbyte byte1 = 0; // Pad with zeros if we go out of bounds
                    sbyte byte2 = 0;
                    sbyte byte3 = 0;
                    if (origCol + 1 < byteMatrix.Cols) byte1 = byteMatrix.Get(row, origCol + 1);
                    if (origCol + 2 < byteMatrix.Cols) byte2 = byteMatrix.Get(row, origCol + 2);
                    if (origCol + 3 < byteMatrix.Cols) byte3 = byteMatrix.Get(row, origCol + 3);

                    int packedValue = PackSBytes(byte0, byte1, byte2, byte3);
                    packedMatrix.Set(row, packedCol, packedValue);
                }
            }

            return packedMatrix;
        }

        // Pack each group of 2 values along each row into a 32-bit int in the result.
        // Pad the number of columns before compression so it's a multiple of 2.
        // The end result is a compressed matrix of size [rows x ceil(cols/2)].
        // NOTE: Does not preserve any padding in the leading dimension!
        public static MatrixX<int> PackMatrixRows(MatrixX<short> shortMatrix)
        {
            int numPackedRows = shortMatrix.Rows;
            int numPackedCols = (shortMatrix.Cols + 1) / 2; // cols/2 rounded up
            var packedMatrix = new MatrixX<int>
            {
                Rows = numPackedRows,
                Cols = numPackedCols,
                Values = new int[numPackedRows * numPackedCols],
            };

            for (int packedCol = 0; packedCol < numPackedCols; ++packedCol)
            {
                for (int row = 0; row < shortMatrix.Rows; ++row)
                {
                    int origCol = packedCol * 2;

                    // Load 2 values from current row
                    short short0 = shortMatrix.Get(row, origCol + 0);
                    short short1 = 0;
                    if (origCol + 1 < shortMatrix.Cols) short1 = shortMatrix.Get(row, origCol + 1);

                    int packedValue = PackShorts(short0, short1);
                    packedMatrix.Set(row, packedCol, packedValue);
                }
            }

            return packedMatrix;
        }
    }
}
