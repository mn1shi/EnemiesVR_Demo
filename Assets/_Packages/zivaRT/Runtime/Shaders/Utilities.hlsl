#pragma once
#ifndef ZIVART_COMPUTE_UTILITIES
#define ZIVART_COMPUTE_UTILITIES

//
// Decompression of packed data
//

// Unpack a compressed set of four sbytes and convert them to floats.
float4 UnpackCompressedInt8sToFloats(int sbytes)
{
    // NOTE: We left-shift then right-shift to preserve the sign extension of each value.
    return float4((float)((sbytes << 24) >> 24), (float)((sbytes << 16) >> 24),
        (float)((sbytes << 8) >> 24), (float)((sbytes << 0) >> 24));
}

// Unpack a compressed set of two shorts and convert them to floats.
float2 UnpackCompressedInt16sToFloats(int shorts)
{
    // NOTE: We left-shift then right-shift to preserve the sign extension of each value.
    return float2((float)((shorts << 16) >> 16), (float)((shorts << 0) >> 16));
}

// Unpack a compressed pair of uint16s from a 32-bit uint.
uint2 UnpackCompressedUint16s(uint ushorts)
{
    return uint2(ushorts & 0xFFFF, ushorts >> 16);
}


//
// Load compressed data blocks from buffers
//

// Retrieve a block of 2 contiguous entries from a buffer of uints-compressed-to-ushorts.
// NOTE: blockStartIdx is an index into the *uncompressed* data! We assume it's a multiple of 2,
// otherwise you won't get the result you expect!
uint2 LoadCompressedUint16s(StructuredBuffer<uint> buffer, uint bufferOffset, uint blockStartIdx)
{
    uint compressedBlock = buffer[bufferOffset + blockStartIdx / 2];
    return UnpackCompressedUint16s(compressedBlock);
}

// Retreive a block of 4 contiguous entries from a buffer of uints-compressed-to-ushorts.
// NOTE: blockStartIdx must be a multiple of 2.
uint4 Load4CompressedUint16s(StructuredBuffer<uint> buffer, uint bufferOffset, uint blockStartIdx)
{
    uint2 uncompressed0 = LoadCompressedUint16s(buffer, bufferOffset, blockStartIdx);
    uint2 uncompressed2 = LoadCompressedUint16s(buffer, bufferOffset, blockStartIdx + 2);
    return uint4(uncompressed0, uncompressed2);
}


//
// Load blocks from compressed matrices
//

// Retrieve a block of 4 entries from the same row of the given compressed matrix.
// Returns matrix[row, col:col+4], ie: 4 elements of the row starting at col.
// The matrix is column-major and compressed by packing each group of 4 columns together into 1.
// This means that this operation is loading a single 32-bit int and decompressing it.
// NOTE: col must be a multiple of 4.
// NOTE: matrixStart and matrixLD are physical offsets within the given buffer of 32-bit ints,
// whereas row and col are logical indices into the uncompressed matrix data.
float4 LoadCompressedInt8MatrixRowBlock(
    StructuredBuffer<int> buffer, uint matrixStart, uint matrixLD, uint row, uint col)
{
    uint offset = matrixLD * col / 4 + row;
    int compressed = buffer[matrixStart + offset];
    return UnpackCompressedInt8sToFloats(compressed);
}

// Retrieve a block of 2 entries from the same row of the given compressed matrix.
// Returns matrix[row, col:col+2], ie: 2 elements of the row starting at col.
// The matrix is column-major and compressed by packing each group of 2 columns together into 1.
// This means that this operation is loading a single 32-bit int and decompressing it.
// NOTE: col must be a multiple of 2.
// NOTE: matrixStart and matrixLD are physical offsets within the given buffer of 32-bit ints,
// whereas row and col are logical indices into the uncompressed matrix data.
float2 LoadCompressedInt16MatrixRowBlock(
    StructuredBuffer<int> buffer, uint matrixStart, uint matrixLD, uint row, uint col)
{
    uint offset = matrixLD * col / 2 + row;
    int compressed = buffer[matrixStart + offset];
    return UnpackCompressedInt16sToFloats(compressed);
}

// Retrieve a block of 4 entries from the same row of the given compressed matrix.
// Returns matrix[row, col:col+4], ie: 4 elements of the row starting at col.
// The matrix is column-major and compressed by packing each group of 2 columns together into 1.
// This means that this operation is loading two non-contiguous 32-bit ints and decompressing them.
// NOTE: col should be a multiple of 4.
// NOTE: matrixStart and matrixLD are physical offsets within the given buffer of 32-bit ints,
// whereas row and col are logical indices into the uncompressed matrix data.
float4 LoadCompressedInt16MatrixRowBlock4(
    StructuredBuffer<int> buffer, uint matrixStart, uint matrixLD, uint row, uint col)
{
    float2 uncompressed0 =
        LoadCompressedInt16MatrixRowBlock(buffer, matrixStart, matrixLD, row, col + 0);
    float2 uncompressed2 =
        LoadCompressedInt16MatrixRowBlock(buffer, matrixStart, matrixLD, row, col + 2);
    return float4(uncompressed0, uncompressed2);
}


//
// Helpers for loading multiple contiguous elements from a buffer.
//

// Retrieve a block of 4 contiguous entries from a buffer of floats.
float4 LoadFloat4Block(StructuredBuffer<float> buffer, uint blockStartIdx)
{
    return float4(buffer[blockStartIdx + 0], buffer[blockStartIdx + 1], buffer[blockStartIdx + 2],
        buffer[blockStartIdx + 3]);
}

float2 LoadFloat2Block(StructuredBuffer<float> buffer, uint blockStartIdx)
{
    return float2(buffer[blockStartIdx + 0], buffer[blockStartIdx + 1]);
}


//
// Horizontal sum of elements of vector
//

float HorizontalSum(float4 v)
{
    return v.x + v.y + v.z + v.w;
}

float HorizontalSum(float2 v)
{
    return v.x + v.y;
}


#endif
