// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace ZivaRT
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

internal struct MatrixXi8 : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static MatrixXi8 GetRootAsMatrixXi8(ByteBuffer _bb) { return GetRootAsMatrixXi8(_bb, new MatrixXi8()); }
  public static MatrixXi8 GetRootAsMatrixXi8(ByteBuffer _bb, MatrixXi8 obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public MatrixXi8 __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public sbyte X(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetSbyte(__p.__vector(o) + j * 1) : (sbyte)0; }
  public int XLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<sbyte> GetXBytes() { return __p.__vector_as_span<sbyte>(4, 1); }
#else
  public ArraySegment<byte>? GetXBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public sbyte[] GetXArray() { return __p.__vector_as_array<sbyte>(4); }
  public int Rows { get { int o = __p.__offset(6); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }
  public int Cols { get { int o = __p.__offset(8); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }

  public static Offset<ZivaRT.MatrixXi8> CreateMatrixXi8(FlatBufferBuilder builder,
      VectorOffset xOffset = default(VectorOffset),
      int rows = 0,
      int cols = 0) {
    builder.StartTable(3);
    MatrixXi8.AddCols(builder, cols);
    MatrixXi8.AddRows(builder, rows);
    MatrixXi8.AddX(builder, xOffset);
    return MatrixXi8.EndMatrixXi8(builder);
  }

  public static void StartMatrixXi8(FlatBufferBuilder builder) { builder.StartTable(3); }
  public static void AddX(FlatBufferBuilder builder, VectorOffset xOffset) { builder.AddOffset(0, xOffset.Value, 0); }
  public static VectorOffset CreateXVector(FlatBufferBuilder builder, sbyte[] data) { builder.StartVector(1, data.Length, 1); for (int i = data.Length - 1; i >= 0; i--) builder.AddSbyte(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateXVectorBlock(FlatBufferBuilder builder, sbyte[] data) { builder.StartVector(1, data.Length, 1); builder.Add(data); return builder.EndVector(); }
  public static void StartXVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(1, numElems, 1); }
  public static void AddRows(FlatBufferBuilder builder, int rows) { builder.AddInt(1, rows, 0); }
  public static void AddCols(FlatBufferBuilder builder, int cols) { builder.AddInt(2, cols, 0); }
  public static Offset<ZivaRT.MatrixXi8> EndMatrixXi8(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<ZivaRT.MatrixXi8>(o);
  }
};


}
