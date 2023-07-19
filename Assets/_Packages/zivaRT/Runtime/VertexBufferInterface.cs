#define MOTION_VECTORS
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace Unity.ZivaRTPlayer
{
    // Class for writing to a Mesh's vertex buffer when using the Burst Jobs solver
    internal class VertexBufferInterface
    {
        // Simple struct for accessing an interleaved data stream
        public struct DataStream
        {
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<byte> Data;
            public int Offset;
            public int Stride;
            public unsafe T* GetPtrAtIndex<T>(int index) where T : unmanaged
            {
                byte* pData = (byte*)NativeArrayUnsafeUtility.GetUnsafePtr(Data);
                return (T*)(pData + Offset + index * Stride);
            }
        }

        // Graphics buffer and data representing a single vertex buffer in a mesh
        class GraphicsBufferDataPair
        {
            public NativeArray<byte> Data;
            public GraphicsBuffer Buffer;

            public GraphicsBufferDataPair(NativeArray<byte> Data, GraphicsBuffer Buffer)
            {
                this.Data = Data;
                this.Buffer = Buffer;
            }

            public void DisposeNativeData()
            {
                Data.Dispose();
                Buffer.Dispose();
            }

            public void SetData()
            {
                Buffer.SetData(Data);
            }
        }

        public enum AttributeStreams
        {
            Position = 0,
            NormalTangentMovecs = 1,
            TotalStreams = 2
        }

        // Graphics buffers and their datas keyed by the vertex stream index
        GraphicsBufferDataPair[] m_StreamMap = new GraphicsBufferDataPair[(int)AttributeStreams.TotalStreams];

        // The four streams that ZivaRT can be requested to update.
        public DataStream PositionsStreamData { get { return m_PositionsStreamData; } }
        DataStream m_PositionsStreamData = new DataStream();
        public DataStream NormalsStreamData { get { return m_NormalsStreamData; } }
        DataStream m_NormalsStreamData = new DataStream();
        public DataStream TangentsStreamData { get { return m_TangentsStreamData; } }
        DataStream m_TangentsStreamData = new DataStream();
        public DataStream MovecsStreamData { get { return m_MovecsStreamData; } }
        DataStream m_MovecsStreamData = new DataStream();

        void UpdateStreamMap(VertexAttribute vertexAttribute, Mesh.MeshData meshData, Mesh targetMesh)
        {
            int streamIndex = meshData.GetVertexAttributeStream(vertexAttribute);
            if (m_StreamMap[streamIndex] == null)
            {
                GraphicsBufferDataPair graphicsBufferDataPair = new GraphicsBufferDataPair
                    (new NativeArray<byte>(meshData.GetVertexData<byte>(streamIndex), Allocator.Persistent),
                    targetMesh.GetVertexBuffer(streamIndex));   // this returns a graphics buffer that
                                                                // wraps the vertex buffer at streamIndex
                m_StreamMap[streamIndex] = graphicsBufferDataPair;
            }
        }

        // setup an interleaved data stream per attribute
        void UpdateDataStream(ref DataStream dataStream, Mesh.MeshData meshData, VertexAttribute vertexAttribute)
        {
            int streamIndex = meshData.GetVertexAttributeStream(vertexAttribute);
            dataStream.Data = m_StreamMap[streamIndex].Data;    // this could be shared between attributes in the same
                                                                // vertex stream
            dataStream.Offset = meshData.GetVertexAttributeOffset(vertexAttribute);
            dataStream.Stride = meshData.GetVertexBufferStride(streamIndex);
        }

        class DescriptorComparer : IComparer<VertexAttributeDescriptor>
        {
            public int Compare(VertexAttributeDescriptor d1, VertexAttributeDescriptor d2)
            {
                if (d1.attribute < d2.attribute)
                    return -1;
                else
                    return 1;
            }
        }

        internal static void CombineAttributes(ref VertexAttributeDescriptor[] finalAttributes,
            VertexAttributeDescriptor[] currentAttributes, VertexAttributeDescriptor[] newAttributes,
            int newAttributesCount)
        {
            // if any attribute didn't exist in source mesh and needs to be added
            if (newAttributesCount > 0)
            {
                // combine the old and new attributes
                finalAttributes = new VertexAttributeDescriptor[newAttributesCount + currentAttributes.Length];
                //copy existing ones
                currentAttributes.CopyTo(finalAttributes, 0);
                // add the new ones
                Array.Copy(newAttributes, 0, finalAttributes, currentAttributes.Length, newAttributesCount);
            }

            // need to sort, because we get a warning if attributes are not in ascending order
            Array.Sort(finalAttributes, new DescriptorComparer());
        }

        internal static int SetupStream(VertexAttributeDescriptor[] newAttributes,Mesh targetMesh,
            RecomputeTangentFrames recomputeTangentFrames, bool calculateMotionVectors,
            int stream)
        {
            int newAttributesCount = 0;
            // if we don't have a normal attribute, but are asked to calculate normals then create it
            if (recomputeTangentFrames != RecomputeTangentFrames.None)
            {
                if (!targetMesh.HasVertexAttribute(VertexAttribute.Normal))
                {
                    newAttributes[newAttributesCount++] =
                        new VertexAttributeDescriptor(VertexAttribute.Normal, VertexAttributeFormat.Float32,
                        3, stream);
                }
                // if we don't have a tangent attribute, but are asked to calculate tangents then create it
                if (recomputeTangentFrames == RecomputeTangentFrames.NormalsAndTangents)
                {
                    if (!targetMesh.HasVertexAttribute(VertexAttribute.Tangent))
                    {
                        newAttributes[newAttributesCount++] =
                            new VertexAttributeDescriptor(VertexAttribute.Tangent, VertexAttributeFormat.Float32,
                            4, stream);
                    }
                }
            }
#if MOTION_VECTORS
            // if we don't have a motion vector attribute, but are asked to calculate motion vectors then create it
            if (calculateMotionVectors && !targetMesh.HasVertexAttribute(VertexAttribute.TexCoord5))
            {
                newAttributes[newAttributesCount++] =
                    new VertexAttributeDescriptor(VertexAttribute.TexCoord5, VertexAttributeFormat.Float32,
                    3, stream);
            }
#endif
            return newAttributesCount;
        }

        // Refactor the target mesh into format that is most efficient for ZivaRT. Positions get computed before
        // Normals, Tangents and Motion Vectors so we put them in their own vertex buffer that we can overlap
        // the positions mesh update with Normal/Tangent/Movecs Burst Jobs. 
        void CreateRequiredAttributes(Mesh targetMesh, RecomputeTangentFrames recomputeTangentFrames,
            bool calculateMotionVectors)
        {                       
            int newAttributesCount = 0;
            int maxZivaRTStreamIndex = (recomputeTangentFrames != RecomputeTangentFrames.None) ? 1 : 0;
#if MOTION_VECTORS
            maxZivaRTStreamIndex = calculateMotionVectors ? 1 : maxZivaRTStreamIndex;
#endif
            VertexAttributeDescriptor[] newAttributes = new VertexAttributeDescriptor[3]; // normals, tangents, movecs
            newAttributesCount = SetupStream(newAttributes, targetMesh, recomputeTangentFrames, calculateMotionVectors,
                (int)AttributeStreams.NormalTangentMovecs);

            VertexAttributeDescriptor[] currentAttributes = targetMesh.GetVertexAttributes();
            // remap any existing attributes so that they are on the streams that we want them to be on
            for (int i = 0; i < currentAttributes.Length; i++)
            {
                switch (currentAttributes[i].attribute)
                {
                    // positions will go to stream 0
                    case VertexAttribute.Position:
                        currentAttributes[i].stream = (int)AttributeStreams.Position;   
                        break;
                    // normals, tangents and motion vectors (if required) will go to stream 1
                    case VertexAttribute.Normal:
                    case VertexAttribute.Tangent:
                    case VertexAttribute.TexCoord5:
                        currentAttributes[i].stream = (int)AttributeStreams.NormalTangentMovecs;
                        break;
                    // move anything else out of the ZivaRT streams
                    default:
                        if (currentAttributes[i].stream <= maxZivaRTStreamIndex)
                        { 
                            currentAttributes[i].stream = maxZivaRTStreamIndex + 1;
                        }
                        break;
                }
            }
            VertexAttributeDescriptor[] finalAttributes = currentAttributes;
            CombineAttributes(ref finalAttributes, currentAttributes, newAttributes, newAttributesCount);
           
            // update mesh format, this also preserves old attribute values
            targetMesh.SetVertexBufferParams(targetMesh.vertexCount, finalAttributes);
        }

        public void Initialize(Mesh targetMesh, RecomputeTangentFrames recomputeTangentFrames, bool calculateMotionVectors)
        {
            CreateRequiredAttributes(targetMesh, recomputeTangentFrames, calculateMotionVectors);

            // For initilization use this Mesh api since it returns the CPU backed copy of mesh data
            // Getting the data from GraphicsBuffer directly would get the GPU copy and require a CPU/GPU sync
            Mesh.MeshDataArray meshData = Mesh.AcquireReadOnlyMeshData(targetMesh);
            ClearStreamMap();
            // setup all the required attribute streams after clearing them all out for easier debugging
            m_PositionsStreamData = new DataStream();
            m_NormalsStreamData = new DataStream();
            m_TangentsStreamData = new DataStream();
            m_MovecsStreamData = new DataStream();
            UpdateStreamMap(VertexAttribute.Position, meshData[0], targetMesh);
            UpdateDataStream(ref m_PositionsStreamData, meshData[0], VertexAttribute.Position);
            if (recomputeTangentFrames != RecomputeTangentFrames.None)
            {
                UpdateStreamMap(VertexAttribute.Normal, meshData[0], targetMesh);
                UpdateDataStream(ref m_NormalsStreamData, meshData[0], VertexAttribute.Normal);
                if (recomputeTangentFrames == RecomputeTangentFrames.NormalsAndTangents)
                {
                    UpdateStreamMap(VertexAttribute.Tangent, meshData[0], targetMesh);
                    UpdateDataStream(ref m_TangentsStreamData, meshData[0], VertexAttribute.Tangent);
                }
            }
#if MOTION_VECTORS
            if (calculateMotionVectors)
            {
                UpdateStreamMap(VertexAttribute.TexCoord5, meshData[0], targetMesh);
                UpdateDataStream(ref m_MovecsStreamData, meshData[0], VertexAttribute.TexCoord5);                
            }
#endif
            meshData.Dispose();
        }

        public void CommitChanges(AttributeStreams attributeStream)
        {
            Profiler.BeginSample("CommitChanges");
            // This is the most expensive call in the whole process of updating the mesh
            // since it performs a copy of all of the attribute data 
            if (m_StreamMap[(int)attributeStream] != null)
            {
                m_StreamMap[(int)attributeStream].SetData();
            }                
            Profiler.EndSample();
        }

        void ClearStreamMap()
        {
            for (int i = 0; i < m_StreamMap.Length; i++)
            {
                if (m_StreamMap[i] != null)
                {
                    m_StreamMap[i].DisposeNativeData();
                    m_StreamMap[i] = null;
                }
            }
        }

        public void Dispose()
        {
            ClearStreamMap();
        }
    }
}
