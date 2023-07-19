using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.ZivaRTPlayer
{
    // Job to compute normals and tangents for all of the faces in a mesh.
    // Parallel-for is over the triangles of the mesh.
    // The face normals are area-weighted instead of being normalized to length==1.0, because this
    // is both cheaper to compute and more useful for averaging face normals into vertex normals.
    // The face tangents are normalized, but they are not orthogonalized.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct ComputeFaceTangentFramesJob : IJobFor
    {
        // Vertex positions of the mesh.
        [ReadOnly]
        public VertexBufferInterface.DataStream Vertices;

        // Whether or not to recalculate tangent vectors as well as normals.
        // If false, the faceTangents output will not be updated.
        public bool CalculateTangents;

        // Vertex indices of all triangles of the mesh.
        [ReadOnly]
        NativeArray<int3> m_Triangles;
        [ReadOnly]
        NativeArray<float2x2> m_UVToTriTransforms;

        // Area-weighted normal vectors for every mesh triangle.
        [WriteOnly]
        public NativeArray<float3> FaceNormals;
        // Unit-length tangent and bitangent vectors for each mesh triangle.
        [WriteOnly]
        public NativeArray<float3x2> FaceTangents;

        public unsafe void Execute(int index)
        {
            int3 triangleVerts = m_Triangles[index];
            float3 v0 = *Vertices.GetPtrAtIndex<float3>(triangleVerts[0]);
            float3 v1 = *Vertices.GetPtrAtIndex<float3>(triangleVerts[1]);
            float3 v2 = *Vertices.GetPtrAtIndex<float3>(triangleVerts[2]);

            float3 e01 = v1 - v0;
            float3 e02 = v2 - v0;

            // NOTE: We are not normalizing the magnitude of the returned face normal vector.
            // The vector's magnitude is determined by the area of the triangle. This means that
            // when we add up the face normals when computing the vertex normals, we get an
            // area-weighted sum (which is arguably the better choice) while actually doing LESS
            // work.
            FaceNormals[index] = math.cross(e01, e02);

            if (CalculateTangents)
            {
                float2x2 uvToTri = m_UVToTriTransforms[index];
                float3x2 triToLocal = new float3x2(e01, e02);
                float3x2 uvToLocal = math.mul(triToLocal, uvToTri);

                // Normalize the axes of the computed basis.
                float3 tangentX = math.normalizesafe(uvToLocal.c0);
                float3 tangentY = math.normalizesafe(uvToLocal.c1);

                FaceTangents[index] = new float3x2(tangentX, tangentY);
            }
        }

        public void Initialize(int[] meshTriangles, MeshTangentFramesInfo tangentFramesInfo)
        {
            // We only actually need the top-left 2x2 block of the given matrix.
            var uvTriTransforms2x2 = new float2x2[tangentFramesInfo.UVToTriTransforms.Length];
            for (int t = 0; t < uvTriTransforms2x2.Length; ++t)
            {
                var transform4x4 = tangentFramesInfo.UVToTriTransforms[t];
                float2 col0 = new float2(transform4x4.m00, transform4x4.m10);
                float2 col1 = new float2(transform4x4.m01, transform4x4.m11);
                uvTriTransforms2x2[t] = new float2x2(col0, col1);
            }

            Initialize(meshTriangles, uvTriTransforms2x2);
        }

        public void Initialize(int[] meshTriangles, float2x2[] uvToTriangleTransforms)
        {
            int numTriangles = meshTriangles.Length / 3;
            Debug.Assert((uvToTriangleTransforms.Length == numTriangles) || (uvToTriangleTransforms.Length == 0));

            m_Triangles = new NativeArray<int3>(numTriangles, Allocator.Persistent);
            m_Triangles.Reinterpret<int>(3 * sizeof(int)).CopyFrom(meshTriangles);

            m_UVToTriTransforms = new NativeArray<float2x2>(uvToTriangleTransforms, Allocator.Persistent);

            FaceNormals = new NativeArray<float3>(numTriangles, Allocator.Persistent);
            FaceTangents = new NativeArray<float3x2>(numTriangles, Allocator.Persistent);
        }

        public int NumTriangles { get { return m_Triangles.Length; } }

        public void ReleaseBuffers()
        {
            if (m_Triangles.IsCreated)
                m_Triangles.Dispose();
            if (m_UVToTriTransforms.IsCreated)
                m_UVToTriTransforms.Dispose();
            if (FaceNormals.IsCreated)
                FaceNormals.Dispose();
            if (FaceTangents.IsCreated)
                FaceTangents.Dispose();
        }
    }

    // Job to compute vertex normals and tangents from mesh face tangent frames.
    // Parallel-for is over the vertices of the mesh.
    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    internal struct ComputeVertexTangentFramesJob : IJobFor
    {
        // Area-weighted normal vectors for each mesh face.
        [ReadOnly]
        public NativeArray<float3> FaceNormals;

        // Whether or not to recalculate tangent vectors as well as normals.
        // If false, the faceTangents input will not be read, and the vertexTangents output will
        // not be updated.
        public bool CalculateTangents;

        // Normalized (but not necessarily orthogonal) tangent basis for eah mesh face.
        [ReadOnly]
        public NativeArray<float3x2> FaceTangents;

        // Lists of faces associated with each vertex.
        // Vertices can have an arbitrary number of faces that influence their normal/tangent, so
        // we store an array of start indices for looking up the sub-arrays of faces for each vertex.

        [ReadOnly]
        NativeArray<int> m_VertexNormalFaces;
        [ReadOnly]
        NativeArray<int> m_VertexNormalFacesStarts;

        [ReadOnly]
        NativeArray<int> m_VertexTangentFaces;
        [ReadOnly]
        NativeArray<int> m_VertexTangentFacesStarts;

        // Normal vectors for each vertex of the mesh.
        [WriteOnly]
        public VertexBufferInterface.DataStream VertexNormals;

        // Tangent vectors for each vertex of the mesh, in Unity-expected format.
        // This allows the slice to alias with the vertexNormals slice, to allow them both to come
        // from the same vertex buffer.
        [Unity.Collections.LowLevel.Unsafe.NativeDisableContainerSafetyRestriction]
        [WriteOnly]
        public VertexBufferInterface.DataStream VertexTangents;

        public int NumVertices { get { return m_VertexNormalFacesStarts.Length - 1; } }

        public unsafe void Execute(int index)
        {
            int normalFacesStartIdx = m_VertexNormalFacesStarts[index];
            int normalFacesEndIdx = m_VertexNormalFacesStarts[index + 1];

            // Accumulate all face normals associated with the current vertex.
            float3 vertexNormal = float3.zero;
            for (int i = normalFacesStartIdx; i < normalFacesEndIdx; ++i)
            {
                int faceIdx = m_VertexNormalFaces[i];
                vertexNormal += FaceNormals[faceIdx];
            }
            vertexNormal = math.normalizesafe(vertexNormal);
            float3* pNormal = VertexNormals.GetPtrAtIndex<float3>(index);
            *pNormal = vertexNormal;

            if (CalculateTangents)
            {
                int tangentFacesStartIdx = m_VertexTangentFacesStarts[index];
                int tangentFacesEndIdx = m_VertexTangentFacesStarts[index + 1];

                // Accumulate all face tangents associated with the current vertex.
                float3x2 tangentSum = float3x2.zero;
                for (int i = tangentFacesStartIdx; i < tangentFacesEndIdx; ++i)
                {
                    int faceIdx = m_VertexTangentFaces[i];
                    tangentSum += FaceTangents[faceIdx];
                }
                if (tangentFacesStartIdx == tangentFacesEndIdx)
                {
                    float3 defaultTangentX = new float3(1.0f, 0.0f, 0.0f);
                    float3 defaultTangentY = new float3(0.0f, 1.0f, 0.0f);
                    tangentSum = new float3x2(defaultTangentX, defaultTangentY);
                }

                // Project tangent-x vector into the plane of the vertexNormal (which is already
                // unit-length), then normalize it.
                float3 tangentX = tangentSum.c0 - math.dot(tangentSum.c0, vertexNormal) * vertexNormal;
                tangentX = math.normalizesafe(tangentX);

                // We don't return a tangentY vector directly. Instead Unity reconstructs tangentY
                // (orthonormalized against both tangentX and vertexNormal) at render-time as:
                // Cross(normal, tangent.xyz) * tangent.w
                // Therefore, we need to set tangent.w to make this reconstruction correct.
                float w = math.sign(math.dot(math.cross(vertexNormal, tangentX), tangentSum.c1));

                float4* pTangent = VertexTangents.GetPtrAtIndex<float4>(index);
                *pTangent = new float4(tangentX, w);
            }
        }

        public void Initialize(MeshTangentFramesInfo tangentFramesInfo)
        {
            int numVertices = tangentFramesInfo.VertexNormalFaces.Length;
            Debug.Assert(tangentFramesInfo.VertexTangentFaces.Length == numVertices);

            // Build flattened buffer of face indices, with start indices for each vertex.
            tangentFramesInfo.GetConcatenatedNormalFaces(out var normalFaces, out var normalFacesStarts);
            m_VertexNormalFaces = new NativeArray<int>(normalFaces, Allocator.Persistent);
            m_VertexNormalFacesStarts = new NativeArray<int>(normalFacesStarts, Allocator.Persistent);
            tangentFramesInfo.GetConcatenatedTangentFaces(out var tangentFaces,
                                                          out var tangentFacesStarts);
            m_VertexTangentFaces = new NativeArray<int>(tangentFaces, Allocator.Persistent);
            m_VertexTangentFacesStarts = new NativeArray<int>(tangentFacesStarts, Allocator.Persistent);
        }

        public void ReleaseBuffers()
        {
            if (m_VertexNormalFaces.IsCreated)
                m_VertexNormalFaces.Dispose();
            if (m_VertexNormalFacesStarts.IsCreated)
                m_VertexNormalFacesStarts.Dispose();
            if (m_VertexTangentFaces.IsCreated)
                m_VertexTangentFaces.Dispose();
            if (m_VertexTangentFacesStarts.IsCreated)
                m_VertexTangentFacesStarts.Dispose();
        }
    }

    // A class to compute and store information used when recalculating mesh tangent frames.
    //
    // MeshTangentFramesInfo tracks both the set of faces that influence the tangent at each
    // vertex, as well as the inverse UV-basis for each triangle in the mesh. Both of these are
    // needed in order to recalculate tangent/bitangent vectors and do not change when mesh vertex
    // positions change. The set of faces that influence the normal at each vertex is stored
    // separately, because it is not necessarily the same as the set of faces for the tangents.
    //
    // A mesh may have multiple vertices that are co-located and are meant to represent the same
    // point on the mesh surface. These co-located vertices are often generated because one or more
    // of their non-positional attributes need to be different on some faces than on others,
    // eg: UV map seams, hard corners from discontinuous normals. For the purpose of computing
    // smooth normals across a mesh, any vertices that are meant to keep both the same position and
    // same normal throughout deformation need to be treated like they're the same vertex of a
    // "smooth" mesh. The "smooth" faces are all the faces surrounding a "smooth" vertex, and these
    // are the faces that contribute to the computation of a smooth normal for that vertex.
    // Practically, all vertices that correspond to a particular "smooth" vertex share the same
    // "smooth" face neighborhood, which ends up being the union of all the faces of all those
    // vertices.
    //
    // NOTE: This is exactly the same concept as "Smoothing Groups" on a mesh, which tell us which
    // vertices of a mesh are meant to keep the same normal vector under deformation.
    // However, there doesn't appear to be a way to extract the smoothing groups of a mesh from the
    // Unity API. So instead we're deducing smoothness ourselves by finding all vertices that share
    // the same position and normal vector and assuming that those vertices should share the same
    // smooth face neighbourhood for the purposes of recomputing normals.
    internal class MeshTangentFramesInfo
    {
        List<int>[] m_TangentFaces;      // Faces used to compute tangents for each vertex
        List<int>[] m_NormalFaces;       // Faces used to compute normals for each vertex
        Matrix4x4[] m_UVToTriTransforms; // Pre-compute inverse UV bases for each triangle

        public List<int>[] VertexTangentFaces { get { return m_TangentFaces; } }
        public List<int>[] VertexNormalFaces { get { return m_NormalFaces; } }
        public Matrix4x4[] UVToTriTransforms { get { return m_UVToTriTransforms; } }

        // Pre-compute information for tangent frame recalculation for a mesh.
        //
        // This function assumes you have already determined which vertices of the Mesh are
        // co-located, aka represent the same point on the mesh. This information is provided to
        // the function as a mapping from Mesh vertex index to a "unique" vertex number, where all
        // vertices sharing the same unique vertex number are co-located. It does not matter what
        // the specific values of the unique vertex numbers are or how they are ordered. It only
        // matters that from this information we can tell which vertices of the Mesh should be
        // treated as being the same point in space.
        public static MeshTangentFramesInfo Build(
            Mesh mesh,
            int[] uniqueVertexMap,
            int numUniqueVertices,
            bool suppressWarningMessages = false)
        {
            int numMeshVertices = mesh.vertexCount;
            Assert.AreEqual(numMeshVertices, uniqueVertexMap.Length);

            int[] meshTriangles = mesh.triangles; // Cache triangles locally for performance.

            // In order to recompute tangent frames efficiently later, we need to know which faces each
            // mesh vertex belongs to.
            // In order to ensure smooth normals, we want co-located vertices with the same normal to
            // be treated as the same vertex, meaning they both belong to the same unified set of faces.
            // Confusingly, we DON'T want to do the same for computing the Tangent vectors; it's okay
            // if co-located vertices don't share the same Tangent vector even if they should keep the
            // same Normal vector.

            // First just go through and record which triangles contain each vertex.
            // This does not take smooth normals into account.
            var vertexFaces = BuildVertexToFaceMap(numMeshVertices, meshTriangles);

            // Now go through and unify triangle lists for co-located vertices that have the same normal.
            var smoothVertexNormalFaces =
                BuildSmoothNormalVertexFaceMap(mesh, uniqueVertexMap, numUniqueVertices, vertexFaces);

            // Finally, pre-compute inverse UV bases and triangle lists needed for tangent recalculation.
            BuildTangentRecalcInfo(
                mesh, meshTriangles, vertexFaces, out var vertexTangentFaces,
                out var uvToTriangleTransforms, suppressWarningMessages);

#if false
            //
            // Validation (for debugging, this is slow for large meshes)
            //
            Profiler.BeginSample("Validation");
            var meshVertices = mesh.vertices; // Cache vertices locally for performance.
            var meshNormals = mesh.normals;
            for (int v = 0; v < numMeshVertices; ++v)
            {
                // Every vertexTangentFace triangle should contain the vertex.
                foreach (int tri in vertexTangentFaces[v])
                {
                    int v0 = meshTriangles[tri * 3 + 0];
                    int v1 = meshTriangles[tri * 3 + 1];
                    int v2 = meshTriangles[tri * 3 + 2];

                    Assert.IsTrue(v == v0 || v == v1 || v == v2);
                }

                // Every smoothVertexFace triangle should contain a vertex with the same position and normal.
                Vector3 position = meshVertices[v];
                Vector3 normal = meshNormals[v];
                foreach(int tri in smoothVertexNormalFaces[v])
                {
                    var vertexMatches = false;
                    for (int i = 0; i < 3; ++i)
                    {
                        int v_i = meshTriangles[tri * 3 + i];

                        Vector3 p = meshVertices[v_i];
                        Vector3 n = meshNormals[v_i];

                        vertexMatches |= (p.Equals(position) && n.Equals(normal));
                    }

                    Assert.IsTrue(vertexMatches);
                }
            }
#endif
            return new MeshTangentFramesInfo
            {
                m_TangentFaces = vertexTangentFaces,
                m_NormalFaces = smoothVertexNormalFaces,
                m_UVToTriTransforms = uvToTriangleTransforms,
            };
        }

        static List<int>[] BuildVertexToFaceMap(int numMeshVertices, int[] meshTriangles)
        {
            Profiler.BeginSample("BuildVertexToFaceMap");

            var vertexFaces = new List<int>[numMeshVertices]; // For computing tangents
            for (int v = 0; v < vertexFaces.Length; ++v)
            {
                vertexFaces[v] = new List<int>(6); // Triangle meshes average 6 faces per vertex
            }

            int numTriangles = meshTriangles.Length / 3;
            for (int t = 0; t < numTriangles; ++t)
            {
                int triangleStartIdx = t * 3;
                for (int i = 0; i < 3; ++i)
                {
                    int vtx = meshTriangles[triangleStartIdx + i];
                    vertexFaces[vtx].Add(t);
                }
            }
            Profiler.EndSample();

            return vertexFaces;
        }

        static List<int>[] BuildSmoothNormalVertexFaceMap(
            Mesh mesh,
            int[] uniqueVertexMap,
            int numUniqueVertices,
            List<int>[] vertexFaces)
        {
            // We are given a mapping from full mesh vertices to unique vertices.
            // What we actually need is the inverse mapping, from unique vertex to all of the co-located
            // vertices at that position in the full mesh.
            Profiler.BeginSample("BuildUniqueToFullMeshMap");
            var uniqueToFullMeshMap = new List<int>[numUniqueVertices];
            for (int u = 0; u < uniqueToFullMeshMap.Length; ++u)
            {
                uniqueToFullMeshMap[u] = new List<int>(1);
            }
            for (int v = 0; v < uniqueVertexMap.Length; ++v)
            {
                int uniqueVtx = uniqueVertexMap[v];
                if (uniqueVtx >= 0)
                {
                    uniqueToFullMeshMap[uniqueVtx].Add(v);
                }
            }
            Profiler.EndSample();

            Profiler.BeginSample("SmoothVertexNormalFaceNeighbourhood");
            int numMeshVertices = mesh.vertexCount;
            var meshNormals = mesh.normals; // Cache normals locally for performance.
            var smoothVertexNormalFaces = new List<int>[numMeshVertices]; // For computing normals
            for (int v = 0; v < smoothVertexNormalFaces.Length; ++v)
            {
                // This set will be very small (~6 integers), so it's not worth using a HashSet.
                var smoothFacesSet = new List<int>(6);

                int uniqueVtx = uniqueVertexMap[v];
                if (uniqueVtx >= 0)
                {
                    var normal = meshNormals[v];

                    var colocatedVertices = uniqueToFullMeshMap[uniqueVtx];
                    foreach (int colocatedVtx in colocatedVertices)
                    {
                        // Check if the colocated vertex shares the same normal as the current vertex.
                        var colocatedNormal = meshNormals[colocatedVtx];
                        if (colocatedNormal.Equals(normal))
                        {
                            // Found a vertex sharing the same position and normal, include all of its
                            // faces in our current vertex's face set.
                            var colocatedVtxFaces = vertexFaces[colocatedVtx];
                            foreach (int face in colocatedVtxFaces)
                            {
                                // Don't add the same face more than once.
                                if (!smoothFacesSet.Contains(face)) smoothFacesSet.Add(face);
                            }
                        }
                    }

                    smoothFacesSet.Sort(); // Restore the original ordering of faces, for consistency.
                    smoothVertexNormalFaces[v] = smoothFacesSet;
                }
                else
                {
                    // This shouldn't happen, but just in case we have no way to find co-located vertices
                    // via the unique vertex mesh mapping, just copy the non-smooth face list.
                    smoothVertexNormalFaces[v] = new List<int>(vertexFaces[v]);
                }
            }

            Profiler.EndSample();

            return smoothVertexNormalFaces;
        }

        static void BuildTangentRecalcInfo(
            Mesh mesh,
            int[] meshTriangles,
            List<int>[] vertexToFaceMap,
            out List<int>[] vertexTangentFaces,
            out Matrix4x4[] uvToTriangleTransforms,
            bool suppressWarningMessages = false)
        {
            var meshUVs = mesh.uv; // TODO: Handle cases where multiple UVs are provided?

            uvToTriangleTransforms =
                ComputeUVToTriangleTransforms(meshUVs, meshTriangles, out var isTriDegenerate);

            vertexTangentFaces = BuildVertexTangentFaceMap(
                vertexToFaceMap, isTriDegenerate, suppressWarningMessages);
        }

        static Matrix4x4[] ComputeUVToTriangleTransforms(
            Vector2[] meshUVs,
            int[] meshTriangles,
            out bool[] isTriangleDegenerate)
        {
            isTriangleDegenerate = new bool[meshTriangles.Length];

            // Not all meshes have UVs. In that case, don't store any transforms.
            if (meshUVs.Length == 0)
                return new Matrix4x4[0];

            int numTriangles = meshTriangles.Length / 3;
            var uvToTriTransforms = new Matrix4x4[numTriangles];

            for (int t = 0; t < numTriangles; ++t)
            {
                int i0 = meshTriangles[3 * t + 0];
                int i1 = meshTriangles[3 * t + 1];
                int i2 = meshTriangles[3 * t + 2];

                var uv0 = meshUVs[i0];
                var uv1 = meshUVs[i1];
                var uv2 = meshUVs[i2];

                var uv01 = uv1 - uv0;
                var uv02 = uv2 - uv0;

                Matrix4x4 triToUV = Matrix4x4.identity;
                triToUV.SetColumn(0, uv01);
                triToUV.SetColumn(1, uv02);

                if (triToUV.determinant == 0.0f)
                {
                    // The UVs for this triangle are degenerate!
                    // We don't have an invertible UV basis, so we don't know how to go from
                    // triangle-basis space to UV-space.
                    uvToTriTransforms[t] = Matrix4x4.zero; // TODO: Is there a better alternative?

                    // Debug.LogWarningFormat(
                    //     "Degenerate UVs on triangle #{0}! (Vertices {1}, {2}, {3}; uv1={4}, uv2={5}) Can't
                    //     ensure smooth tangents.", t, i0, i1, i2, uv01, uv02);

                    isTriangleDegenerate[t] = true;
                }
                else
                {
                    uvToTriTransforms[t] = triToUV.inverse;
                    isTriangleDegenerate[t] = false;
                }
            }

            return uvToTriTransforms;
        }

        static List<int>[] BuildVertexTangentFaceMap(
            List<int>[] vertexToFaceMap,
            bool[] isTriDegenerate,
            bool suppressWarningMessages = false)
        {
            var vertexTangentFaces = new List<int>[vertexToFaceMap.Length];
            // Indices of vertices that tangents cannot be computed for
            List<int> badUVVertices = new List<int>();

            // Filter out triangles with degenerate UV bases from the vertex-to-face map to create
            // the set of triangles used to compute the tangents of each vertex.
            for (int v = 0; v < vertexToFaceMap.Length; ++v)
            {
                vertexTangentFaces[v] = new List<int>(vertexToFaceMap[v].Count);
                foreach (int t in vertexToFaceMap[v])
                {
                    if (!isTriDegenerate[t])
                    {
                        vertexTangentFaces[v].Add(t);
                    }
                }

                if (vertexTangentFaces[v].Count == 0)
                {
                    badUVVertices.Add(v);
                }
            }
            if (badUVVertices.Count > 0 && !suppressWarningMessages)
            {
                Debug.LogWarningFormat(
                    "Cannot compute tangent vectors for the following vertices due to degenerate face UVs: {0}",
                    string.Join(", ", badUVVertices));
            }

            // TODO: Ensure that we create the same set of triangles for each vertex that mikktspace does.

            return vertexTangentFaces;
        }

        static void CreateConcatenatedArray(
            List<int>[] arrays,
            out int[] concatenated,
            out int[] subarrayStarts)
        {
            // First store the start indices of each subarray in the final concatenated array.
            // We actually store numSubarrays + 1 entries, and the final entry acts as the end of
            // the index range of the final subarray. This allows us to easily obtain the index
            // range for any subarray, with no special-casing. It also gives us a total count of
            // the number of elements in the concatenated array.
            int numSubarrays = arrays.Length;
            subarrayStarts = new int[numSubarrays + 1];
            subarrayStarts[0] = 0;
            for (int i = 0; i < numSubarrays; ++i)
            {
                int numSubarrayElements = arrays[i].Count;
                subarrayStarts[i + 1] = subarrayStarts[i] + numSubarrayElements;
            }

            // Now allocate space for the full concatenated array, and fill it in.
            int totalNumElements = subarrayStarts[subarrayStarts.Length - 1];
            concatenated = new int[totalNumElements];
            for (int i = 0; i < numSubarrays; ++i)
            {
                var currArray = arrays[i];
                int startIdx = subarrayStarts[i];
                for (int j = 0; j < currArray.Count; ++j)
                {
                    concatenated[startIdx + j] = currArray[j];
                }
            }
        }

        public void GetConcatenatedNormalFaces(out int[] concatenated, out int[] subarrayStarts)
        {
            CreateConcatenatedArray(this.m_NormalFaces, out concatenated, out subarrayStarts);
        }

        public void GetConcatenatedTangentFaces(out int[] concatenated, out int[] subarrayStarts)
        {
            CreateConcatenatedArray(this.m_TangentFaces, out concatenated, out subarrayStarts);
        }
    }

    // Reference implementation of tangent basis calculations for a mesh.
    // Given a mesh's vertex positions, triangle indices, and pre-computed tangent frame info,
    // compute new tangent frames for each vertex.
    // This is done by first computing tangent frames for each face, and then averaging together
    // all the faces that should contribute to the normals and tangents of each vertex.
    internal class ComputeTangentFrames
    {
        struct TangentFrame
        {
            public Vector3 TangentX;
            public Vector3 TangentY;
            public Vector3 Normal;

            // public Vector3 VertexFaceAngles; // For weighting sums by angle
        }
        TangentFrame[] m_FaceTangents;

        // Constructor
        public ComputeTangentFrames(int numTriangles) { m_FaceTangents = new TangentFrame[numTriangles]; }

        public void Compute(
            Vector3[] meshVertices,
            int[] meshTriangles,
            MeshTangentFramesInfo tangentFramesInfo,
            ref Vector3[] vertexNormals,
            ref Vector4[] vertexTangents)
        {
            CalcTangentFrames(
                meshVertices, meshTriangles, tangentFramesInfo, ref this.m_FaceTangents, ref vertexNormals,
                ref vertexTangents);
        }

        public void Compute(
            Vector3[] meshVertices,
            int[] meshTriangles,
            MeshTangentFramesInfo tangentFramesInfo,
            ref Vector3[] vertexNormals)
        {
            Vector4[] vertexTangents = null;
            CalcTangentFrames(
                meshVertices, meshTriangles, tangentFramesInfo, ref this.m_FaceTangents, ref vertexNormals,
                ref vertexTangents);
        }

        static void CalcTangentFrames(
            Vector3[] meshVertices,
            int[] meshTriangles,
            MeshTangentFramesInfo tangentFramesInfo,
            ref TangentFrame[] faceTangents,
            ref Vector3[] vertexNormals,
            ref Vector4[] vertexTangents)
        {
            bool computeTangents = (vertexTangents != null && vertexTangents.Length > 0);

            int numVertices = meshVertices.Length;
            int numTriangles = meshTriangles.Length / 3;

            // Start by computing face normals for every triangle in the mesh.
            Profiler.BeginSample("Compute Face Tangent Frames");
            for (int t = 0; t < numTriangles; ++t)
            {
                var uvToTriTransform = tangentFramesInfo.UVToTriTransforms[t];
                faceTangents[t] =
                    CalcFaceTangentFrame(meshVertices, meshTriangles, t, uvToTriTransform, computeTangents);
            }
            Profiler.EndSample();

            // Then go over all vertices, accumulating the normals and tangents of the faces they
            // belong to.
            Profiler.BeginSample("Calc Vertex Tangent Frames");
            for (int v = 0; v < numVertices; ++v)
            {
                var vertexNormal = CalcVertexNormal(v, tangentFramesInfo, faceTangents);
                vertexNormals[v] = vertexNormal;

                if (computeTangents)
                {
                    vertexTangents[v] = CalcVertexTangent(
                        v, tangentFramesInfo,
                        // meshVertices, meshTriangles,
                        faceTangents, vertexNormal);
                }
            }
            Profiler.EndSample();
        }

        static TangentFrame CalcFaceTangentFrame(
            Vector3[] meshVertices,
            int[] meshTriangles,
            int triangle,
            Matrix4x4 uvToTri,
            bool computeTangents)
        {
            int triangleStartIdx = triangle * 3;
            int i0 = meshTriangles[triangleStartIdx + 0];
            int i1 = meshTriangles[triangleStartIdx + 1];
            int i2 = meshTriangles[triangleStartIdx + 2];

            Vector3 p0 = meshVertices[i0];
            Vector3 p1 = meshVertices[i1];
            Vector3 p2 = meshVertices[i2];

            Vector3 e01 = p1 - p0;
            Vector3 e02 = p2 - p0;

            // NOTE: We are NOT normalizing the magnitude of the normal vector, so the magnitude is
            // weighted by the area of the triangle. This allows us to do an area-weighted sum of
            // face normals while actually doing LESS work.
            Vector3 n = Vector3.Cross(e01, e02);

            Vector3 tangentX = Vector3.zero, tangentY = Vector3.zero;
            if (computeTangents)
            {
                Matrix4x4 triToLocal = new Matrix4x4(e01, e02, Vector4.zero, Vector4.zero);
                Matrix4x4 uvToLocal = triToLocal * uvToTri;
                tangentX = unsafeNormalized(uvToLocal.GetColumn(0));
                tangentY = unsafeNormalized(uvToLocal.GetColumn(1));
            }

            //// If we want to angle-weight the sum of tangent frames at each vertex, we can compute
            //// each of this triangle's angles and store them associated with the corresponding
            //// vertex.
            // Vector3 vertexFaceAngles = Vector3.zero;
            // e01.Normalize();
            // e02.Normalize();
            // Vector3 e12 = (p2 - p1).normalized;
            //// NOTE: These angles are in the plane of the triangle, but mikktspace actually
            //// computes them in the plane of the vertex normal! We don't know the vertex normal yet
            //// at this point in the code. :(
            // vertexFaceAngles[0] = math.acos(Vector3.Dot(e01, e02));
            // vertexFaceAngles[1] = math.acos(-Vector3.Dot(e01, e12));
            // vertexFaceAngles[2] = math.acos(Vector3.Dot(e02, e12));

            return new TangentFrame
            {
                TangentX = tangentX,
                TangentY = tangentY,
                Normal = n,
                // VertexFaceAngles = vertexFaceAngles,
            };
        }

        static Vector3 CalcVertexNormal(
            int vtx,
            MeshTangentFramesInfo tangentFramesInfo,
            TangentFrame[] faceTangents)
        {
            // NOTE: Because ComputeFaceTangentFrame() returns an area-weighted normal, we get
            // an area-weighted sum.
            var vertexNormal = Vector3.zero;
            var faces = tangentFramesInfo.VertexNormalFaces[vtx];
            var numFaces = faces.Count;
            for (int i = 0; i < numFaces; ++i)
            {
                int t = faces[i];
                vertexNormal += faceTangents[t].Normal;
            }
            return unsafeNormalized(vertexNormal);
        }

        static Vector4 CalcVertexTangent(
            int vtx,
            MeshTangentFramesInfo tangentFramesInfo,
            // Vector3[] meshVertices, int[] meshTriangles,
            TangentFrame[] faceTangents,
            Vector3 vertexNormal)
        {
            // Accumulate tangents from all faces containing the vertex.
            Vector3 tangentX = Vector3.zero;
            Vector3 tangentY = Vector3.zero;
            var faces = tangentFramesInfo.VertexTangentFaces[vtx];
            var numFaces = faces.Count;
            for (int i = 0; i < numFaces; ++i)
            {
                int t = faces[i];

                // Unweighted sum:
                tangentX += faceTangents[t].TangentX;
                tangentY += faceTangents[t].TangentY;

                //// If we want to do an angle-weighted sum, we need the angle of the current
                //// triangle at the current vertex.
                // int i0 = meshTriangles[3 * t + 0];
                // int i1 = meshTriangles[3 * t + 1];
                // int i2 = meshTriangles[3 * t + 2];

                // int vtx1 = i1, vtx2 = i2;
                // if (vtx == i1)
                //{
                //     vtx1 = i2;
                //     vtx2 = i0;
                // }
                // else if (vtx == i2)
                //{
                //     vtx1 = i0;
                //     vtx2 = i1;
                // }

                //// NOTE: mikktspace computes angles in the plane of the vertex normal, NOT in the
                //// plane of the triangle itself! That means we can't compute these angles until
                //// after the vertex normal has been calculated.
                // Vector3 p0 = meshVertices[vtx];
                // Vector3 p1 = meshVertices[vtx1];
                // Vector3 p2 = meshVertices[vtx2];
                // Vector3 e01 = p1 - p0;
                // Vector3 e02 = p2 - p0;
                // Vector3 e01_proj = Vector3.ProjectOnPlane(e01, vertexNormal);
                // Vector3 e02_proj = Vector3.ProjectOnPlane(e02, vertexNormal);

                // var cosAngle = Vector3.Dot(e01_proj.normalized, e02_proj.normalized);
                // var angle = math.acos(cosAngle);

                //// NOTE: mikktspace projects face tangents into the plane of the vertex normal
                //// before summing.
                // tangentX += angle * Vector3.ProjectOnPlane(faceTangents[t].tangentX,
                // vertexNormal).normalized; tangentY += angle *
                // Vector3.ProjectOnPlane(faceTangents[t].tangentY, vertexNormal).normalized;
            }

            if (numFaces == 0)
            {
                // We don't have any non-degenerate faces from which to compute a tangent!

                // TODO: What if tangentX == Vector3.zero? Should we also do this then?

                // Default to a consistent vector that is NOT the normal vector:
                tangentX = new Vector3(1, 0, 0);
                if (vertexNormal == tangentX) tangentX = new Vector3(0, 1, 0);

                tangentY = Vector3.Cross(vertexNormal, tangentX);

                // Debug.LogWarningFormat(
                //     "Cannot compute tangent for vtx #{0}! Defaulting to: Normal = {1}, TangentX = {2},
                //     TangentY = {3}", vtx, vertexNormal, tangentX, tangentY);
            }

            // Project tangentX into the plane defined by the normal.
            Orthonormalize(ref tangentX, vertexNormal);

            // We don't actually return the tangentY vector, instead we store its orientation
            // in the w-component of the returned tangent vector.
            return ConvertToUnityTangent(tangentX, tangentY, vertexNormal);
        }

        // Normalize a non-normalized normal/tangent vector.
        //
        // Vector3.Normalize is NOT suitable for normalizing normals/tangents.
        // Normals, computed via cross product, will have a length approximately 
        // equal to the area of their triangle. For a 1mm triangle measured in meters,
        // this will be a vector magnitude of 1e-6. Unity's Vector3.Normalize code
        // treats lengths less than 1e-4 as zero, producing totally broken normals.        
        //
        // If the input is too large, small, or not finite,
        // then this returns NaN or Inf or other bad results.
        // What else might we do in the case of such a failure?
        // We are computing vertex normals and tangents.
        // There is no safe "default" normal/tangent.
        // If we return the zero vector, then the shading will be all screwed up.
        // If we return [1,0,0], then the shading will be all screwed up.
        // That's the same result we'll get if we just return the 'bad' result.
        // So, we do NOT detect failure. Just let it happen.
        static Vector3 unsafeNormalized(Vector3 x)
        {
            // Note that x / x.magnitude can fail due to many things, including:
            // -- overflow of x.magnitude,
            // -- underflow of x.magnitude,
            // -- x has a nan or inf component,
            // -- weird stuff with denormals,
            // -- ...
            // Don't try to predict failure. If you really want to handle that case,
            // then try to catch it afterwards, e.g. !float.IsFinite(result[0]).
            return x / x.magnitude;
        }

        // NOTE: perpendicularTo must already be unit-length!
        static void Orthonormalize(ref Vector3 makeOrtho, Vector3 perpendicularTo)
        {
            // Subtract out the component of makeOrtho that's in the direction of perpendicularTo,
            // aka the projection of makeOrtho onto perpendicularTo.
            // The projection math is simpified because perpendicularTo is already unit-length.
            makeOrtho -= Vector3.Dot(perpendicularTo, makeOrtho) * perpendicularTo;
            makeOrtho = unsafeNormalized(makeOrtho);
        }

        static Vector4 ConvertToUnityTangent(Vector3 tangentX, Vector3 tangentY, Vector3 normal)
        {
            // At render-time the tangentY vector is reconstructed by:
            // Cross(normal, tangent.xyz) * tangent.w
            // Therefore, we need to set tangent.w to make this reconstruction correct.
            var tripleProduct = Vector3.Dot(Vector3.Cross(normal, tangentX), tangentY);
            float tangentYOrientation;
            if (tripleProduct < 0.0)
            {
                tangentYOrientation = -1.0f;
            }
            else
            {
                tangentYOrientation = 1.0f;
            }
            Vector4 vertexTangent = tangentX;
            vertexTangent.w = tangentYOrientation;
            return vertexTangent;
        }
    }
}
