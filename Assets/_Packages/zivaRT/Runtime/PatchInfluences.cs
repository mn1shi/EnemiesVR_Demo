using System.Collections.Generic;

namespace Unity.ZivaRTPlayer
{
    // Common utility functions for dealing with the set of vertices influenced by ZRT patches.
    // A patch influences a set of vertices of a mesh, and each vertex can be influenced by an
    // arbitrary number of patches.
    // The ZRT Character data stores the set of vertices influenced by a patch, but for some of our
    // solver implementations what we actually need is the set of patches that influence a given
    // vertex.
    internal static class PatchInfluences
    {
        // Convenience struct that allows us to locate a specific vertex in a specific patch.
        // Our goal during initialization is to convert these two-integer lookup indices into a
        // single index integer into the concatenated displacements array.
        public struct PatchVertexIndex
        {
            public int Patch;
            public int Vertex;
        }

        // The zivaAsset has a list of patches, each of which has a list of shape vertices that
        // the patch influences. We want to "invert" that into a list of patch-vertices (aka
        // displacements) that influence a specific shape vertex.
        public static List<PatchVertexIndex>[] InvertPatchVertexMap(ZivaRTRig zivaAsset)
        {
            int numShapeVertices = zivaAsset.m_Character.NumVertices;
            var vertexInfluenceLists = new List<PatchVertexIndex>[numShapeVertices];
            for (int i = 0; i < vertexInfluenceLists.Length; ++i)
            {
                vertexInfluenceLists[i] = new List<PatchVertexIndex>(1);
            }

            for (int p = 0; p < zivaAsset.m_Patches.Length; ++p)
            {
                var patch = zivaAsset.m_Patches[p];

                // For each vertex that the current patch influences, register the patch and
                // patch-vertex-displacement-index in the list of influences for that vertex.
                for (int patchVertex = 0; patchVertex < patch.Vertices.Length; ++patchVertex)
                {
                    var shapeVertex = patch.Vertices[patchVertex];

                    vertexInfluenceLists[shapeVertex].Add(new PatchVertexIndex
                    {
                        Patch = p,
                        Vertex = patchVertex,
                    });
                }
            }
            return vertexInfluenceLists;
        }

        // When flattening/concatenating our vertex influence lists into a single array, we need to
        // track the indexing structure of the sub-arrays.
        // Each entry in the returned array tell us the start index for each shape vertex's
        // sub-array of influences, ie: each vertex is influenced by the range:
        // influences[influenceStarts[vertIdx] : influenceStarts[vertIdx+1]]
        public static int[] CalcInfluenceStartIndices(List<PatchVertexIndex>[] vertexInfluenceLists)
        {
            int numShapeVertices = vertexInfluenceLists.Length;
            var influencesStarts = new int[numShapeVertices + 1];
            influencesStarts[0] = 0;
            for (int shapeVertex = 0; shapeVertex < numShapeVertices; ++shapeVertex)
            {
                int numInfluences = vertexInfluenceLists[shapeVertex].Count;
                influencesStarts[shapeVertex + 1] = influencesStarts[shapeVertex] + numInfluences;
            }
            return influencesStarts;
        }
    }
}
