using System;
using UnityEngine;

namespace Unity.ZivaRTPlayer
{
    internal class MeshRegistration
    {
        /// <summary>
        /// Partition the array `arr` into two non-empty parts,
        /// where every element of the first part is less than or equal to
        /// every element of the second part.
        /// Post: 
        /// - arr is a permutation of its original entries.
        /// - returns the partition point `i`.
        /// - there exists some element, `p`, in the array s.t.
        ///   - arr[0, i) are all <= p
        ///   - arr[i,end) are all >= p.
        /// </summary>
        static int HoarePartition(int[] arr, int start, int end, Func<int, int, bool> less)
        {
            // A standard Hoare-style partition. See Wikipedia's page on QuickSort.

            Debug.Assert(start <= end); // Cannot partition inverted range.            
            if (end <= start + 1)
                return start; // Empty or Single-element array is already partitioned.

            // pivot must never be the last element of the array,
            // or the logic below will break.
            int pivot = arr[(start + end - 1) / 2];

            int i = start - 1;
            int j = end;

            while (true)
            {
                do
                {
                    ++i;
                }
                while (less(arr[i], pivot));
                do
                {
                    --j;
                }
                while (less(pivot, arr[j]));
                if (j <= i)
                {
                    return j + 1; // +1 to convert back to our convention of one-past-the-end
                }
                int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }

        /// <summary>
        /// Re-order the elements of arr[left:right) so that
        /// arr[left:mid) <= arr[mid] <= arr[mid+1:right).
        /// Like C++'s nth_element.
        /// </summary>
        static void QuickSelect(int[] arr, int left, int mid, int right, Func<int, int, bool> less)
        {
            while (left + 1 < right) // while [left:right) is more than 1 element.
            {
                Debug.Assert(left <= mid);
                Debug.Assert(mid < right);

                int pivot = HoarePartition(arr, left, right, less);

                if (pivot <= mid)
                    left = pivot;
                else
                    right = pivot;
            }
        }

        /// <summary>
        /// [first, last) is a kdTree. Get its midpoint/root.
        /// The kdTreeBuild and kdTreeQuery steps must exactly match, so they both call this.
        /// </summary>
        static int midpoint(int first, int last)
        {
            return first + (last - first) / 2;
        }

        /// <summary>
        /// Build a KD-Tree out of \p points.
        /// The tree is represented as list of indices into the \p points array.
        /// [first, last) is the list of which points to build the tree out of.
        /// \p axis is which axis to split the root of the tree based on.
        /// </summary>
        static void kdTreeBuild(Vector3[] points, int[] kdtree, int first, int last, int axis = 0)
        {
            int numPoints = last - first;

            // A tree of 0 or 1 points is already in order.
            if (numPoints <= 1)
                return;

            Func<int, int, bool> compare = (int p1, int p2) => (points[p1][axis] < points[p2][axis]);

            int mid = midpoint(first, last);

            QuickSelect(kdtree, first, mid, last, compare);

            axis = (axis + 1) % 3;

            // Note: QuickSelect is doing a perfectly balanced partition,
            // so this recusion depth is bounded by log2(numPoints).
            kdTreeBuild(points, kdtree, first, mid, axis);
            kdTreeBuild(points, kdtree, mid + 1, last, axis);
        }

        /// <summary> 
        /// Find the point in a kdTree that is nearest to the query point \p,
        /// and within the ball of radius \p limit.
        /// </summary>
        /// <param name="x"><The query point./param>
        /// <param name="points"></param>
        /// <param name="kdtree"></param>
        /// <param name="limit">Points outside this radius will be ignored.</param>
        /// <param name="best">If one exists, this is set to the index of the best point found.</param>
        /// <param name="first">[first, last) is a range of point IDs forming a kd-Tree from \see kdTreeBuild()</param>
        /// <param name="last"></param>
        /// <param name="nodeCount">Is incremented each time a node of the tree is visited. 
        ///                         Use this to help programatically verify the log(N) complexity expectation.</param>
        /// <param name="axis">Which axis is the tree's root split by. 
        ///                    The top-most root of the entire tree is split on axis=0 </param>
        static void kdTreeNearest(
            Vector3 x,
            Vector3[] points,
            int[] kdtree,
            ref double limit,
            ref int best,
            int first,
            int last,
            ref int nodeCount,
            int axis = 0)
        {
            int numPoints = last - first;

            if (numPoints <= 0)
                return; // We've gone off the end of the tree.

            // 'mid' is the root of the tree.
            // The left subtree is [first, mid). The right is [mid+1, last).
            int mid = MeshRegistration.midpoint(first, last);

            // Visit this subtree's root/mid-point.
            ++nodeCount;
            Vector3 midpoint = points[kdtree[mid]];
            double distToMid = Vector3.Distance(midpoint, x);
            if (distToMid <= limit)
            {
                best = kdtree[mid];
                limit = distToMid;
            }

            if (numPoints <= 1)
                return; // There are no children.

            // If 'x' is on the left of the cut-plane, then go down the left side first.
            // Afterwards, go down the right side if necessary.
            // Else, do the opposite.
            int nextAxis = (axis + 1) % 3;
            if (x[axis] < midpoint[axis])
            {
                kdTreeNearest(x, points, kdtree, ref limit, ref best, first, mid, ref nodeCount, nextAxis);
                // Note: the call to 'nearest' may have updated 'limit'
                if (midpoint[axis] <= x[axis] + limit)
                {
                    kdTreeNearest(x, points, kdtree, ref limit, ref best, mid + 1, last, ref nodeCount, nextAxis);
                }
            }
            else
            {
                kdTreeNearest(x, points, kdtree, ref limit, ref best, mid + 1, last, ref nodeCount, nextAxis);
                if (x[axis] - limit <= midpoint[axis])
                {
                    kdTreeNearest(x, points, kdtree, ref limit, ref best, first, mid, ref nodeCount, nextAxis);
                }
            }
        }

        public static int[] BuildIndexMap(
            Vector3[] fromVertices,
            Vector3[] toVertices,
            float distTolerance)
        {

            var stopwatch1 = new System.Diagnostics.Stopwatch();
            stopwatch1.Start();

            int[] kdtree = new int[toVertices.Length];
            for (int i = 0; i != toVertices.Length; ++i)
            {
                kdtree[i] = i;
            }
            kdTreeBuild(toVertices, kdtree, 0, toVertices.Length);

            stopwatch1.Stop();

            var stopwatch2 = new System.Diagnostics.Stopwatch();
            stopwatch2.Start();

            var indexMap = new int[fromVertices.Length];
            int countInspectedTargets = 0;
            int totalFailedQueries = 0;
            for (int i = 0; i < fromVertices.Length; ++i)
            {
                double closestDistance = distTolerance;
                int closestVertexId = -1;

                kdTreeNearest(
                    fromVertices[i], toVertices, kdtree, ref closestDistance, ref closestVertexId, 0,
                    toVertices.Length, ref countInspectedTargets);
                indexMap[i] = closestVertexId;
                if (closestVertexId == -1)
                    totalFailedQueries++;
            }

            stopwatch2.Stop();

            bool logStatistics = false;
            if (logStatistics)
            {
                float inspectedPerVertex = countInspectedTargets / (float)fromVertices.Length;
                Debug.Log($"Mesh Registration Stats: Build time: {stopwatch1.ElapsedMilliseconds} ms" +
                    $", Query time: {stopwatch2.ElapsedMilliseconds} ms" +
                    $", Average # of inspected vertices per query: {inspectedPerVertex}" +
                    ", Failed queries: " + totalFailedQueries);
            }

            return indexMap;
        }
    }
}
