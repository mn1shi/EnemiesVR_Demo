#define MOTION_VECTORS

using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.ZivaRTPlayer
{
    /// <summary>
    /// The selection mode for if tangent frames should be recomputed
    /// </summary>
    public enum RecomputeTangentFrames
    {
        /// <summary>
        /// Do not recompute tangent frame
        /// </summary>
        None,
        /// <summary>
        /// Recompute only the normals portion of the tangent frame
        /// </summary>
        NormalsOnly,
        /// <summary>
        /// Recompute both the normals and tangents of the tangent frame
        /// </summary>
        NormalsAndTangents,
    }

    internal abstract class Solver : IDisposable
    {
        // For a custom solver, override the Init() but call the base class Init() function too.
        public virtual bool Init(
            ZivaRTRig rig,
            Mesh targetMesh,
            int[] vertexIndexMap,
            MeshTangentFramesInfo tangentFramesInfo,
            RecomputeTangentFrames recomputeTangentFrames,
            bool calculateMotionVectors,
            ZivaShaderData shaderData)
        {
            m_Mesh = targetMesh;

            // If there are no UVs we cannot calculate tangents. This is the best place I can find for this
            // for now, really the tangent code needs a refactor I think.
            if (targetMesh)
            {
                HasValidUVs = targetMesh.uv.Length > 0;
            }

            Assert.IsNotNull(shaderData);

            return m_Mesh != null;
        }

        // Starts an asynchronous solve, should be overriden by the custom solver.
        // Once the solve is started neither this class nor the arguments to this function
        // should be modified. See WaitOnAsyncSolve()
        public abstract void StartAsyncSolve(
            JobSolver.JobFuture<float> currentPose,
            JobSolver.JobFuture<float> jointWorldTransforms,
            RecomputeTangentFrames recomputeTangentFrames,
            bool doCorrectives,
            bool doSkinning);

        // Methods for accessing ZivaRT computed vertex attributes. Can also be accessed via
        // The sharedMesh component of the target mesh, but sharedMesh component is accessable
        // to other systems outside of ZivaRT, so if we want the actual ZivaRT results we need
        // to access them through these methods. Compute Shader results can only be accessed
        // through sharedMesh.
        public abstract void GetPositions(Vector3[] positions);
        public abstract void GetNormals(Vector3[] normals);
        public abstract void GetTangents(Vector4[] tangents);        
        public abstract void GetMotionVectors(Vector3[] motionVectors);

        // WaitOnAsyncSolve() must be called for each StartAsyncSolve() before the next
        // StartAsyncSolve() is called. Both function functions should be called on the main
        // thread.
        public abstract void WaitOnAsyncSolve();

        public abstract void Dispose();

        public bool HasValidUVs = false;
        protected Mesh m_Mesh;
#if MOTION_VECTORS
        protected bool m_IsFirstTime;
#endif
        public virtual NativeArray<float3x4> RelativeTransforms
        {
            get;
        }
    }
}
