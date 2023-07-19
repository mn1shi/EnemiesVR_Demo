using System.Runtime.CompilerServices;
using UnityEngine;

[assembly: InternalsVisibleTo("Unity.ZivaRTPlayer.Editor")]
namespace Unity.ZivaRTPlayer
{
    /// <summary>
    /// An imported ziva rig asset that can be used in conjunction
    /// with a ziva player component to perform skinning on played
    /// back animations or state machines.
    /// </summary>
    public class ZivaShaderData : ScriptableObject
    {
        [SerializeField]
        internal ComputeShader m_CalculateTangents;
        [SerializeField]
        internal ComputeShader m_ComputeMotionVectors;
        [SerializeField]
        internal ComputeShader m_CopyPositionBuffer;
        [SerializeField]
        internal ComputeShader m_RemapVertices;
        [SerializeField]
        internal ComputeShader m_Skinning;
        [SerializeField]
        internal ComputeShader m_ZivaRT;
    }
}
