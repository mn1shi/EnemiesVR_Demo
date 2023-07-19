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
    public class ZivaRTRig : ScriptableObject
    {
        [SerializeField]
        internal string m_ZrtVersion;
        [SerializeField]
        internal CharacterComponent m_Character;
        [SerializeField]
        internal CorrectiveType m_CorrectiveType;
        [SerializeField]
        internal Patch[] m_Patches;
        [SerializeField]
        internal Skinning m_Skinning;
        [SerializeField]
        internal string[] m_ExtraParameterNames;
    }
}
