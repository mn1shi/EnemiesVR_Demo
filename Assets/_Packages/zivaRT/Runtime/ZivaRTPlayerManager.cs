using System.Collections.Generic;
using Unity.ZivaRTPlayer;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.PlayerLoop;

// Manager to schedule and handle updates for Ziva.
// We want to do this instead of going via the normal
// update system so we can manage visibility and similar
// on a system wide level instead of needing each component
// to manage itself.
class ZivaRTPlayerManager
{
    static ZivaRTPlayerManager s_Instance;
    public static ZivaRTPlayerManager Instance => s_Instance ??= new ZivaRTPlayerManager();

    List<ZivaRTPlayer> m_RegisteredPlayers = new List<ZivaRTPlayer>();

    ZivaShaderData m_ShaderData = null; 

    internal ZivaShaderData ShaderData
    {
        get
        {
            // Do this here, because under certain circumstances Unity can unload
            // the shader data resource.
            if( m_ShaderData == null)
            {
                m_ShaderData = Resources.Load<ZivaShaderData>("ZivaShaderData");
                Debug.Assert(m_ShaderData != null);
            }
            return m_ShaderData;
        }
    }

    public void RegisterPlayer(ZivaRTPlayer player)
    {
        if (player == null || m_RegisteredPlayers.Contains(player))
            return;

        m_RegisteredPlayers.Add(player);
    }

    public void UnregisterPlayer(ZivaRTPlayer player)
    {
        if (player == null)
            return;

        m_RegisteredPlayers.Remove(player);
        m_WasVisisble.Remove(player);
    }

    void PostUpdate()
    {
        foreach (var player in m_RegisteredPlayers)
            player.ResetForFrame();

        foreach (var player in m_WasVisisble)
            player.ZivaUpdate();
    }

    void PreLateUpdate()
    {
        foreach (var player in m_WasVisisble)
            player.ZivaLateUpdate();

        m_WasVisisble.Clear();
    }

    ZivaRTPlayerManager()
    {
        // This places our hooks in the playerloop. Note that we need to make *new* entries
        // as using the delegates on existing entries may look like it works but is actually
        // undefined behaviour (may be different standalone / editor, may cause hangs etc.)

        // Note: all entries are registrered with this.GetType() as type. This is just to enable
        // us to find the entries we inserted when we want to unregister.
        var rootLoop = PlayerLoop.GetCurrentPlayerLoop();
        var newList = new List<PlayerLoopSystem>();
        for (int i = 0; i < rootLoop.subSystemList.Length; i++)
        {
            var type = rootLoop.subSystemList[i].type;

            newList.Add(rootLoop.subSystemList[i]);

            // Add updates we want BEFORE late update but after the unity PreLateUpdate systems.
            if (type == typeof(PreLateUpdate))
            {
                PlayerLoopSystem s = default;
                s.updateDelegate += PreLateUpdate;
                s.type = GetType();
                newList.Add(s);
            }

            // Add updates we want AFTER the built in ones.
            if (type == typeof(Update))
            {
                PlayerLoopSystem s = default;
                s.updateDelegate += PostUpdate;
                s.type = GetType();
                newList.Add(s);
            }
        }

        rootLoop.subSystemList = newList.ToArray();
        PlayerLoop.SetPlayerLoop(rootLoop);
    }

    // we don't call this (once ziva is registered we leave it in the player loop)
    // but leave this here for reference in case we want to move it in the future.
    void UnregisterUpdates()
    {
        var rootLoop = PlayerLoop.GetCurrentPlayerLoop();
        var newList = new List<PlayerLoopSystem>();
        for (int i = 0; i < rootLoop.subSystemList.Length; i++)
        {
            // Remove all entries coming from us
            var type = rootLoop.subSystemList[i].type;
            if (type != GetType())
                newList.Add(rootLoop.subSystemList[i]);
        }

        rootLoop.subSystemList = newList.ToArray();
        PlayerLoop.SetPlayerLoop(rootLoop);
    }

    List<ZivaRTPlayer> m_WasVisisble = new List<ZivaRTPlayer>();
    public void RegisterWasVisible(ZivaRTPlayer zivaRTPlayer)
    {
        if (zivaRTPlayer != null && !m_WasVisisble.Contains(zivaRTPlayer))
            m_WasVisisble.Add(zivaRTPlayer);
    }
}
