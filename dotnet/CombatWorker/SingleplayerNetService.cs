using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Quality;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Platform;

// Match NetSingleplayerGameService, except NetId uses the recorded player's anonymized ID.
// The native service hard-codes 1, which makes LocalContext.GetMe fail on recordings.
// Retaining the recorded ID preserves checkpoint hashes; singleplayer branches depend on Type.
sealed class SingleplayerNetService(ulong netId) : INetGameService
{
    bool _isLoading;

    public ulong NetId => netId;
    public NetGameType Type => NetGameType.Singleplayer;
    public PlatformType Platform => PlatformType.None;
    public bool IsConnected => true;
    public bool IsGameLoading => _isLoading;
    public PeerVersionInfo LocalVersion { get; } = PeerVersionInfo.LocalDefault();

#pragma warning disable CS0067 // never raised, as in the game's singleplayer service
    public event Action<NetErrorInfo>? Disconnected;
#pragma warning restore CS0067

    public void SendMessage<T>(T message, ulong playerId) where T : INetMessage { }
    public void SendMessage<T>(T message) where T : INetMessage { }
    public void RegisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage { }
    public void UnregisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage { }
    public void Update() { }
    public void Disconnect(NetError reason, bool now = false) { }
    public ConnectionStats GetStatsForPeer(ulong peerId) => throw new NotImplementedException();
    public void SetGameLoading(bool isLoading) => _isLoading = isLoading;
    public void SetBufferMessages(bool bufferMessages) { }
    public string? GetRawLobbyIdentifier() => null;
}
