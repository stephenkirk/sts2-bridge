// Shared by ReplayCheck and CombatWorker: boot the headless substrate (GodotStubs + IL-patched sts2.dll + HeadlessInit) with the three harness fixes
// (net-type registries, re-enabled checksum tracker, restored post-action checkpoints), the headless guards for game
// code that assumes a UI, and the helpers both drivers use to hash states and wait on the game's own async loop.
using System.Diagnostics;
using System.IO.Hashing;
using System.Reflection;
using System.Runtime.Loader;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;

// No sts2 types in here: it runs before the resolver can find sts2.dll.
static class Sts2Resolver
{
    public static string LibraryPath { get; private set; } = "";

    public static void Install(Assembly entry)
    {
        string lib = Environment.GetEnvironmentVariable("STS2_LIB")
            ?? entry.GetCustomAttributes<AssemblyMetadataAttribute>().Single(a => a.Key == "Sts2Lib").Value!;
        LibraryPath = Path.GetFullPath(lib);
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            string p = Path.Combine(LibraryPath, name.Name + ".dll");
            return File.Exists(p) ? ctx.LoadFromAssemblyPath(Path.GetFullPath(p)) : null;
        };
    }
}

static class Substrate
{
    public const int StepTimeoutMs = 30_000;

    // HeadlessInit: TestMode.IsOn=true, inline SynchronizationContext, Harmony patches (Cmd.Wait, TalkCmd, localization,
    // Neutralize), ModelDb, ModelIdSerializationCache. It never ran the OneTimeInitialization net registries; recorded
    // net actions need them.
    public static void Boot()
    {
        HeadlessInit.Run();
        MegaCrit.Sts2.Core.Multiplayer.Serialization.MessageTypes.Initialize();
        MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionTypes.Initialize();
        HeadlessGuards.Apply();
    }

    // TestMode.IsOn makes NonInteractiveMode.IsActive true, and ActionExecutor's non-interactive branch never
    // subscribes GameAction.JustBeforeFinished, so RunManager.SendPostActionChecksum never fires. Re-attach the
    // same logic to the same event (raised inside GameAction.Execute just before completion).
    public static void RestorePostActionChecksum()
    {
        var attached = new HashSet<GameAction>();
        RunManager.Instance.ActionExecutor.BeforeActionExecuted += action =>
        {
            if (!attached.Add(action)) return; // resumed after a player choice: already attached
            action.JustBeforeFinished += a =>
            {
                if (a.State == GameActionState.Finished && CombatManager.Instance.IsInProgress
                    && a is not EndPlayerTurnAction && a is not ReadyToBeginEnemyTurnAction)
                    RunManager.Instance.ChecksumTracker.GenerateChecksum($"finished action execution {a}", a);
            };
        };
    }

    public static uint Hash(NetFullCombatState s)
    {
        var w = new PacketWriter();
        w.Write(s);
        w.ZeroByteRemainder();
        return XxHash32.HashToUInt32(w.Buffer.AsSpan(0, w.BytePosition));
    }

    // PlayCardAction.ToString embeds a per-process card instance number, e.g. "CARD.MODDED (37885600)".
    public static string NormalizeContext(string? c) => System.Text.RegularExpressions.Regex.Replace(c ?? "", @"\(\d+\)", "(#)");

    // Each poll is a frame: first the deferred calls Godot would flush at idle time (GameAction.Cancel completes its
    // task that way; see ../GodotStubs/Types.cs), then the condition. The game's own work finishes on pool threads
    // within microseconds, so the first polls only yield; sleeping 1 ms each time was most of a step.
    public static bool WaitFor(Func<bool> cond, int timeoutMs = StepTimeoutMs)
    {
        var sw = Stopwatch.StartNew();
        while (true)
        {
            Godot.DeferredCalls.Flush();
            if (cond()) return true;
            if (sw.ElapsedMilliseconds > timeoutMs) return false;
            if (sw.ElapsedMilliseconds < 2) Thread.Yield();
            else Thread.Sleep(1);
        }
    }

    public static void Await(Task t, string what)
    {
        Require(WaitFor(() => t.IsCompleted), what);
        t.GetAwaiter().GetResult();
    }

    public static void Require(bool ok, string what) { if (!ok) throw new TimeoutException($"timed out waiting for {what}"); }
}

// Game code that dereferences a UI singleton without the null check the rest of the game uses. Headless the node is
// null, so the call throws inside the turn loop and the fight is stuck. Each guard keeps the method's gameplay effect
// and drops only the UI line. Found by random-policy sweeps over real decks; see the worker README.
static class HeadlessGuards
{
    public static void Apply()
    {
        var harmony = new HarmonyLib.Harmony("combat-parity.headless-guards");
        MethodInfo afterDeath = typeof(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus).GetMethod("AfterDeath", BindingFlags.NonPublic | BindingFlags.Instance)!;
        harmony.Patch(afterDeath, prefix: new HarmonyLib.HarmonyMethod(typeof(HeadlessGuards).GetMethod(nameof(SoulNexusAfterDeath), BindingFlags.NonPublic | BindingFlags.Static)));
        var audio = typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager);
        harmony.Patch(audio.GetProperty("Instance")!.GetGetMethod(), postfix: Guard(nameof(SilentDebugAudio)));
        harmony.Patch(audio.GetMethod("Play")!, prefix: Guard(nameof(SkipPlay)));
        harmony.Patch(audio.GetMethod("Stop")!, prefix: Guard(nameof(Skip)));
        harmony.Patch(audio.GetMethod("StopAll")!, prefix: Guard(nameof(Skip)));
    }

    static HarmonyLib.HarmonyMethod Guard(string name) =>
        new(typeof(HeadlessGuards).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static));

    // NDebugAudioManager.Instance is NGame.Instance?.DebugAudio, null headless. Event and rest-site code
    // (JungleMazeAdventure, DenseVegetation, DollRoom, DigRestSiteOption) calls .Play/.Stop on it without `?.`, so the
    // option's task dies before its gameplay lines run. Headless it is a silent stand-in: sound only, no state.
    static MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager? _silentAudio;
    static void SilentDebugAudio(ref MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager? __result) =>
        __result ??= _silentAudio ??= (MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager)
            System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager));
    static bool SkipPlay(ref int __result) { __result = 0; return false; }
    static bool Skip() => false;

    // SoulNexus.AfterDeath: `Creature.Died -= AfterDeath;` then NCombatRoom.Instance.GetCreatureNode(...) with no `?.`.
    static bool SoulNexusAfterDeath(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus __instance, MethodBase __originalMethod)
    {
        if (MegaCrit.Sts2.Core.Nodes.Rooms.NCombatRoom.Instance != null) return true;
        __instance.Creature.Died -= (Action<MegaCrit.Sts2.Core.Entities.Creatures.Creature>)Delegate.CreateDelegate(
            typeof(Action<MegaCrit.Sts2.Core.Entities.Creatures.Creature>), __instance, (MethodInfo)__originalMethod);
        return false;
    }
}
