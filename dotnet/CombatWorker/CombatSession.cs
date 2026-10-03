using System.Diagnostics;
using System.Reflection;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using static Substrate;

// One combat, loaded from a recording's initial state and then driven only by caller-chosen inputs.
sealed class CombatSession
{
    static IDisposable? _selectorScope;
    static IDisposable? _rewardSelectorScope;
    // Error-level lines the game logged since the last reply. TaskHelper.RunSafely and the turn loop log an exception
    // and carry on, so a fight can be broken without anything throwing at the worker.
    static readonly List<string> _gameErrors = new();
    static CombatSession()
    {
        MegaCrit.Sts2.Core.Logging.Log.LogCallback += (level, text, _) =>
        {
            if (level == MegaCrit.Sts2.Core.Logging.LogLevel.Error) lock (_gameErrors) _gameErrors.Add(text);
        };
    }

    readonly RunState _run;
    readonly Player _me;
    readonly CallerSelector _selector = new();
    // Hashing serializes full combat state per reply and action; parity tests enable it, training skips it.
    bool _hashes = true;
    long _loadMs;
    readonly Dictionary<string, double> _loadPhases = new();
    readonly List<object> _checkpoints = new();         // generated since the last report
    readonly List<GameAction> _enqueued = new();         // every action queued since the last input
    readonly HashSet<GameAction> _ours = new();          // the subset the worker queued itself

    CombatSession(RunState run)
    {
        _run = run;
        _me = run.Players[0];
    }

    // Attach to the run's existing state before its first combat-node vote; do not reload or enter a debug room.
    public static CombatSession Attach(RunState run)
    {
        _selectorScope?.Dispose();
        _rewardSelectorScope?.Dispose();
        lock (_gameErrors) _gameErrors.Clear();
        var session = new CombatSession(run);
        RunManager rm = RunManager.Instance;
        rm.ChecksumTracker.IsEnabled = true;
        rm.ChecksumTracker.ChecksumGenerated += (data, context, _) =>
            session._checkpoints.Add(new { id = data.id, hash = data.checksum, context = NormalizeContext(context) });
        RestorePostActionChecksum();
        rm.ActionQueueSet.ActionEnqueued += a => session._enqueued.Add(a);
        rm.CombatStateSynchronizer.IsDisabled = true;
        ChoiceContext.Install();
        _selectorScope = CardSelectCmd.UseSelector(session._selector, localOnly: true);
        return session;
    }

    public void WaitForBoundary() => AwaitBoundary();

    public void SetRewardCardPick(int cardIndex) => _selector.RewardCardPick = cardIndex;
    public void SetRewardAlternative(string optionId) => _selector.RewardAlternative = optionId;
    public static List<string> DrainGameErrors() => TakeGameErrors();
    public static bool HasGameErrors => AnyGameError;
    public bool HasPendingRunChoice => _selector.Pending != null;
    public object? RunChoice => _selector.Pending?.Describe(this);
    public List<object> RunChoiceLegal => _selector.Pending?.LegalPicks() ?? new List<object>();
    public void AnswerRunChoice(JsonObject action)
    {
        if (action["picks"] is not JsonArray picks) throw new ArgumentException("choice needs picks");
        _selector.Answer(picks.Select(p => (int)p!).ToList());
    }

    // CardReward consults CardSelectCmd.Selector, whereas combat choices consult LocalSelector. Keep the global
    // selector scoped to rewards so combat choice ids continue through the game's normal local choice path.
    public void EnableRewardSelector() => _rewardSelectorScope ??= CardSelectCmd.UseSelector(_selector);
    public void DisableRewardSelector() { _rewardSelectorScope?.Dispose(); _rewardSelectorScope = null; }

    static readonly Dictionary<string, SerializableActMap> MapCache = new();

    // A recorded fight: the .mcr's initial state, its id cursors, and the room it was saved in.
    public static CombatSession Load(string mcrPath, bool hashes = true)
    {
        var sw = Stopwatch.StartNew();
        CombatReplay replay = Tape.ReadReplay(mcrPath);
        SerializableRun save = replay.serializableRun;
        return Enter(sw, save, hashes, (rm, _) =>
        {
            // The recording's id cursors are part of its initial state: every id the game hands out from here on
            // (actions, hooks, choices, rewards, checkpoints) continues from where the real run was.
            rm.ActionQueueSet.FastForwardNextActionId(replay.nextActionId);
            rm.ActionQueueSynchronizer.FastForwardHookId(replay.nextHookId);
            typeof(MegaCrit.Sts2.Core.Multiplayer.Game.ChecksumTracker).GetProperty("NextId")!.SetValue(rm.ChecksumTracker, replay.nextChecksumId);
            rm.PlayerChoiceSynchronizer.FastForwardChoiceIds(replay.choiceIds);
            rm.RewardsSetSynchronizer.FastForwardRewardIds(replay.rewardIds);
        }, (rm, run) => rm.LoadIntoLatestMapCoord(AbstractRoom.FromSerializable(save.PreFinishedRoom, run)));
    }

    // Enter through the game's debug room path, retaining seeded encounter RNG.
    // Use an unmarked node of the encounter's type: Fur Coat reads CurrentMapPoint at combat start.
    // reuseMap caches by spec without seed. Map generation uses a separate "act_N_map" RNG stream;
    // test_combat_spec checks cached starts against normal starts at each combat hash.
    public static CombatSession Start(JsonObject spec, bool hashes = true, bool reuseMap = false)
    {
        var sw = Stopwatch.StartNew();
        (SerializableRun save, EncounterModel encounter) = CombatSpec.ToSave(spec);
        double toSave = sw.Elapsed.TotalMilliseconds;
        string? mapKey = null;
        if (reuseMap)
        {
            JsonObject key = spec.DeepClone().AsObject();
            key.Remove("seed");
            mapKey = key.ToJsonString();
            if (MapCache.TryGetValue(mapKey, out SerializableActMap? cached)) save.Acts[save.CurrentActIndex].SavedMap = cached;
        }
        CombatSession session = Enter(sw, save, hashes, (_, run) =>
        {
            if (mapKey != null && !MapCache.ContainsKey(mapKey) && run.Map is { } map)
            {
                if (MapCache.Count >= 256) MapCache.Clear();
                MapCache[mapKey] = SerializableActMap.FromActMap(map);
            }
        }, (rm, run) =>
        {
            MapPointType type = encounter.RoomType switch { RoomType.Elite => MapPointType.Elite, RoomType.Boss => MapPointType.Boss, _ => MapPointType.Monster };
            MapPoint? node = type == MapPointType.Boss ? run.Map.BossMapPoint
                : run.Map.GetAllMapPoints().Where(p => p.PointType == type && p.Quests.Count == 0)
                    .OrderBy(p => p.coord.row).ThenBy(p => p.coord.col).FirstOrDefault();
            return node == null
                ? rm.EnterRoomDebug(encounter.RoomType, type, encounter.ToMutable(), showTransition: false)
                : rm.EnterMapCoordDebug(node.coord, encounter.RoomType, type, encounter.ToMutable(), showTransition: false);
        });
        session._loadPhases["to_save"] = toSave;
        return session;
    }

    static CombatSession Enter(Stopwatch sw, SerializableRun save, bool hashes, Action<RunManager, RunState> beforeRoom, Func<RunManager, RunState, Task> enterRoom)
    {
        var phases = new Dictionary<string, double>();
        double mark = sw.Elapsed.TotalMilliseconds;
        void Phase(string name) { double now = sw.Elapsed.TotalMilliseconds; phases[name] = now - mark; mark = now; }
        if (RunManager.Instance.IsInProgress) RunManager.Instance.CleanUp(graceful: true);
        _selectorScope?.Dispose(); // CardSelectCmd.Reset leaves selectors alone while TestMode is on
        _rewardSelectorScope?.Dispose();
        _rewardSelectorScope = null;
        RewardsSet.testSelector = null;
        BundleSelector.Deactivate();
        lock (_gameErrors) _gameErrors.Clear();

        Phase("clean_up");
        RunState run = RunState.FromSerializable(save);
        var session = new CombatSession(run) { _hashes = hashes };
        Phase("from_save");
        SetUpSingleplayer(run, save);
        Phase("set_up");
        RunManager rm = RunManager.Instance;
        // TestMode.IsOn (set by HeadlessInit) disables the game's checksum tracker; turn it back on, unless hashes are off.
        rm.ChecksumTracker.IsEnabled = hashes;
        rm.ChecksumTracker.ChecksumGenerated += (data, context, full) =>
            session._checkpoints.Add(Environment.GetEnvironmentVariable("STS2_DUMP") == null
                ? new { id = data.id, hash = data.checksum, context = NormalizeContext(context) }
                : new { id = data.id, hash = data.checksum, context = NormalizeContext(context), dump = full.ToString() });
        if (hashes) RestorePostActionChecksum();
        rm.ActionQueueSet.ActionEnqueued += a => session._enqueued.Add(a);
        rm.CombatStateSynchronizer.IsDisabled = true;
        rm.Launch(); // sets LocalContext.NetId from the net service
        Await(rm.GenerateMap(), "GenerateMap");
        Phase("map");
        beforeRoom(rm, run);
        ChoiceContext.Install();
        _selectorScope = CardSelectCmd.UseSelector(session._selector, localOnly: true);
        Await(enterRoom(rm, run), "room entry");
        Phase("room");
        // Combat start runs under TaskHelper.RunSafely: if it throws, the error is logged and the executor stays paused.
        WaitFor(() => !rm.ActionExecutor.IsPaused || AnyGameError);
        if (rm.ActionExecutor.IsPaused)
            throw new InvalidOperationException("the combat did not start: " + (AnyGameError ? FirstLine(LastGameError()) : "timed out waiting for the action executor"));
        session.AwaitBoundary();
        Phase("first_boundary");
        session._loadMs = sw.ElapsedMilliseconds;
        foreach ((string k, double v) in phases) session._loadPhases[k] = v;
        return session;
    }

    // RunManager.SetUpSavedSingleplayer minus SaveManager.IncrementNumReloads (a save-profile write) and with saving
    // off. The net service is singleplayer; SetUpReplay would install NetReplayGameService, which makes the game stop
    // queueing ReadyToBeginEnemyTurnAction and route local choices to a remote wait.
    static void SetUpSingleplayer(RunState run, SerializableRun save)
    {
        RunManager rm = RunManager.Instance;
        const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        typeof(RunManager).GetProperty("State", Private)!.SetValue(rm, run);
        var net = new SingleplayerNetService(run.Players[0].NetId);
        typeof(RunManager).GetMethod("InitializeShared", Private)!.Invoke(rm, new object?[]
            { net, new PeerInputSynchronizer(net), false, save.DailyTime, save.StartTime, save.RunTime, save.WinTime, save.NumReloads });
        var players = run.Players.Select(p => new RunLobbyPlayer { id = p.NetId, isModded = net.LocalVersion.IsModded() }).ToList();
        typeof(RunManager).GetMethod("InitializeRunLobby", Private)!.Invoke(rm, new object?[] { net, run, players });
        typeof(RunManager).GetMethod("InitializeSavedRun", Private)!.Invoke(rm, new object?[] { save });
    }

    // ---- stepping ----

    public object Step(JsonObject action)
    {
        var t0 = Stopwatch.GetTimestamp();
        string boundary = Boundary();
        string type = (string?)action["type"] ?? throw new ArgumentException("action needs a type");
        if (boundary == "terminal") throw new InvalidOperationException("combat is over");
        if (boundary == "awaiting_choice" && type != "choose") throw new InvalidOperationException($"a choice is pending; got {type}");
        if (boundary == "awaiting_input" && type == "choose") throw new InvalidOperationException("no choice is pending");
        // The action paused on a pending choice resumes after the answer; the step waits for it too.
        List<GameAction> paused = _enqueued.Where(a => a.State == GameActionState.GatheringPlayerChoice).ToList();
        _enqueued.Clear();
        _ours.Clear();
        if (type == "choose") _enqueued.AddRange(paused);
        switch (type)
        {
            case "play": Play(Int(action, "hand")!.Value, Int(action, "target")); break;
            case "end_turn": Enqueue(new EndPlayerTurnAction(_me, _me.PlayerCombatState!.TurnNumber)); break;
            case "potion": Potion(Int(action, "slot")!.Value, Int(action, "target")); break;
            case "choose": _selector.Answer(action["picks"]?.AsArray().Select(n => (int)n!).ToList() ?? throw new ArgumentException("choose needs picks")); break;
            default: throw new ArgumentException($"unknown action type {type}");
        }
        AwaitBoundary();
        return Report("step", Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
    }

    // CardModel.TryManualPlay, with the action kept so the step can wait for it to resolve.
    void Play(int hand, int? targetSlot)
    {
        IReadOnlyList<CardModel> cards = _me.PlayerCombatState!.Hand.Cards;
        if (hand < 0 || hand >= cards.Count) throw new ArgumentException($"hand index {hand} out of range (hand has {cards.Count})");
        CardModel card = cards[hand];
        Creature? target = TargetFor(targetSlot);
        if (!card.CanPlayTargeting(target)) throw new ArgumentException($"{card.Id} cannot be played at target {targetSlot?.ToString() ?? "none"}");
        TaskHelper.RunSafely(card.OnEnqueuePlayVfx(target));
        Enqueue(new PlayCardAction(card, target));
    }

    // PotionModel.EnqueueManualUse builds and queues the action itself; ActionEnqueued tells us which one it was.
    void Potion(int slot, int? targetSlot)
    {
        PotionModel potion = _me.GetPotionAtSlotIndex(slot) ?? throw new ArgumentException($"no potion in slot {slot}");
        Creature? target = TargetFor(targetSlot);
        if (!PotionTargets(potion).Contains(target)) throw new ArgumentException($"{potion.Id} cannot be used at target {targetSlot?.ToString() ?? "none"}");
        int before = _enqueued.Count;
        potion.EnqueueManualUse(target);
        foreach (GameAction a in _enqueued.Skip(before).OfType<UsePotionAction>()) _ours.Add(a);
    }

    void Enqueue(GameAction action)
    {
        _ours.Add(action);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);
    }

    Creature? TargetFor(int? slot)
    {
        if (slot == null) return null;
        IReadOnlyList<Creature> enemies = State.Enemies;
        if (slot < 0 || slot >= enemies.Count) throw new ArgumentException($"target slot {slot} out of range ({enemies.Count} enemies)");
        return enemies[slot.Value];
    }

    // ---- boundaries ----

    CombatState State => CombatManager.Instance.DebugOnlyGetState() ?? throw new InvalidOperationException("no combat state");

    bool Terminal => !CombatManager.Instance.IsInProgress || _me.Creature.IsDead;

    // Every action queued since the last input has resolved (finished, cancelled, or paused on the pending choice).
    bool InputResolved => _enqueued.All(a => a.State is GameActionState.Finished or GameActionState.Canceled
        || (a.State == GameActionState.GatheringPlayerChoice && _selector.Pending != null));

    bool AtPlayerDecision =>
        InputResolved
        && RunManager.Instance.ActionExecutor.FinishedExecutingActions().IsCompleted
        && !CombatManager.Instance.EndingPlayerTurnPhaseOne && !CombatManager.Instance.EndingPlayerTurnPhaseTwo
        && State.CurrentSide == CombatSide.Player
        && RunManager.Instance.ActionQueueSynchronizer.CombatState == ActionSynchronizerCombatState.PlayPhase
        && !CombatManager.Instance.IsPlayerReadyToEndTurn(_me);

    string Boundary() => Terminal ? "terminal" : _selector.Pending != null ? "awaiting_choice" : AtPlayerDecision ? "awaiting_input" : "running";

    // Returns only at a decision boundary. A new turn number is never a boundary by itself: after an end turn the
    // enemy turn and the next player turn start both run to completion inside this wait.
    void AwaitBoundary()
    {
        if (!WaitFor(() => Boundary() != "running" || TurnLoopDied))
            throw new TimeoutException("timed out waiting for a decision boundary: " + string.Join(", ",
                $"queued=[{string.Join("; ", _enqueued.Select(a => $"{a.GetType().Name}:{a.State}"))}]",
                $"queue_drained={RunManager.Instance.ActionExecutor.FinishedExecutingActions().IsCompleted}",
                $"ending_p1={CombatManager.Instance.EndingPlayerTurnPhaseOne}", $"ending_p2={CombatManager.Instance.EndingPlayerTurnPhaseTwo}",
                $"side={State.CurrentSide}", $"sync={RunManager.Instance.ActionQueueSynchronizer.CombatState}",
                $"ready_to_end={CombatManager.Instance.IsPlayerReadyToEndTurn(_me)}"));
        if (TurnLoopDied && Boundary() == "running")
            throw new InvalidOperationException("the game's combat turn loop died: " + FirstLine(LastGameError()));
        if (Terminal) WaitFor(() => RunManager.Instance.ActionExecutor.FinishedExecutingActions().IsCompleted, 5_000);
    }

    // CombatManager logs this when its turn loop throws; the combat can then never reach another boundary.
    static bool TurnLoopDied { get { lock (_gameErrors) return _gameErrors.Any(e => e.Contains("turn loop died")); } }
    static bool AnyGameError { get { lock (_gameErrors) return _gameErrors.Count > 0; } }
    static string LastGameError() { lock (_gameErrors) return _gameErrors.LastOrDefault(e => e.Contains("turn loop died")) ?? _gameErrors.LastOrDefault() ?? ""; }
    static string FirstLine(string s) => s.Split('\n')[0].Trim();

    // ---- reporting ----

    public object Report(string what, double? stepMs = null)
    {
        string boundary = Boundary();
        var checkpoints = _checkpoints.ToList();
        _checkpoints.Clear();
        return new
        {
            ok = true,
            boundary,
            state_hash = _hashes ? Hash(NetFullCombatState.FromRun(_run, null)) : (uint?)null,
            obs = Observe(),
            legal = boundary switch { "awaiting_input" => LegalInputs(), "awaiting_choice" => _selector.Pending!.LegalPicks(), _ => new List<object>() },
            choice = _selector.Pending?.Describe(this),
            checkpoints,
            enqueued_by_game = _enqueued.Where(a => !_ours.Contains(a)).Select(a => a.GetType().Name).ToList(),
            game_errors = TakeGameErrors(),
            ms = what == "load" ? (object)new { load = _loadMs, phases = _loadPhases.ToDictionary(kv => kv.Key, kv => Math.Round(kv.Value, 2)) } : new { step = stepMs },
        };
    }

    static List<string> TakeGameErrors()
    {
        lock (_gameErrors)
        {
            var first = _gameErrors.Select(FirstLine).ToList();
            _gameErrors.Clear();
            return first;
        }
    }

    List<object> LegalInputs()
    {
        var legal = new List<object>();
        IReadOnlyList<CardModel> hand = _me.PlayerCombatState!.Hand.Cards;
        IReadOnlyList<Creature> enemies = State.Enemies;
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel c = hand[i];
            if (c.CanPlayTargeting(null)) legal.Add(new { type = "play", hand = i, target = (int?)null });
            for (int s = 0; s < enemies.Count; s++)
                if (c.CanPlayTargeting(enemies[s])) legal.Add(new { type = "play", hand = i, target = (int?)s });
        }
        for (int slot = 0; slot < _me.PotionSlots.Count; slot++)
        {
            PotionModel? p = _me.PotionSlots[slot];
            if (p == null) continue;
            foreach (Creature? t in PotionTargets(p))
                legal.Add(new { type = "potion", slot, target = t == null ? null : (int?)IndexOf(enemies, t) });
        }
        legal.Add(new { type = "end_turn" });
        return legal;
    }

    // What NPotionHolder.UsePotion would let a human do: enemy-targeted potions pick a living enemy, self-targeted
    // ones are aimed at the owner by EnqueueManualUse, the rest take no target. Ally throws need a second player.
    List<Creature?> PotionTargets(PotionModel p)
    {
        if (p.IsQueued || p.Usage is not (PotionUsage.CombatOnly or PotionUsage.AnyTime)) return new();
        if (p.TargetType == TargetType.AnyEnemy) return State.Enemies.Where(e => e.IsAlive && p.IsValidTarget(e)).Cast<Creature?>().ToList();
        return p.IsValidTarget(null) || p.IsValidTarget(_me.Creature) ? new() { null } : new();
    }

    static int IndexOf(IReadOnlyList<Creature> list, Creature c)
    {
        for (int i = 0; i < list.Count; i++) if (list[i] == c) return i;
        return -1;
    }

    object Observe()
    {
        CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
        PlayerCombatState? pcs = _me.PlayerCombatState;
        Creature body = _me.Creature;
        return new
        {
            turn = pcs?.TurnNumber,
            round = cs?.RoundNumber,
            side = cs?.CurrentSide.ToString(),
            encounter = cs?.Encounter?.Id.ToString(),
            player = new
            {
                character = _me.Character.Id.ToString(),
                hp = body.CurrentHp, max_hp = body.MaxHp, block = body.Block,
                energy = pcs?.Energy, max_energy = pcs?.MaxEnergy, stars = pcs?.Stars,
                // Necrobinder's Osty: HP only. It never has block, and the powers that work through it are the player's.
                osty = _me.Osty is { } osty ? new { hp = osty.CurrentHp, max_hp = osty.MaxHp, alive = osty.IsAlive } : null,
                powers = Powers(body),
                orbs = pcs?.OrbQueue.Orbs.Select(o => new { id = o.Id.ToString(), passive = (int)o.PassiveVal, evoke = (int)o.EvokeVal }).ToList(),
                orb_capacity = pcs?.OrbQueue.Capacity,
                potions = _me.PotionSlots.Select(p => p?.Id.ToString()).ToList(),
                // What the relic bar shows: the counter when the relic displays one, and whether it is spent.
                relics = _me.Relics.Select(r => new { id = r.Id.ToString(), counter = r.ShowCounter ? r.DisplayAmount : (int?)null, used_up = r.IsUsedUp }).ToList(),
            },
            hand = pcs?.Hand.Cards.Select(Card).ToList(),
            // Draw order is hidden from a player; expose it as a multiset, sorted so the order carries nothing.
            draw = Multiset(pcs?.DrawPile.Cards),
            discard = Multiset(pcs?.DiscardPile.Cards),
            exhaust = Multiset(pcs?.ExhaustPile.Cards),
            this_turn = ThisTurn(cs, body),
            enemies = cs?.Enemies.Select((e, slot) => new
            {
                slot, combat_id = e.CombatId, id = e.Monster?.Id.ToString(),
                hp = e.CurrentHp, max_hp = e.MaxHp, block = e.Block, alive = e.IsAlive,
                powers = Powers(e),
                intent = e.IsAlive && e.Monster != null ? Intent(e, cs) : null,
            }).ToList(),
        };
    }

    internal static object Card(CardModel c) => new
    {
        id = c.Id.ToString(),
        combat_card = c.IsMutable && c.Pile is { IsCombatPile: true } ? NetCombatCard.FromModel(c).CombatCardIndex : (uint?)null,
        upgrades = c.CurrentUpgradeLevel,
        cost = c.EnergyCost.CostsX ? (int?)null : c.EnergyCost.GetAmountToSpend(),
        x_cost = c.EnergyCost.CostsX,
        type = c.Type.ToString(),
        target = c.TargetType.ToString(),
        playable = c.CanPlay(),
        enchantment = c.Enchantment is EnchantmentModel e ? new { id = e.Id.ToString(), amount = e.Amount } : null,
        vars = Vars(c),
    };

    static List<object>? Multiset(IEnumerable<CardModel>? cards) => cards?
        .OrderBy(c => c.Id.ToString(), StringComparer.Ordinal).ThenBy(c => c.CurrentUpgradeLevel)
        .ThenBy(c => c.Enchantment?.Id.ToString(), StringComparer.Ordinal).ThenBy(c => c.EnergyCost.GetAmountToSpend())
        .Select(Card).ToList();

    // What a player with perfect memory knows of this turn: the cards it has played and drawn so far. The game answers
    // its own once-per-turn questions from this history (Iteration counts this turn's CardDrawnEntry), so it stands in
    // for a flag per effect. A card played several times in one series (Burst, Echo Form) counts once.
    static object? ThisTurn(CombatState? cs, Creature body)
    {
        if (cs == null) return null;
        var turn = CombatManager.Instance.History.Entries.Where(e => e.HappenedThisTurn(cs) && e.Actor == body).ToList();
        return new
        {
            played = Multiset(turn.OfType<CardPlayFinishedEntry>().Where(e => e.CardPlay.IsFirstInSeries).Select(e => e.CardPlay.Card)),
            drawn = Multiset(turn.OfType<CardDrawnEntry>().Select(e => e.Card)),
        };
    }

    // The numbers the card's text shows. In hand they are the game's untargeted preview (strength, weak, relics...),
    // computed as NCard computes them for display; PreviewValue is display-only and never feeds game state.
    static Dictionary<string, int> Vars(CardModel c)
    {
        if (c.IsMutable && c.Pile?.Type == PileType.Hand) c.UpdateDynamicVarPreview(CardPreviewMode.Normal, null, c.DynamicVars);
        return c.DynamicVars.Values.ToDictionary(v => v.Name, v => (int)(c.Pile?.Type == PileType.Hand ? v.PreviewValue : v.EnchantedValue));
    }

    static List<object> Powers(Creature c) => c.Powers.Select(p => (object)new { id = p.Id.ToString(), amount = p.Amount }).ToList();

    static object Intent(Creature e, CombatState cs)
    {
        MoveState move = e.Monster!.NextMove;
        IReadOnlyList<Creature> targets = cs.PlayerCreatures;
        return new
        {
            move = move.StateId,
            intents = move.Intents.Select(i => i is AttackIntent a
                ? (object)new { type = i.IntentType.ToString(), damage = a.GetSingleDamage(targets, e), hits = Math.Max(1, a.Repeats) }
                : new { type = i.IntentType.ToString() }).ToList(),
        };
    }

    static int? Int(JsonObject o, string key) => o[key] is JsonNode n ? (int)n : null;

    // ---- choices ----

    // CardSelectCmd's LocalSelector: consulted only after the game has reserved the choice id and paused the action,
    // exactly where the hand-select or grid-select screen would open. The step returns while this task is pending.
    sealed class CallerSelector : ICardSelector
    {
        public PendingChoice? Pending { get; private set; }
        public int? RewardCardPick { get; set; }
        public string? RewardAlternative { get; set; }

        public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
        {
            Pending = new PendingChoice(options.ToList(), minSelect, maxSelect, ChoiceContext.Take());
            return Pending.Result.Task;
        }

        // The card reward screen's answer. It is asked synchronously, and asked again after an alternative that keeps
        // the screen open (a reroll) or a pick that allows another: with nothing supplied, the answer is "close the
        // screen", as a player may. A rerolled reward then stays in its set with its new cards, to be taken again.
        public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        {
            if (RewardAlternative is string optionId)
            {
                RewardAlternative = null;
                return new CardRewardSelection { alternative = alternatives.FirstOrDefault(a => a.OptionId == optionId)
                    ?? throw new ArgumentException($"card reward has no {optionId} option") };
            }
            if (RewardCardPick is not int index) return default;
            RewardCardPick = null;
            if (index < 0 || index >= options.Count)
                throw new ArgumentException($"card reward index {index} out of range ({options.Count} cards)");
            return new CardRewardSelection { card = options[index].Card };
        }

        public void Answer(List<int> picks)
        {
            PendingChoice p = Pending ?? throw new InvalidOperationException("no choice is pending");
            if (picks.Count < p.Min || picks.Count > p.Max) throw new ArgumentException($"pick between {p.Min} and {p.Max} options, got {picks.Count}");
            if (picks.Distinct().Count() != picks.Count || picks.Any(i => i < 0 || i >= p.Options.Count)) throw new ArgumentException("picks must be distinct option indexes");
            Pending = null;
            p.Result.SetResult(picks.Select(i => p.Options[i]).ToList());
        }
    }

    sealed record PendingChoice(List<CardModel> Options, int Min, int Max, ChoiceContext.Context? Context)
    {
        public TaskCompletionSource<IEnumerable<CardModel>> Result { get; } = new();

        // screen: the CardSelectCmd method; prompt: the game's prompt key; source: the model that asked, or else the
        // card whose play is paused on the choice.
        public object Describe(CombatSession session) => new
        {
            min = Min, max = Max, options = Options.Select(Card).ToList(),
            screen = Context?.Screen, prompt = Context?.Prompt,
            source = Context?.Source ?? session._enqueued.OfType<PlayCardAction>()
                .FirstOrDefault(a => a.State == GameActionState.GatheringPlayerChoice)?.CardModelId.ToString(),
        };

        // Enumerated only when a single pick is asked for; multi-picks are described by min/max over the options.
        public List<object> LegalPicks()
        {
            if (Max != 1) return new();
            var picks = Enumerable.Range(0, Options.Count).Select(i => (object)new { type = "choose", picks = new[] { i } }).ToList();
            if (Min == 0) picks.Add(new { type = "choose", picks = Array.Empty<int>() });
            return picks;
        }
    }
}
