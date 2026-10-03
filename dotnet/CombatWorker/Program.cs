// CombatWorker: a persistent process that loads a combat, from a recorded .mcr's initial state or from a spec of
// what the player brings into it, and lets a caller step it one chosen input at a time. One JSON object per line in each direction:
//
//   {"cmd":"load","mcr":"<path>"}           enter the recorded fight's room; returns the first boundary
//   {"cmd":"start","spec":{...}}            enter any fight from a character, deck, relics, HP and encounter
//                                           (CombatSpec.cs); returns the first boundary
//                                           load and start take "hashes":false for training: no state_hash and no
//                                           checkpoints, which cost a full-state serialisation each and feed nothing;
//                                           start takes "reuse_map":true to generate the act map once per spec
//   {"cmd":"step","action":{...}}           play / end_turn / potion / choose; returns the next boundary
//   {"cmd":"observe"}                       the current boundary again, without acting
//   {"cmd":"tape","mcr":"<path>"}           the recorded inputs and checkpoints, decoded (no game state touched)
//   {"cmd":"catalog"}                       the characters (with starting HP and deck) and encounters a spec can name
//   {"cmd":"start_run","spec":{...}}       start a seeded native run (character, seed, ascension?, unlocks?)
//   {"cmd":"run_step","action":{...}}     take the next run-level or combat decision
//   {"cmd":"run_observe"}                   report the current run boundary again
//   {"cmd":"run_combat_snapshot","path":…} write the game's recording of the current fight as an .mcr to `load`
//   {"cmd":"quit"}
//
// Every input goes through the path a human click takes: CardModel.TryManualPlay's body, the end-turn button's
// EndPlayerTurnAction, PotionModel.EnqueueManualUse, and card choices through CardSelectCmd's LocalSelector (choice
// ids reserved, the action paused and resumed by the game). The net service is the normal singleplayer one, so the
// game queues ReadyToBeginEnemyTurnAction and resumes choice-paused actions itself.
//
// Run it from a scratch working directory: user:// paths resolve relative to cwd under GodotStubs.
// Protocol lines go to stdout; everything the game and substrate log goes to stderr.
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

static class Program
{
    static int Main(string[] args)
    {
        Sts2Resolver.Install(typeof(Program).Assembly);
        return Worker.Run();
    }
}

// Separate type so no sts2 type is JIT-resolved before the assembly resolver above is installed.
static class Worker
{
    static readonly JsonSerializerOptions Json = new() { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never };

    public static int Run()
    {
        var protocol = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
        Console.SetOut(Console.Error);
        var boot = Stopwatch.StartNew();
        Substrate.Boot();
        protocol.WriteLine(JsonSerializer.Serialize(new { ok = true, ready = true, boot_ms = boot.Elapsed.TotalMilliseconds }, Json));

        CombatSession? session = null;
        RunSession? runSession = null;
        string? line;
        while ((line = Console.In.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            object response;
            try
            {
                JsonObject req = JsonNode.Parse(line)!.AsObject();
                string cmd = (string?)req["cmd"] ?? throw new ArgumentException("missing cmd");
                if (cmd == "quit") break;
                if (cmd == "start_run") session = null;
                if (cmd is "load" or "start") runSession = null;
                response = cmd switch
                {
                    "load" => (session = CombatSession.Load((string)req["mcr"]!, Hashes(req))).Report("load"),
                    "start" => (session = CombatSession.Start(req["spec"]?.AsObject() ?? throw new ArgumentException("missing spec"), Hashes(req), (bool?)req["reuse_map"] ?? false)).Report("load"),
                    "step" => (session ?? throw new InvalidOperationException("no combat loaded")).Step(req["action"]?.AsObject() ?? throw new ArgumentException("missing action")),
                    "observe" => (session ?? throw new InvalidOperationException("no combat loaded")).Report("observe"),
                    "start_run" => (runSession = RunSession.Start(req["spec"]?.AsObject() ?? throw new ArgumentException("missing spec"))).Report(),
                    "run_step" => (runSession ?? throw new InvalidOperationException("no run started")).Step(req["action"]?.AsObject() ?? throw new ArgumentException("missing action")),
                    "run_observe" => (runSession ?? throw new InvalidOperationException("no run started")).Report(),
                    "run_combat_snapshot" => (runSession ?? throw new InvalidOperationException("no run started")).CombatSnapshot((string)req["path"]!),
                    "tape" => Tape.Read((string)req["mcr"]!),
                    "catalog" => CombatSpec.Catalog(),
                    _ => throw new ArgumentException($"unknown cmd {cmd}"),
                };
            }
            catch (Exception ex)
            {
                Exception inner = ex is System.Reflection.TargetInvocationException { InnerException: { } i } ? i : ex;
                response = new { ok = false, error = $"{inner.GetType().Name}: {inner.Message}" };
            }
            protocol.WriteLine(JsonSerializer.Serialize(response, Json));
        }
        return 0;
    }

    static bool Hashes(JsonObject req) => (bool?)req["hashes"] ?? true;
}
