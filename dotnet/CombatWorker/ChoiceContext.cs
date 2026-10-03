using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Models;

// Capture selection method, prompt key, and source before the selector receives options and pick counts.
// Harmony prefixes read arguments without changing selection behavior.
static class ChoiceContext
{
    static bool _installed;
    static Context? _next;

    public sealed record Context(string Screen, string? Prompt, string? Source);

    public static void Install()
    {
        if (_installed) return;
        var harmony = new Harmony("combat-parity.choice-context");
        var prefix = new HarmonyMethod(typeof(ChoiceContext).GetMethod(nameof(Prefix), BindingFlags.Static | BindingFlags.NonPublic)!);
        // FromChooseABundleScreen is Neow's pack choice, which BundleSelector answers.
        foreach (MethodInfo m in typeof(CardSelectCmd).GetMethods(BindingFlags.Static | BindingFlags.Public)
                     .Where(m => m.Name.StartsWith("From") && m.Name != "FromChooseABundleScreen"))
            harmony.Patch(m, prefix: prefix);
        _installed = true;
    }

    static void Prefix(MethodBase __originalMethod, object[] __args)
    {
        ParameterInfo[] ps = __originalMethod.GetParameters();
        string? prompt = null, source = null;
        for (int i = 0; i < ps.Length && i < __args.Length; i++)
        {
            if (__args[i] is CardSelectorPrefs prefs && prefs.Prompt is { } p) prompt = p.LocEntryKey;
            else if (ps[i].Name == "source" && __args[i] is AbstractModel model) source = model.Id.ToString();
        }
        _next = new Context(__originalMethod.Name, prompt, source);
    }

    // Consume the latest context; each patched CardSelectCmd call replaces it.
    public static Context? Take()
    {
        Context? c = _next;
        _next = null;
        return c;
    }
}
