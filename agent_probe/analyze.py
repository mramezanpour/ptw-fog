"""Analyze collected traces: how early does the agent's reasoning reveal its next tool?

For every agent step (an LLM turn that ends in a tool call) we replay the reasoning stream and,
after each streamed chunk, ask three predictors which tool comes next:

  markov   : most likely next tool given the previous one (learned on the OTHER tasks only)
  request  : tools the user's request asks for (natural-language cues), in the order mentioned,
             minus the ones already called -- known before any reasoning starts
  mention  : the tool referred to last in the reasoning so far (exact name or natural-language cue)
  combined : mention if any, else request, else markov   (what PTW would use)

Lead time = time from the moment the prediction becomes correct and stays correct until the
tool call arrives. P(lead >= delta) is exactly what the simulator needs: a sandbox with cold-start
delta can only be warmed in time if the right tool is known at least delta before the call.

Usage:  python analyze.py                 (all files in traces/)
        python analyze.py traces/qwen3_4b.jsonl
Output: results/<model>/summary.json, steps.csv, fig_progress.png, fig_lead.png
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import tools

START = "<start>"
FRACTIONS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
DELTA_S = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
DELTA_TOK = [10, 25, 50, 100, 200, 400]


def _patterns():
    pats = []
    for n in tools.NAMES:
        variants = {n, n.replace("_", " "), n.replace("_", "-")}
        pats.append((n, re.compile("|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)),
                                   re.IGNORECASE)))
    return pats


PATS = _patterns()
CUES = [(name, re.compile(p, 0 if re.search(r"\[A-Z\]|AAPL", p) else re.IGNORECASE))
        for name, ps in tools.CUES.items() for p in ps]


def mentions(text, cues=True):
    """Sorted list of (end_char, tool) for every reference to a tool in text."""
    out = []
    for name, p in PATS + (CUES if cues else []):
        out += [(m.end(), name) for m in p.finditer(text)]
    return sorted(out)


def request_plan(text):
    """Tools the request refers to, ordered by first mention."""
    first = {}
    for name, p in CUES + PATS:
        for m in p.finditer(text):
            first[name] = min(first.get(name, m.start()), m.start())
    return sorted(first, key=first.get)


def _load_tasks():
    """Original tasks (t01..) plus the held-out tasks (h01..) if they exist."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for fn in ["tasks.json", "tasks_heldout.json"]:
        try:
            out.update({t["id"]: t["task"] for t in json.load(open(os.path.join(here, fn), encoding="utf-8"))})
        except OSError:
            pass
    return out


def load_steps(path):
    steps, episodes, errors, no_tool = [], 0, 0, 0
    for line in open(path, encoding="utf-8"):
        ep = json.loads(line)
        episodes += 1
        if ep.get("error"):
            errors += 1
            continue
        prev = START
        used = []
        any_tool = False
        for k, turn in enumerate(ep["turns"]):
            if not turn["tool_calls"] or turn["t_tool"] is None:
                continue
            any_tool = True
            evs = [e for e in turn["events"] if e[0] <= turn["t_tool"]]
            actual = turn["tool_calls"][0]["name"]
            steps.append(dict(task=ep["task_id"], rep=ep["repeat"], model=ep["model"], step=k, prev=prev,
                              actual=actual, calls=[c["name"] for c in turn["tool_calls"]], used=list(used),
                              n_calls=len(turn["tool_calls"]), t_tool=turn["t_tool"],
                              times=[e[0] for e in evs], texts=[e[2] for e in evs],
                              n_think=sum(e[1] == "think" for e in evs),
                              eval_count=turn.get("eval_count"), eval_duration=turn.get("eval_duration"),
                              prefill_s=(turn.get("prompt_eval_duration") or 0) / 1e9))
            prev = actual
            used += [c["name"] for c in turn["tool_calls"]]
        no_tool += not any_tool
    return steps, episodes, errors, no_tool


def markov_predictions(steps):
    """Leave-one-task-out argmax of P(next | prev)."""
    by_task = defaultdict(list)
    for s in steps:
        by_task[s["task"]].append(s)
    total = defaultdict(Counter)
    glob_counts = Counter()
    for s in steps:
        total[s["prev"]][s["actual"]] += 1
        glob_counts[s["actual"]] += 1
    for task, ss in by_task.items():
        own = defaultdict(Counter)
        own_glob = Counter()
        for s in ss:
            own[s["prev"]][s["actual"]] += 1
            own_glob[s["actual"]] += 1
        for s in ss:
            c = total[s["prev"]] - own[s["prev"]]
            if not c:
                c = glob_counts - own_glob
            s["markov"] = c.most_common(1)[0][0] if c else None


def request_predictions(steps):
    tasks = _load_tasks()
    for s in steps:
        plan = [t for t in request_plan(tasks.get(s["task"], "")) if t not in s["used"]]
        s["request"] = plan[0] if plan else s["markov"]


def replay(s):
    """Predictions after 0..n chunks (index 0 = before any reasoning text)."""
    text = "".join(s["texts"])
    ends = np.cumsum([0] + [len(t) for t in s["texts"]])
    ms = mentions(text)
    mention_at, j, last = [], 0, None
    for pos in ends:
        while j < len(ms) and ms[j][0] <= pos:
            last = ms[j][1]
            j += 1
        mention_at.append(last)
    combined_at = [m if m is not None else s["request"] for m in mention_at]
    return mention_at, combined_at


def stable_from(preds, actual):
    """First index from which every prediction equals actual (None if the last one is wrong)."""
    if not preds or preds[-1] != actual:
        return None
    i = len(preds) - 1
    while i > 0 and preds[i - 1] == actual:
        i -= 1
    return i


def analyze(path, outdir):
    steps, episodes, errors, no_tool = load_steps(path)
    if not steps:
        print(f"{path}: no tool-calling steps found")
        return None
    markov_predictions(steps)
    request_predictions(steps)
    rows = []
    acc = {k: np.zeros(len(FRACTIONS)) for k in ["markov", "request", "mention", "combined"]}
    any_call = 0
    for s in steps:
        m_at, c_at = replay(s)
        n = len(m_at) - 1
        for i, f in enumerate(FRACTIONS):
            j = int(math.ceil(f * n))
            acc["markov"][i] += s["markov"] == s["actual"]
            acc["request"][i] += s["request"] == s["actual"]
            acc["mention"][i] += m_at[j] == s["actual"]
            acc["combined"][i] += c_at[j] == s["actual"]
        any_call += c_at[-1] in s["calls"]
        times = [0.0] + s["times"]                       # index 0 = request sent
        tok_total = s["eval_count"] or n
        r = dict(task=s["task"], rep=s["rep"], step=s["step"], prev=s["prev"], actual=s["actual"],
                 n_calls=s["n_calls"], markov=s["markov"], request=s["request"], mention_end=m_at[-1], step_s=s["t_tool"], prefill_s=s["prefill_s"],
                 chunks=n, eval_count=s["eval_count"],
                 tps=(s["eval_count"] / (s["eval_duration"] / 1e9)) if s["eval_count"] and s["eval_duration"] else None)
        for name, preds in [("combined", c_at), ("mention", m_at)]:
            j = stable_from(preds, s["actual"])
            r[f"lead_s_{name}"] = s["t_tool"] - times[j] if j is not None else -1.0
            r[f"lead_tok_{name}"] = tok_total - j if j is not None else -1
        rows.append(r)
    N = len(steps)
    for k in acc:
        acc[k] = (acc[k] / N).round(4).tolist()

    lead_c = np.array([r["lead_s_combined"] for r in rows])
    lead_m = np.array([r["lead_s_mention"] for r in rows])
    tok_c = np.array([r["lead_tok_combined"] for r in rows])
    tok_m = np.array([r["lead_tok_mention"] for r in rows])
    step_s = np.array([r["step_s"] for r in rows])
    tps = np.array([r["tps"] for r in rows if r["tps"]])
    logt = np.log(np.maximum(step_s, 1e-3))
    summary = dict(
        source=os.path.basename(path), model=steps[0]["model"], episodes=episodes, errors=errors,
        episodes_without_tool_call=no_tool, steps=N,
        tools_used=dict(Counter(r["actual"] for r in rows).most_common()),
        tokens_per_s_median=float(np.median(tps)) if len(tps) else None,
        step_time_s=dict(median=float(np.median(step_s)), p90=float(np.percentile(step_s, 90)),
                         lognormal_mu=float(logt.mean()), lognormal_sigma=float(logt.std())),
        accuracy_at_call=dict(markov=acc["markov"][-1], request=acc["request"][-1], mention=acc["mention"][-1],
                              combined=acc["combined"][-1], combined_matches_any_parallel_call=any_call / N),
        steps_without_text_before_call=float(np.mean([r["chunks"] == 0 for r in rows])),
        steps_with_parallel_calls=float(np.mean([r["n_calls"] > 1 for r in rows])),
        accuracy_vs_reasoning_fraction=dict(fraction=FRACTIONS, **acc),
        p_lead_at_least_s=dict(delta_s=DELTA_S, combined=[float((lead_c >= d).mean()) for d in DELTA_S],
                               mention=[float((lead_m >= d).mean()) for d in DELTA_S]),
        p_lead_at_least_tokens=dict(delta_tokens=DELTA_TOK,
                                    combined=[float((tok_c >= d).mean()) for d in DELTA_TOK],
                                    mention=[float((tok_m >= d).mean()) for d in DELTA_TOK]),
    )
    os.makedirs(outdir, exist_ok=True)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    with open(os.path.join(outdir, "steps.csv"), "w", encoding="utf-8") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    plot(summary, lead_c, lead_m, outdir)
    return summary


def plot(sm, lead_c, lead_m, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    COL = {"markov": "#2a78d6", "mention": "#eb6834", "request": "#1baf7a", "combined": "#eda100"}
    LBL = {"markov": "Markov prior (previous tool only)", "request": "User request (cues, in order)",
           "mention": "Last tool referred to in reasoning", "combined": "Combined (PTW)"}
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 200})
    a = sm["accuracy_vs_reasoning_fraction"]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for k in ["markov", "request", "mention", "combined"]:
        ax.plot(np.array(a["fraction"]) * 100, np.array(a[k]) * 100, marker="o", ms=5, lw=2, color=COL[k],
                label=LBL[k])
    ax.set_xlabel("Share of the step's reasoning seen so far (%)")
    ax.set_ylabel("Next tool predicted correctly (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title(f"{sm['model']}: how early the next tool is known ({sm['steps']} steps)", loc="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_progress.png"))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    xs = np.linspace(0, max(6.0, float(np.percentile(lead_c[lead_c > 0], 95)) if (lead_c > 0).any() else 6), 200)
    for k, lead in [("combined", lead_c), ("mention", lead_m)]:
        ax.plot(xs, [(lead >= x).mean() * 100 for x in xs], lw=2, color=COL[k], label=LBL[k])
    ax.axvspan(0.8, 3.0, color=GRID, alpha=0.6, lw=0)
    ax.annotate("cold-start range\nused in the simulator", (1.9, 96), ha="center", va="top", fontsize=8, color=INK)
    ax.set_xlabel("Lead time Δ before the tool call (s, measured on this machine)")
    ax.set_ylabel("Steps where the right tool is known ≥ Δ ahead (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 0.8))
    ax.set_title("Is the signal early enough to warm a sandbox?", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_lead.png"))
    plt.close("all")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "traces", "*.jsonl")))
    if not files:
        sys.exit("No traces found. Run collect.py first.")
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        sm = analyze(p, os.path.join(here, "results", name))
        if not sm:
            continue
        acc, lead = sm["accuracy_at_call"], sm["p_lead_at_least_s"]
        print(f"\n== {sm['model']}: {sm['episodes']} episodes, {sm['steps']} tool steps, "
              f"{sm['errors']} errors, {sm['episodes_without_tool_call']} without any tool call")
        print(f"   tokens/s (median): {sm['tokens_per_s_median']}")
        print(f"   accuracy at the call: markov {acc['markov']:.2f} | request {acc['request']:.2f} | "
              f"mention {acc['mention']:.2f} | combined {acc['combined']:.2f}")
        print(f"   steps with no text before the call: {sm['steps_without_text_before_call']:.2f} | "
              f"with parallel calls: {sm['steps_with_parallel_calls']:.2f}")
        for d, pc, pm in zip(lead["delta_s"], lead["combined"], lead["mention"]):
            print(f"   P(right tool known >= {d:>4} s ahead): combined {pc:.2f} | mention only {pm:.2f}")
        print(f"   results in results/{name}/")


if __name__ == "__main__":
    main()
