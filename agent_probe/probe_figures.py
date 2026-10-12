"""Figs. 3-4 and the numbers of the agent-signal study (predictors without hand-written cues).

  history   : most frequent successor of the previous tool (each task left out of its own training data)
  request   : first not-yet-called tool of the plan returned by an LLM planner (Qwen2.5-7B, zero-shot,
              sees only the request and the tool names and descriptions)
  reasoning : last exact tool name in the streamed reasoning
  override  : reasoning if a name has appeared, else request
  next      : both request and reasoning candidates

  figP1_accuracy.png, figP2_lead.png, probe_table.csv, probe_stats.json  (in OUT)
Confidence intervals: cluster bootstrap over episodes (1000 resamples).
"""
import collections
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze
from names import names_only_last

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "results_paper"
PLANS_FILE = os.path.join(HERE, "..", "planner_bfcl", "results", "plans_qwen2.5_7b.jsonl")
MODELS = [("qwen3_4b", "Qwen3-4B (thinking)"), ("qwen2.5_7b_react", "Qwen2.5-7B (ReAct)")]
PRED = ["history", "request", "reasoning", "override", "next"]
LBL = {"history": "History (previous tool)", "request": "User request (LLM planner)",
       "reasoning": "Reasoning text (tool names)", "override": "Reasoning else request",
       "next": "Request + reasoning (2 candidates)"}
COL = {"history": "#2a78d6", "request": "#1baf7a", "reasoning": "#eb6834", "override": "#e87ba4", "next": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
XTOK = np.arange(0, 1201, 10)
rng = np.random.default_rng(0)

PLANS = {}
for line in open(PLANS_FILE, encoding="utf-8"):
    r = json.loads(line)
    if r["source"] == "own":
        PLANS[r["id"]] = r["plan"]


def per_step(path):
    steps, *_ = analyze.load_steps(path)
    analyze.markov_predictions(steps)
    rows = []
    for s in steps:
        plan = [t for t in PLANS[s["task"]] if t not in s["used"]]
        req = plan[0] if plan else s["markov"]
        m_at = names_only_last(s)
        tok = s["eval_count"] or len(m_at) - 1
        over = [m if m is not None else req for m in m_at]
        A = s["actual"]
        jm, jo = analyze.stable_from(m_at, A), analyze.stable_from(over, A)
        lead = {"history": tok if s["markov"] == A else -1, "request": tok if req == A else -1,
                "reasoning": tok - jm if jm is not None else -1, "override": tok - jo if jo is not None else -1}
        lead["next"] = max(lead["request"], lead["reasoning"])
        names = [m for i, m in enumerate(m_at) if m is not None and (i == 0 or m != m_at[i - 1])]
        rows.append(dict(ep=(s["task"], s["rep"]), first=not s["used"], tok=tok, lead=lead,
                         distinct=len(set(m for m in m_at if m is not None)), changes=max(len(names) - 1, 0),
                         named=m_at[-1] is not None,
                         ok={"history": s["markov"] == A, "request": req == A, "reasoning": m_at[-1] == A,
                             "override": over[-1] == A, "next": A in (req, m_at[-1])}))
    return rows


def boot(rows, f):
    eps = sorted({r["ep"] for r in rows})
    by = {e: [r for r in rows if r["ep"] == e] for e in eps}
    vals = []
    for _ in range(1000):
        pick = rng.integers(len(eps), size=len(eps))
        vals.append(f([r for i in pick for r in by[eps[i]]]))
    return np.percentile(vals, [2.5, 97.5])


def coverage(path):
    calls = hit = 0
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("error"):
            continue
        plan = set(PLANS[e["task_id"]])
        used = [c["name"] for t in e["turns"] for c in t["tool_calls"]]
        calls += len(used)
        hit += sum(u in plan for u in used)
    return hit / calls


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 200})
    data, table, stats = {}, [], {}
    for name, label in MODELS:
        path = os.path.join(HERE, "traces", name + ".jsonl")
        rows = per_step(path)
        data[name] = rows
        for p in PRED:
            acc = np.mean([r["ok"][p] for r in rows])
            lo, hi = boot(rows, lambda rs: np.mean([r["ok"][p] for r in rs]))
            lead = np.array([r["lead"][p] for r in rows])
            table.append(dict(model=label, predictor=LBL[p], accuracy=round(acc, 3), ci_low=round(lo, 3),
                              ci_high=round(hi, 3), **{f"P(lead>={t} tok)": round(float(np.mean(lead >= t)), 3)
                                                        for t in [50, 100, 200, 400]}))
        ok = lambda p: np.array([r["ok"][p] for r in rows])
        stats[name] = dict(
            steps=len(rows), calls_in_plan=round(coverage(path), 4),
            request_first=round(float(np.mean([r["ok"]["request"] for r in rows if r["first"]])), 3),
            request_later=round(float(np.mean([r["ok"]["request"] for r in rows if not r["first"]])), 3),
            reasoning_only=int(np.sum(ok("reasoning") & ~ok("request"))),
            request_only=int(np.sum(ok("request") & ~ok("reasoning"))),
            both_wrong=int(np.sum(~ok("reasoning") & ~ok("request"))),
            share_named=round(float(np.mean([r["named"] for r in rows])), 3),
            share_two_or_more=round(float(np.mean([r["distinct"] >= 2 for r in rows])), 3),
            median_distinct=float(np.median([r["distinct"] for r in rows])),
            median_changes=float(np.median([r["changes"] for r in rows])),
            turn_tokens_median=float(np.median([r["tok"] for r in rows])),
            turn_tokens_p90=float(np.percentile([r["tok"] for r in rows], 90)))
    keys = list(table[0])
    with open(os.path.join(OUT, "probe_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    json.dump(stats, open(os.path.join(OUT, "probe_stats.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.16
    for i, p in enumerate(PRED):
        xs = np.arange(len(MODELS)) + (i - 2) * width
        vals, errs = [], [[], []]
        for name, label in MODELS:
            row = next(r for r in table if r["model"] == label and r["predictor"] == LBL[p])
            vals.append(row["accuracy"] * 100)
            errs[0].append((row["accuracy"] - row["ci_low"]) * 100)
            errs[1].append((row["ci_high"] - row["accuracy"]) * 100)
        ax.bar(xs, vals, width * 0.9, color=COL[p], label=LBL[p], edgecolor="white", linewidth=1.5,
               yerr=errs, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        for x, v, e in zip(xs, vals, errs[1]):
            ax.annotate(f"{v:.0f}", (x, v + e), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK)
    ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.set_ylabel("Next tool predicted correctly at the call (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figP1_accuracy.png"))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
    for idx, (ax, (name, label), xmax) in enumerate(zip(axs, MODELS, [1200, 400])):
        xs = XTOK[XTOK <= xmax]
        for p in PRED:
            lead = np.array([r["lead"][p] for r in data[name]])
            ax.plot(xs, [np.mean(lead >= x) * 100 for x in xs], lw=2, color=COL[p], label=LBL[p],
                    ls=(0, (4, 2)) if p == "override" else "-", zorder=3 if p == "override" else 2)
        ax.axvspan(80, 250, color=GRID, alpha=0.7, lw=0)
        ax.annotate(f"({'ab'[idx]})", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -42),
                    textcoords="offset points", ha="center", va="top", fontsize=11, color=INK, annotation_clip=False)
        ax.set_xlabel("Lead before the call (tokens of agent output)")
        ax.set_ylim(0, 100)
    axs[0].set_ylabel("Steps where the right tool is known (%)")
    axs[0].annotate("cold start 0.8–2.5 s\nat 100 tok/s", (165, 62), ha="center", va="center", fontsize=8, color=INK)
    axs[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figP2_lead.png"), bbox_inches="tight", pad_inches=0.05)
    print("written to", OUT)


if __name__ == "__main__":
    main()
