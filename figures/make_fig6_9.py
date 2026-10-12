"""Figures 6-9: trace-driven results (latency, memory and speed sweeps) and value of location prediction."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulator" / "results"
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)
MODELS = [("qwen3_4b", "Qwen3-4B"), ("qwen2.5_7b_react", "Qwen2.5-7B")]
ORDER = ["keepalive", "history", "request", "reasoning", "override", "next", "plan", "specbox", "ptw", "oracle", "oracle_all"]
NAME = {"keepalive": "Keep-alive", "history": "History", "request": "Request", "reasoning": "Reasoning",
        "override": "Reasoning\nelse request", "next": "Next step", "plan": "Whole plan", "specbox": "SpecBox-\nlike", "ptw": "PTW",
        "oracle": "Oracle\n(next step)", "oracle_all": "Oracle\n(all steps)"}
SWEEP = ["keepalive", "history", "request", "specbox", "ptw", "oracle_all"]
COL = {"specbox": "#7a5cc4", "keepalive": "#2a78d6", "history": "#eb6834", "request": "#1baf7a", "ptw": "#eda100", "oracle_all": "#e87ba4"}
INK, MUTED, GRID, NEUTRAL, ORACLE = "#0b0b0b", "#898781", "#e6e5e1", "#b9b7b0", "#dedcd5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.grid.axis": "y", "axes.axisbelow": True, "savefig.dpi": 220})
df = pd.read_csv(SIM / "v2_raw.csv")
df["episodes"] = df.episodes.str.replace("_llm", "")


def label(ax, s, off=-42):
    ax.annotate(s, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, off), textcoords="offset points",
                ha="center", va="top", fontsize=11, color=INK, annotation_clip=False)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# Fig. 6: main result and ablation
t1 = df[df.exp == "V1_main"]
fig, axs = plt.subplots(2, 1, figsize=(10, 7.6))
for k, (ax, (ep, _)) in enumerate(zip(axs, MODELS)):
    g = t1[t1.episodes == ep].groupby("policy").mean_lat_ms.agg(["mean", "std"])
    x = np.arange(len(ORDER))
    colors = ["#eda100" if p == "ptw" else (ORACLE if p.startswith("oracle") else NEUTRAL) for p in ORDER]
    ax.bar(x, [g.loc[p, "mean"] for p in ORDER], 0.66, yerr=[g.loc[p, "std"] for p in ORDER], color=colors,
           edgecolor="white", linewidth=2, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    base = g.loc["keepalive", "mean"]
    for xi, p in zip(x, ORDER):
        v, e = g.loc[p, "mean"], g.loc[p, "std"]
        txt = f"{v:.0f}" if p == "keepalive" else f"{v:.0f}\n({100 * (v - base) / base:+.0f}%)"
        ax.annotate(txt, (xi, v + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.set_xticks(x, [NAME[p] for p in ORDER], fontsize=8.5)
    ax.set_ylabel("Mean tool-call latency (ms)")
    ax.margins(y=0.22)
    label(ax, "(a)" if k == 0 else "(b)", -36)
fig.tight_layout(h_pad=3.0)
save(fig, "fig6_main.png")


def sweep(exp, xcol, xlabel, name, xfmt):
    t = pd.concat([df[df.exp == exp], df[(df.exp == "V1_main") & df.policy.isin(SWEEP)]])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
    for k, (ax, (ep, _)) in enumerate(zip(axs, MODELS)):
        for p in SWEEP:
            g = t[(t.episodes == ep) & (t.policy == p)].groupby(xcol).mean_lat_ms.mean().sort_index()
            ax.plot(g.index, g.values, marker="o", ms=6, lw=2, color=COL[p], label=NAME[p].replace("-\n", "-").replace("\n", " "))
        ax.set_xscale("log", base=2)
        vals = sorted(t[xcol].unique())
        ax.set_xticks(vals, [xfmt(v) for v in vals])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1500)
        label(ax, "(a)" if k == 0 else "(b)")
    axs[0].set_ylabel("Mean tool-call latency (ms)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=3.0)
    save(fig, name)


sweep("V3_memory", "mem_mb", "Sandbox memory per fog node (GB)", "fig7_memory.png", lambda v: f"{v / 1024:.0f}")
sweep("V4_speed", "target_tps", "Agent decoding speed (tokens/s)", "fig8_speed.png", lambda v: f"{v:.0f}")

# Fig. 9: value of location knowledge (synthetic simulator)
syn = pd.read_csv(SIM / "raw.csv")
e4 = syn[syn.exp == "E4_mobility"]
P4 = ["tool_only", "ptw", "perfect_loc", "oracle"]
N4 = {"tool_only": "Tool prediction only", "ptw": "PTW", "perfect_loc": "PTW with perfect location",
      "oracle": "Perfect prediction"}
C4 = {"tool_only": "#eb6834", "ptw": "#eda100", "perfect_loc": "#008300", "oracle": "#e87ba4"}
levels = ["p_move=0.02", "p_move=0.3"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
w = 0.2
for i, p in enumerate(P4):
    m = [e4[(e4.level == lv) & (e4.policy == p)].mean_lat_ms.mean() for lv in levels]
    xs = np.arange(len(levels)) + (i - 1.5) * w
    ax.bar(xs, m, width=w * 0.92, color=C4[p], edgecolor="white", linewidth=2, label=N4[p])
    for xx, mm in zip(xs, m):
        ax.annotate(f"{mm:.0f}", (xx, mm), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                    color=INK)
ax.set_xticks(range(len(levels)), ["Slow users\n(move every 25 s on average)", "Fast users\n(move every 1.7 s on average)"])
ax.set_ylabel("Mean tool-call latency (ms)")
ax.set_ylim(0, 1250)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
save(fig, "fig9_mobility.png")
print("ok")
