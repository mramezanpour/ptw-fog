"""Fig. 1: timeline of a real Qwen3-4B task (t12) under keep-alive and under PTW, replayed at 100 tokens/s."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)

TOOLS = {"get_calendar_events": (1.0, 0.15), "get_route": (1.2, 0.30), "create_calendar_event": (1.0, 0.15)}
MARK = {"get_calendar_events": "o", "get_route": "^", "create_calendar_event": "s"}
ep = json.load(open(ROOT / "simulator" / "data" / "episodes" / "qwen3_4b_llm.json"))["episodes"][11]
ZETA = 1.0   # the request planner answers 1 s after the task starts
TPS = 100.0
COLT = {"get_calendar_events": "#2a78d6", "get_route": "#1baf7a", "create_calendar_event": "#e87ba4"}
INK, EDGE = "#0b0b0b", "#3d3b36"
plt.rcParams.update({"font.size": 9, "xtick.direction": "in", "ytick.direction": "in"})
STY = dict(reason=dict(fc="#e6e5e1", ec=EDGE, lw=0.8), cold=dict(fc="#eb6834", ec=EDGE, lw=0.8),
           exe=dict(fc="#3d3b36", ec=EDGE, lw=0.8), warm=dict(fc="#eda100", ec=EDGE, lw=0.8))
fig, axs = plt.subplots(2, 1, figsize=(9, 3.4), sharex=True)
for k, ax in enumerate(axs):
    ptw = k == 1
    t = 0.0
    for j, turn in enumerate(ep["turns"]):
        dur = turn["prefill_s"] + turn["tokens"] / TPS
        ax.broken_barh([(t, dur)], (1.0, 0.8), **STY["reason"])
        ax.text(t + dur / 2, 1.4, f"turn {j + 1}", ha="center", va="center", fontsize=8.5)
        if ptw:
            for tok, tool in turn.get("mention", []):
                if tool in MARK:
                    ax.plot(t + turn["prefill_s"] + tok / TPS, 2.05, marker=MARK[tool], ms=5, mfc=COLT[tool],
                            mec=EDGE, mew=0.6, ls="none")
        t_call = t + dur
        if not turn["calls"]:
            t = t_call
            break
        c, d = TOOLS[turn["calls"][0]]
        if ptw:
            ws = t + (ZETA if j == 0 else 0.0)
            ax.broken_barh([(ws, c)], (0.2, 0.5), **STY["warm"])
            assert ws + c <= t_call
            ax.broken_barh([(t_call, d)], (1.0, 0.8), **STY["exe"])
            t = t_call + d
        else:
            ax.broken_barh([(t_call, c)], (1.0, 0.8), **STY["cold"])
            ax.broken_barh([(t_call + c, d)], (1.0, 0.8), **STY["exe"])
            t = t_call + c + d
    ax.axvline(t, color="black", lw=0.8, ls="--")
    ax.text(t + 0.2, 1.4, f"{t:.1f} s", va="center", fontsize=9)
    ax.set_ylim(0, 2.4)
    ax.set_yticks([1.4] + ([0.45] if ptw else []), ["agent"] + (["warm-up"] if ptw else []))
    ax.set_xlim(0, 25)
    ax.set_xlabel("Time (s)" if ptw else "")
    ax.text(0.5, -0.52 if ptw else -0.14, "(b)" if ptw else "(a)", transform=ax.transAxes, ha="center", va="top",
            fontsize=10)
h = [Patch(label="reasoning", **STY["reason"]), Patch(label="cold start", **STY["cold"]),
     Patch(label="tool execution", **STY["exe"]), Patch(label="warm-up", **STY["warm"])]
h += [Line2D([], [], marker=m, ls="none", mfc=COLT[n], mec=EDGE, mew=0.6, label=n) for n, m in MARK.items()]
fig.legend(handles=h, loc="upper center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06),
           handlelength=1.6, columnspacing=1.0)
fig.tight_layout(h_pad=2.0)
fig.savefig(OUT / "fig1_timeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("ok")
