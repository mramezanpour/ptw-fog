"""Fig. 2: shared instances vs per-session sandboxes (synthetic workload)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.grid.axis": "y", "axes.axisbelow": True})
df = pd.read_csv(ROOT / "simulator" / "results" / "raw.csv")
P = ["keepalive", "mob_only", "ptw"]
NAME = {"keepalive": "Keep-alive", "mob_only": "Popularity-based\nwarming", "ptw": "Agent-aware\nwarming"}
COL = {"keepalive": "#2a78d6", "mob_only": "#1baf7a", "ptw": "#eda100"}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for k, (ax, exp) in enumerate(zip(axs, ["E5_shared", "E1_main"])):
    g = df[df.exp == exp].groupby("policy")
    m, s, cr = g.mean_lat_ms.mean(), g.mean_lat_ms.std(), g.cold_rate.mean()
    x = np.arange(len(P))
    ax.bar(x, [m[p] for p in P], 0.6, yerr=[s[p] for p in P], color=[COL[p] for p in P], edgecolor="white",
           linewidth=2, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    for xi, p in zip(x, P):
        ax.annotate(f"{m[p]:.0f} ms\n{100 * cr[p]:.1f}% cold" if 100 * cr[p] < 10 else f"{m[p]:.0f} ms\n{100 * cr[p]:.0f}% cold", (xi, m[p] + s[p]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [NAME[p] for p in P], fontsize=8.5)
    ax.set_ylim(0, 2300)
    ax.annotate("(a)" if k == 0 else "(b)", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -40),
                textcoords="offset points", ha="center", va="top", fontsize=11, color=INK, annotation_clip=False)
axs[0].set_ylabel("Mean tool-call latency (ms)")
fig.tight_layout(w_pad=3)
fig.savefig(OUT / "fig2_isolation.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
print("ok")
