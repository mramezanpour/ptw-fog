"""Make paper figures from results/raw.csv into results/fig_*.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour per policy (validated categorical palette; colour follows the entity everywhere).
COL = {"keepalive": "#2a78d6", "tool_only": "#eb6834", "mob_only": "#1baf7a",
       "ptw": "#eda100", "oracle": "#e87ba4", "perfect_loc": "#008300"}
NAME = {"keepalive": "Keep-alive (TTL)", "tool_only": "Tool-only (PASTE-like)", "mob_only": "Mobility-only",
        "ptw": "PTW (proposed)", "oracle": "Perfect prediction", "perfect_loc": "PTW + perfect location"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 200})

df = pd.read_csv("results/raw.csv")
e1 = df[df.exp == "E1_main"]
e1_5 = e1  # reference level for sweeps (all 10 seeds)


def bars(ax, sub, policies, metric, fmt, ylabel):
    g = sub.groupby("policy")[metric]
    mean, std = g.mean(), g.std()
    x = np.arange(len(policies))
    ax.bar(x, [mean[p] for p in policies], yerr=[std[p] for p in policies], width=0.62,
           color=[COL[p] for p in policies], edgecolor="white", linewidth=2,
           error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    for xi, p in zip(x, policies):
        ax.annotate(fmt.format(mean[p]), (xi, mean[p] + std[p]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [NAME[p].replace(" (", "\n(") for p in policies], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.15)


# Fig 1 — main comparison, per-session sandboxes
P1 = ["keepalive", "mob_only", "tool_only", "ptw", "oracle"]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
bars(axs[0], e1, P1, "mean_lat_ms", "{:.0f}", "Mean tool-call latency (ms)")
bars(axs[1], e1, P1, "task_time_s", "{:.1f}", "Mean agent task time (s)")
e1c = e1.assign(cold_pct=100 * e1.cold_rate)
bars(axs[2], e1c, P1, "cold_pct", "{:.0f}%", "Calls hitting a cold start (%)")
fig.suptitle("Per-session sandboxes: PTW removes most cold starts (10 seeds, mean ± sd)", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("results/fig1_main.png")

# Fig 2 — node memory
P2 = ["keepalive", "tool_only", "ptw", "oracle"]
e2 = pd.concat([df[df.exp == "E2_memory"], e1_5[e1_5.policy.isin(P2)].assign(level="4096MB")])
e2["mem_gb"] = e2.mem_mb / 1024
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for p in P2:
    g = e2[e2.policy == p].groupby("mem_gb").mean_lat_ms.mean()
    lbl = NAME[p] + (" — overlaps PTW" if p == "tool_only" else "")
    ax.plot(g.index, g.values, marker="o", ms=7, lw=2, color=COL[p], label=lbl,
            ls="--" if p == "tool_only" else "-", zorder=3 if p == "tool_only" else 2)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8], ["2", "4", "8"])
ax.set_xlabel("Warm-container memory per fog node (GB)")
ax.set_ylabel("Mean tool-call latency (ms)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Latency vs. node memory (10 seeds)", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("results/fig2_memory.png")

# Fig 3 — quality of the agent signal
e3 = pd.concat([df[df.exp == "E3_hint"], e1_5[e1_5.policy == "ptw"].assign(level="acc=0.8")])
e3["acc"] = e3.level.str.replace("acc=", "").astype(float)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
g = e3.groupby("acc").mean_lat_ms.agg(["mean", "std"])
ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=7, lw=2, color=COL["ptw"], ecolor=MUTED,
            capsize=3, label=NAME["ptw"])
for p in ["keepalive", "oracle"]:
    v = e1_5[e1_5.policy == p].mean_lat_ms.mean()
    ax.axhline(v, color=COL[p], lw=1.5, ls="--")
    ax.annotate(f"{NAME[p]} ({v:.0f} ms)", (0.0, v), xytext=(4, 4), textcoords="offset points", ha="left",
                fontsize=8.5, color=INK)
ax.set_xticks([0, 0.5, 0.8, 1.0], ["0\n(pattern only)", "0.5", "0.8", "1.0"])
ax.set_xlabel("P(reasoning trace reveals the next tool)")
ax.set_ylabel("Mean tool-call latency (ms)")
ax.set_ylim(0, 2100)
ax.set_title("PTW gain vs. quality of the agent signal (10 seeds)", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("results/fig3_hint.png")

# Fig 4 — value of location knowledge
P4 = ["tool_only", "ptw", "perfect_loc", "oracle"]
e4 = df[df.exp == "E4_mobility"]
levels = ["p_move=0.02", "p_move=0.3"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
w = 0.2
for i, p in enumerate(P4):
    m = [e4[(e4.level == lv) & (e4.policy == p)].mean_lat_ms.mean() for lv in levels]
    xs = np.arange(len(levels)) + (i - 1.5) * w
    ax.bar(xs, m, width=w * 0.92, color=COL[p], edgecolor="white", linewidth=2, label=NAME[p])
    for xx, mm in zip(xs, m):
        ax.annotate(f"{mm:.0f}", (xx, mm), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                    color=INK)
ax.set_xticks(range(len(levels)), ["Slow users\n(move ≈ every 25 s)", "Fast users\n(move ≈ every 1.7 s)"])
ax.set_ylabel("Mean tool-call latency (ms)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.margins(y=0.2)
ax.set_title("Value of knowing where the user will be (200 ms/hop, 10 seeds)", loc="left", fontsize=11,
             color=INK)
fig.tight_layout()
fig.savefig("results/fig4_mobility.png")

# Fig 5 — keep-alive TTL vs PTW: latency / memory trade-off
e6 = pd.concat([df[df.exp == "E6_ttl"], e1_5[e1_5.policy.isin(["keepalive", "ptw"])].assign(level="ttl=30s")])
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for p in ["keepalive", "ptw"]:
    g = e6[e6.policy == p].groupby("ttl_s")[["mem_gbs", "mean_lat_ms"]].mean().sort_index()
    ax.plot(g.mem_gbs / 1000, g.mean_lat_ms, marker="o", ms=7, lw=2, color=COL[p], label=NAME[p])
    offs = {10.0: (0, 9), 30.0: (-6, -16), 120.0: (6, 9)}
    for ttl, row in g.iterrows():
        dx, dy = offs[ttl]
        ax.annotate(f"TTL {ttl:.0f}s", (row.mem_gbs / 1000, row.mean_lat_ms), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK, ha="right" if dx < 0 else "left" if dx > 0 else "center")
ax.set_xlabel("Memory held by warm containers (thousand GB·s)")
ax.set_ylabel("Mean tool-call latency (ms)")
ax.set_ylim(0, 2100)
ax.set_xlim(left=45, right=58)
ax.grid(axis="x", color=GRID)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("A longer keep-alive cannot buy PTW's latency (10 seeds)", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("results/fig5_ttl.png")

# Fig 6 — shared instances
e5 = df[df.exp == "E5_shared"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
bars(axs[0], e5, P1, "mean_lat_ms", "{:.0f}", "Mean tool-call latency (ms)")
e5g = e5.assign(mem_k=e5.mem_gbs / 1000)
bars(axs[1], e5g, P1, "mem_k", "{:.1f}", "Warm memory (thousand GB·s)")
fig.suptitle("Shared instances: popularity-based Mobility-only is best; the agent signal adds nothing (10 seeds)", x=0.01,
             ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("results/fig6_shared.png")
print("figures written")
