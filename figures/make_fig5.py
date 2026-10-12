"""Fig. 5: architecture of PTW."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9})
EDGE, ACC, ACCF, GRP, WARM = "#3d3b36", "#c98a00", "#fdf2d6", "#898781", "#eda100"
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=EDGE, ls="-", lw=0.9):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, linespacing=1.25)


def arrow(p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=0.9, color="black", shrinkA=0, shrinkB=0))


def group(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fc="#f4f3ef", ec=GRP, lw=0.8, ls=(0, (4, 3))))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center", fontsize=10)


# agent platform
group(1, 8, 33, 37, "Agent platform")
box(4, 32, 27, 6.5, "LLM agent\n(reasoning stream, tool calls)")
box(4, 22, 12.5, 6.5, "Request\nplanner")
box(18.5, 22, 12.5, 6.5, "Reasoning\nmonitor")
box(4, 11, 27, 7.5, "Candidate generation\nand timing weights", fc=ACCF, ec=ACC)
for x in (10.25, 24.75):
    arrow((x, 32), (x, 28.5))
    arrow((x, 22), (x, 18.5))

# fog controller
group(40, 8, 26, 37, "Fog controller")
box(43, 32, 20, 6.5, "Mobility\npredictor")
box(43, 22, 20, 6.5, "Demand\nestimation")
box(43, 11, 20, 7.5, "DG-PTW planner", fc=ACCF, ec=ACC)
arrow((53, 32), (53, 28.5))
arrow((53, 22), (53, 18.5))
ax.plot([31, 37], [14.75, 14.75], color="black", lw=0.9)
ax.plot([37, 37], [14.75, 25.25], color="black", lw=0.9)
arrow((37, 25.25), (43, 25.25))
ax.text(40, 25.9, "hints", ha="center", va="bottom", fontsize=8.5)

# fog nodes
group(72, 8, 27, 37, "Fog nodes")
for y in (30, 19):
    for x in (75, 88):
        ax.add_patch(Rectangle((x, y), 9, 7, fc="white", ec=EDGE, lw=0.9))
        for k in range(3):
            ax.add_patch(Rectangle((x + 0.9 + k * 2.75, y + 1.5), 2.2, 4, fc=WARM if (k + x + y) % 3 == 0 else "white",
                                   ec=EDGE, lw=0.6))
ax.plot([84, 88], [33.5, 33.5], color="black", lw=0.8)
ax.plot([84, 88], [22.5, 22.5], color="black", lw=0.8)
ax.plot([79.5, 79.5], [26, 30], color="black", lw=0.8)
ax.plot([92.5, 92.5], [26, 30], color="black", lw=0.8)
ax.add_patch(Rectangle((75.9, 12.2), 2.2, 3.2, fc=WARM, ec=EDGE, lw=0.6))
ax.text(79.2, 13.8, "warm sandbox", va="center", fontsize=8.5)
arrow((63, 14.75), (72, 14.75))
ax.text(67.3, 15.6, "warm/evict", ha="center", va="bottom", fontsize=8.5)

# users and calls
box(4, 0.5, 18, 5, "Mobile users")
arrow((13, 5.5), (13, 8))
ax.text(14, 6.7, "request", fontsize=8.5, va="center")
ax.plot([22, 92.5], [3, 3], color="black", lw=0.9)
arrow((92.5, 3), (92.5, 8))
ax.text(57, 3.6, "tool calls", ha="center", va="bottom", fontsize=8.5)
fig.savefig(OUT / "fig5_arch.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("ok")
