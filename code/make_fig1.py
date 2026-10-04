# -*- coding: utf-8 -*-
"""Figure 1 - the idea as a left-to-right story, so an editor gets it at a glance.
Four steps: (1) a phone films the arm from the side; (2) the tilted arm projects
SHORT (foreshortening); (3) because the arm's true length L is known, the missing
out-of-plane depth sqrt(L^2 - p^2) is added back; (4) the true abduction angle is
recovered. The precise right-triangle geometry appears once, small, in step 3.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc, Circle

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})

BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#9aa6ad", "#c0392b"

fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.20, wspace=0.45)
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

TITLES = ["1.  Oblique camera", "2.  Foreshortening",
          "3.  Restore length L", "4.  Estimate the angle"]
CAPS = ["A phone films the\narm from an oblique view.",
        "The tilted arm looks\nshorter in the photo (p < L).",
        "L is known, so the missing\ndepth √(L²−p²) is added back.",
        "Reduced error in tested\nstatic holds up to 60° azimuth."]
for ax, t, c in zip(axes, TITLES, CAPS):
    ax.text(0.5, 1.16, t, ha="center", va="bottom", fontsize=12.5, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, -0.20, c, ha="center", va="top", fontsize=10.5, transform=ax.transAxes)


def stick(ax, x0, y0, arm_to, arm_color=BLUE, arm_lw=3.5, ghost=None):
    """Tiny front-facing figure: head, torso, legs, and one raised (abducting) arm."""
    head_r = 0.05
    ax.add_patch(Circle((x0, y0 + 0.30), head_r, fill=False, ec="k", lw=1.8))
    ax.plot([x0, x0], [y0, y0 + 0.25], color="k", lw=1.8)            # torso
    ax.plot([x0, x0 - 0.07], [y0, y0 - 0.12], color="k", lw=1.8)     # legs
    ax.plot([x0, x0 + 0.07], [y0, y0 - 0.12], color="k", lw=1.8)
    sh = (x0, y0 + 0.20)                                             # shoulder
    if ghost is not None:
        ax.plot([sh[0], ghost[0]], [sh[1], ghost[1]], color=GREY, lw=2,
                ls=(0, (2, 2)), alpha=0.7)
    ax.plot([sh[0], arm_to[0]], [sh[1], arm_to[1]], color=arm_color, lw=arm_lw,
            solid_capstyle="round")
    return sh


# ---- panel 1: phone films from the side ----
ax = axes[0]
ax.add_patch(FancyBboxPatch((0.06, 0.34), 0.12, 0.30,
             boxstyle="round,pad=0.005,rounding_size=0.02", fc="#dfe7ec", ec="k", lw=1.5))
ax.add_patch(Circle((0.12, 0.585), 0.018, fc="k"))            # lens
ax.text(0.12, 0.30, "phone", ha="center", va="top", fontsize=9, color="#555")
ax.add_patch(FancyArrowPatch((0.21, 0.49), (0.46, 0.49), arrowstyle="-|>",
             mutation_scale=14, color="#444", lw=1.5))
ax.text(0.34, 0.55, "films from\nan angle", ha="center", va="bottom", fontsize=8.5, color="#555")
stick(ax, 0.74, 0.30, arm_to=(0.93, 0.62), arm_color=BLUE)

# ---- panel 2: foreshortening (arm projects short) ----
ax = axes[1]
sh = stick(ax, 0.38, 0.30, arm_to=(0.52, 0.62), arm_color=ORANGE, arm_lw=5,
           ghost=(0.82, 0.70))
ax.text(0.53, 0.65, "p", color="#b3790a", ha="left", va="bottom", fontsize=12, fontweight="bold")
ax.text(0.84, 0.70, "true L\n(unseen)", color="#7f8a91", ha="left", va="center", fontsize=9)
ax.text(0.58, 0.40, "p < L", color=RED, ha="center", va="center", fontsize=11)

# ---- panel 3: restore length (the one small right triangle) ----
ax = axes[2]
L = 1.0; p = 0.60; d = np.sqrt(L**2 - p**2)
sx, sy, sc = 0.62, 0.55, 0.30            # scale/placement of the triangle
ox, oy = 0.26, 0.24
O = (ox, oy); top = (ox, oy + p * sc); E = (ox + d * sc, oy + p * sc)
ax.plot([O[0], top[0]], [O[1], top[1]], color=ORANGE, lw=5, solid_capstyle="round")   # p
ax.plot([top[0], E[0]], [top[1], E[1]], color=GREEN, lw=2.6, ls="--")                 # depth
ax.plot([O[0], E[0]], [O[1], E[1]], color=BLUE, lw=4, solid_capstyle="round")         # L
s = 0.03
ax.plot([top[0], top[0] + s, top[0] + s], [top[1] - s, top[1] - s, top[1]], color="k", lw=1.1)
ax.text(O[0] - 0.04, (O[1] + top[1]) / 2, "p", color="#b3790a", ha="right", va="center",
        fontsize=12, fontweight="bold")
ax.text((top[0] + E[0]) / 2, top[1] + 0.03, r"$\sqrt{L^{2}-p^{2}}$", color="#0a7a5c",
        ha="center", va="bottom", fontsize=12)
ax.text((O[0] + E[0]) / 2 + 0.04, (O[1] + E[1]) / 2 - 0.02, "L", color=BLUE,
        ha="left", va="top", fontsize=13, fontweight="bold")
ax.add_patch(Circle(O, 0.012, fc="k"))
ax.text(O[0], O[1] - 0.05, "shoulder", ha="center", va="top", fontsize=8.5, color="#555")

# ---- panel 4: recovered true angle ----
ax = axes[3]
sh = stick(ax, 0.42, 0.30, arm_to=(0.74, 0.60), arm_color=BLUE, arm_lw=4)
# trunk reference + angle arc between trunk (down) and arm
ax.add_patch(Arc(sh, 0.34, 0.34, angle=0, theta1=270, theta2=377.35, color=GREEN, lw=2.2))
ax.text(sh[0] + 0.16, sh[1] - 0.13, r"$\theta$", color="#0a7a5c", fontsize=14, ha="center")
ax.text(0.80, 0.74, "✓", color=GREEN, fontsize=20, ha="center", va="center", fontweight="bold")

# ---- arrows linking the four steps ----
for xb in (0.262, 0.508, 0.754):
    fig.patches.append(FancyArrowPatch((xb - 0.012, 0.50), (xb + 0.012, 0.50),
                       transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=20, color="#666", lw=2.0))

dest = Path(__file__).resolve().parents[1] / 'figures'
dest.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(dest / f"fig1.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print("wrote", dest / "fig1.{png,svg}")
