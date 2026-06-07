"""
Publication-quality figure of SS2D architecture as implemented in code.

Code sources:
  • models/vss_block.py   → class SS2D, class VSSBlock
  • mamba_ssm library     → class Mamba (used inside SS2D)

Layout (3 columns):
  Col 1 – VSSBlock   : 3 sequential residual branches (LN+SS2D | DWConv | LN+MLP)
  Col 2 – SS2D       : 2-D feature map → flatten → Mamba → reshape 2-D
  Col 3 – Mamba block: in_proj / conv1d / SiLU / x_proj / SSM / gate / out_proj

Run:
    python scripts/draw_ss2d_paper.py
    python scripts/draw_ss2d_paper.py -o figures/ss2d_paper.png --dpi 300
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


# ─────────────────────────── global style ────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.spines.left":   False,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.spines.bottom": False,
    "text.usetex":       False,
})

# ─────────────────────────── palette ─────────────────────────────

C = {
    # (facecolor, edgecolor)
    "io":     ("#DBEAFE", "#1D4ED8"),   # blue  – input / output
    "ln":     ("#FEF3C7", "#B45309"),   # amber – LayerNorm
    "ss2d":   ("#EDE9FE", "#6D28D9"),   # violet – SS2D / Mamba
    "conv":   ("#FCE7F3", "#9D174D"),   # pink  – Conv / DWConv
    "mlp":    ("#D1FAE5", "#065F46"),   # green – MLP / Linear
    "act":    ("#FFF7ED", "#C2410C"),   # orange – activations
    "add":    ("#FEF9C3", "#92400E"),   # yellow – residual add
    "drop":   ("#F1F5F9", "#475569"),   # slate  – DropPath
    "param":  ("#F0FDF4", "#166534"),   # light green – learned params
    "ssm":    ("#FDF4FF", "#7E22CE"),   # purple – SSM core
    "gate":   ("#FFF1F2", "#9F1239"),   # rose   – gate
    "arrow":  "#374151",
    "skip":   "#DC2626",
    "dim":    "#4B5563",
}


# ─────────────────────────── helpers ─────────────────────────────

def _box(ax, x, y, w, h, text,
         fc="#DBEAFE", ec="#1D4ED8",
         fontsize=8, tc="#1E3A8A",
         bold=False, lw=1.3, radius=0.025, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=tc,
            fontweight="bold" if bold else "normal",
            linespacing=1.35, zorder=zorder + 1)
    return x + w / 2, y + h / 2


def _arr(ax, x1, y1, x2, y2,
         color="#374151", lw=1.25, ms=9, rad=0.0, z=6, style="-|>"):
    cs = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color,
        connectionstyle=cs,
        shrinkA=2, shrinkB=2, zorder=z,
    ))


def _darr(ax, x1, y1, x2, y2, color="#DC2626", lw=1.1, ms=8, rad=0.0):
    cs = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        connectionstyle=cs,
        linestyle="dashed",
        shrinkA=2, shrinkB=2, zorder=6,
    ))


def _plus(ax, cx, cy, r=0.17):
    ax.add_patch(plt.Circle(
        (cx, cy), r,
        facecolor=C["add"][0], edgecolor=C["add"][1],
        linewidth=1.3, zorder=5,
    ))
    ax.text(cx, cy, "+", ha="center", va="center",
            fontsize=11, color="#78350F", fontweight="bold", zorder=6)


def _dim(ax, x, y, text, ha="center"):
    ax.text(x, y, text,
            ha=ha, va="center",
            fontsize=6.5, color=C["dim"], style="italic",
            bbox=dict(boxstyle="round,pad=0.1",
                      fc="#F8FAFC", ec="#CBD5E1", lw=0.5),
            zorder=7)


def _label(ax, x, y, text, fontsize=7.5, color="#374151",
           ha="center", va="center", bold=False):
    ax.text(x, y, text, ha=ha, va=va,
            fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal", zorder=7)


def _divider(ax, y, x0=0.1, x1=None, lw=0.6, color="#CBD5E1"):
    xlim = ax.get_xlim()
    if x1 is None:
        x1 = xlim[1] - 0.1
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls="--", zorder=1)


def _section_bg(ax, x, y, w, h,
                fc="#F8FAFC", ec="#94A3B8", lw=0.8, alpha=0.5, ls="--"):
    ax.add_patch(Rectangle((x, y), w, h,
                            facecolor=fc, edgecolor=ec,
                            linewidth=lw, linestyle=ls,
                            alpha=alpha, zorder=1))


# ═══════════════════════════════════════════════════════════════
# PANEL 1  ─  VSSBlock  (models/vss_block.py → class VSSBlock)
# ═══════════════════════════════════════════════════════════════

def draw_vssblock(ax):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── title ──
    ax.text(2.5, 15.6,
            "VSSBlock",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1E3A8A")
    ax.text(2.5, 15.2,
            "(models/vss_block.py)",
            ha="center", va="center",
            fontsize=7.5, color="#6B7280", style="italic")
    ax.plot([0.2, 4.8], [14.95, 14.95], color="#1E3A8A", lw=1.8)

    BW = 3.4   # box width
    BX = 0.8   # box left x
    CX = BX + BW / 2   # box center x = 2.5

    # ── Input ──
    _box(ax, BX, 14.3, BW, 0.45, "Input  x",
         *C["io"], fontsize=9, bold=True, tc="#1E3A8A")
    _dim(ax, CX + BW / 2 + 0.25, 14.52, "(B, C, H, W)")

    # ══ BRANCH 1 : LN + SS2D ══════════════════════════════════
    _arr(ax, CX, 14.3, CX, 13.82)

    # skip1 tap  ←  taken from input
    _section_bg(ax, 0.15, 11.5, 4.7, 2.72, fc="#EEF2FF", ec="#818CF8")
    _label(ax, 1.15, 14.14, "Branch 1", fontsize=7, color="#4338CA", bold=True, ha="left")

    # shortcut branch drawn as dashed line on the right
    ax.plot([4.55, 4.55], [14.52, 11.95],
            color=C["skip"], lw=1.0, ls="--", zorder=5)
    _darr(ax, 4.55, 11.95, 4.02, 11.95, color=C["skip"])
    _label(ax, 4.65, 13.24, "skip₁", fontsize=6.8, color=C["skip"],
           ha="left")

    _box(ax, BX, 13.35, BW, 0.43,
         "LayerNorm  (per-channel, applied in h-w-c space)",
         *C["ln"], fontsize=7.5, tc="#78350F")
    _dim(ax, CX + BW / 2 + 0.25, 13.56, "(B, H, W, C)")

    _arr(ax, CX, 13.35, CX, 12.85)
    _box(ax, BX, 12.38, BW, 0.45,
         "SS2D  (d_model=C, d_state=16, d_conv=3, expand=2)",
         *C["ss2d"], fontsize=7.5, tc="#4C1D95", bold=True)
    _dim(ax, CX + BW / 2 + 0.25, 12.61, "(B, C, H, W)")

    _arr(ax, CX, 12.38, CX, 11.92)

    # DropPath
    _box(ax, BX, 11.45, BW, 0.43,
         "DropPath  (stochastic depth, p=drop_path_rate)",
         *C["drop"], fontsize=7.5, tc="#334155")

    _arr(ax, CX, 11.45, CX, 11.05)
    _plus(ax, CX, 10.88)
    _arr(ax, CX, 10.71, CX, 10.28)

    # ══ BRANCH 2 : DWConv + BN + GELU ════════════════════════
    _section_bg(ax, 0.15, 8.55, 4.7, 1.72, fc="#FDF4FF", ec="#C084FC")
    _label(ax, 1.15, 10.19, "Branch 2", fontsize=7, color="#7E22CE", bold=True, ha="left")

    ax.plot([4.55, 4.55], [10.56, 8.9],
            color=C["skip"], lw=1.0, ls="--", zorder=5)
    _darr(ax, 4.55, 8.9, 4.02, 8.9, color=C["skip"])
    _label(ax, 4.65, 9.73, "skip₂", fontsize=6.8, color=C["skip"], ha="left")

    _box(ax, BX, 9.80, BW, 0.43,
         r"DWConv2d  (kernel 3×3, groups=C, bias=False)",
         *C["conv"], fontsize=7.5, tc="#831843")
    _arr(ax, CX, 9.80, CX, 9.37)
    _box(ax, BX, 8.90, BW, 0.43,
         "BatchNorm2d  +  GELU",
         *C["act"], fontsize=7.5, tc="#9A3412")
    _dim(ax, CX + BW / 2 + 0.25, 9.56, "(B, C, H, W)")

    _arr(ax, CX, 8.90, CX, 8.50)
    _plus(ax, CX, 8.33)
    _arr(ax, CX, 8.16, CX, 7.75)

    # ══ BRANCH 3 : LN + MLP ══════════════════════════════════
    _section_bg(ax, 0.15, 5.55, 4.7, 2.18, fc="#F0FDF4", ec="#4ADE80")
    _label(ax, 1.15, 7.65, "Branch 3", fontsize=7, color="#065F46", bold=True, ha="left")

    ax.plot([4.55, 4.55], [8.01, 6.0],
            color=C["skip"], lw=1.0, ls="--", zorder=5)
    _darr(ax, 4.55, 6.0, 4.02, 6.0, color=C["skip"])
    _label(ax, 4.65, 7.01, "skip₃", fontsize=6.8, color=C["skip"], ha="left")

    _box(ax, BX, 7.25, BW, 0.43,
         "LayerNorm  (per-channel, applied in h-w-c space)",
         *C["ln"], fontsize=7.5, tc="#78350F")
    _arr(ax, CX, 7.25, CX, 6.82)
    _box(ax, BX, 6.38, BW, 0.42,
         "Linear (C → 4C)  +  GELU  +  Linear (4C → C)",
         *C["mlp"], fontsize=7.5, tc="#064E3B")
    _dim(ax, CX + BW / 2 + 0.25, 6.79, "(B, H, W, C)")

    _arr(ax, CX, 6.38, CX, 5.97)
    _box(ax, BX, 5.50, BW, 0.43,
         "DropPath  (stochastic depth, p=drop_path_rate)",
         *C["drop"], fontsize=7.5, tc="#334155")

    _arr(ax, CX, 5.50, CX, 5.10)
    _plus(ax, CX, 4.93)
    _arr(ax, CX, 4.76, CX, 4.35)

    # ── Output ──
    _box(ax, BX, 3.88, BW, 0.43, "Output  y",
         *C["io"], fontsize=9, bold=True, tc="#1E3A8A")
    _dim(ax, CX + BW / 2 + 0.25, 4.10, "(B, C, H, W)")

    # ── notation box ──
    ax.text(0.25, 3.50,
            "Notation:  C = hidden_dim  |  B = batch  |  H,W = spatial",
            fontsize=7, color="#374151", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC",
                      ec="#CBD5E1", lw=0.7))

    ax.text(0.25, 3.00,
            "Flow:  x  →  [LN → SS2D → +skip] "
            "→  [DWConv+BN+GELU → +skip]  →  [LN → MLP → +skip]  →  y",
            fontsize=7.2, color="#1E3A8A", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="#EFF6FF",
                      ec="#93C5FD", lw=0.7))

    # outer border
    ax.add_patch(Rectangle((0, 0), 5, 16,
                            facecolor="none", edgecolor="#94A3B8",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════
# PANEL 2  ─  SS2D  (models/vss_block.py → class SS2D)
# ═══════════════════════════════════════════════════════════════

def draw_ss2d(ax):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(2.5, 15.6, "SS2D Module",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#4C1D95")
    ax.text(2.5, 15.2,
            "(models/vss_block.py → class SS2D)",
            ha="center", va="center",
            fontsize=7.5, color="#6B7280", style="italic")
    ax.plot([0.2, 4.8], [14.95, 14.95], color="#4C1D95", lw=1.8)

    BW = 3.5
    BX = 0.75
    CX = BX + BW / 2   # 2.5

    # ── Input feature map ──────────────────────────────────────
    _box(ax, BX, 14.30, BW, 0.45,
         "Input  x  :  (B, C, H, W)",
         *C["io"], fontsize=8.5, bold=True, tc="#1E3A8A")

    # grid illustration
    g = 0.07
    ox, oy = 1.78, 13.75
    for r in range(4):
        for cc in range(4):
            fc_g = "#93C5FD" if (r + cc) % 2 == 0 else "#BFDBFE"
            ax.add_patch(Rectangle(
                (ox + cc * g, oy + r * g), g, g,
                facecolor=fc_g, edgecolor="#1D4ED8", linewidth=0.4, zorder=3))
    ax.text(ox + 4 * g + 0.04, oy + 2 * g,
            "Feature map\n(H × W grid)", fontsize=6, color="#1E3A8A", va="center")

    _arr(ax, CX, 14.30, CX, 13.56)
    _dim(ax, CX + 1.0, 13.93, "(B, C, H, W)")

    # ── rearrange 2-D → 1-D ───────────────────────────────────
    _box(ax, BX, 13.08, BW, 0.44,
         "rearrange:  'b c h w  →  b (h w) c'\n"
         "(flatten H×W into sequence length L = H·W)",
         *C["act"], fontsize=7.5, tc="#7C2D12")
    _arr(ax, CX, 13.08, CX, 12.53)
    _dim(ax, CX + 1.0, 12.80, "(B, L, C)  L=H·W")

    # token sequence illustration
    tw, th = 0.065, 0.075
    tx0 = 1.55
    for i in range(7):
        fc_t = "#4ADE80" if i == 0 else ("#86EFAC" if i < 4 else "#BBF7D0")
        ax.add_patch(Rectangle((tx0 + i * tw, 12.24), tw, th,
                                facecolor=fc_t, edgecolor="#166534",
                                linewidth=0.4, zorder=3))
    ax.text(tx0 + 7 * tw + 0.02, 12.275, "···",
            fontsize=9, color="#166534", va="center")
    ax.text(tx0 + 8.5 * tw + 0.02, 12.275, "(L tokens)",
            fontsize=6.5, color="#166534", va="center", style="italic")

    _arr(ax, CX, 12.21, CX, 11.60)
    _dim(ax, CX + 1.0, 11.90, "(B, L, C)")

    # ── Mamba block (the core SSM) ────────────────────────────
    _section_bg(ax, 0.2, 9.92, 4.6, 1.70,
                fc="#F5F3FF", ec="#7C3AED", lw=1.2, alpha=0.55, ls="-")
    _label(ax, 2.5, 11.44,
           "Mamba  (mamba_ssm.Mamba)", fontsize=9.5,
           color="#4C1D95", bold=True)

    _box(ax, 1.1, 11.0, 2.8, 0.38,
         "d_model = C   |   d_state = 16", *C["ss2d"], fontsize=7.5, tc="#4C1D95")
    _box(ax, 1.1, 10.56, 2.8, 0.38,
         "d_conv = 3   |   expand = 2", *C["ss2d"], fontsize=7.5, tc="#4C1D95")
    _label(ax, 4.42, 10.94,
           "← details\n  Panel 3", fontsize=6.5, color="#7C3AED", ha="left")

    _arr(ax, CX, 9.92, CX, 9.37)
    _dim(ax, CX + 1.0, 9.65, "(B, L, C)")

    # ── rearrange 1-D → 2-D ───────────────────────────────────
    _box(ax, BX, 8.90, BW, 0.44,
         "rearrange:  'b (h w) c  →  b c h w'\n"
         "(restore spatial dimensions H, W)",
         *C["act"], fontsize=7.5, tc="#7C2D12")

    _arr(ax, CX, 8.90, CX, 8.38)
    _dim(ax, CX + 1.0, 8.64, "(B, C, H, W)")

    # output grid
    ox2, oy2 = 1.78, 7.85
    for r in range(4):
        for cc in range(4):
            fc_g = "#4ADE80" if (r + cc) % 2 == 0 else "#86EFAC"
            ax.add_patch(Rectangle(
                (ox2 + cc * g, oy2 + r * g), g, g,
                facecolor=fc_g, edgecolor="#166534", linewidth=0.4, zorder=3))
    ax.text(ox2 + 4 * g + 0.04, oy2 + 2 * g,
            "Output map\n(H × W grid)", fontsize=6, color="#065F46", va="center")

    # ── Output ──
    _box(ax, BX, 7.47, BW, 0.43,
         "Output  x_out  :  (B, C, H, W)",
         *C["io"], fontsize=8.5, bold=True, tc="#1E3A8A")

    # ── scan order note ───────────────────────────────────────
    ax.text(0.25, 6.90,
            "Scan order (raster):\n"
            "  token 0 = pixel (0,0),  token 1 = pixel (0,1),  …\n"
            "  token k = pixel (k//W, k%W)",
            fontsize=7.5, color="#374151", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#FEFCE8", ec="#FDE047", lw=0.8))

    # pixel index grid
    g2 = 0.11
    ox3, oy3 = 2.85, 5.35
    labels = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for r in range(3):
        for cc in range(3):
            idx = labels[2 - r][cc]
            dark = idx < 4
            ax.add_patch(Rectangle(
                (ox3 + cc * g2, oy3 + r * g2), g2, g2,
                facecolor="#1D4ED8" if dark else "#93C5FD",
                edgecolor="#1E3A8A", linewidth=0.5, zorder=3))
            ax.text(ox3 + cc * g2 + g2 / 2,
                    oy3 + r * g2 + g2 / 2,
                    str(idx), ha="center", va="center",
                    fontsize=6, color="white" if dark else "#1E3A8A", zorder=4)
    _arr(ax, ox3 + 3 * g2 + 0.02, oy3 + 1.5 * g2,
         ox3 + 3 * g2 + 0.35, oy3 + 1.5 * g2, color="#1D4ED8")
    ax.text(ox3 + 3 * g2 + 0.38, oy3 + 1.5 * g2,
            "[0, 1, 2, 3, 4, 5, 6, 7, 8 …]",
            va="center", fontsize=6.5, color="#1D4ED8")

    ax.text(0.25, 5.05,
            r"$L = H \times W$   (flatten spatial)",
            fontsize=8, color="#1E3A8A", ha="left",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#EFF6FF", ec="#93C5FD", lw=0.7))

    ax.add_patch(Rectangle((0, 0), 5, 16,
                            facecolor="none", edgecolor="#94A3B8",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════
# PANEL 3  ─  Mamba Internals  (mamba_ssm library)
# ═══════════════════════════════════════════════════════════════

def draw_mamba_internals(ax):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(2.5, 15.6, "Mamba Block Internals",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="#6D28D9")
    ax.text(2.5, 15.2,
            "(mamba_ssm.Mamba — called by SS2D)",
            ha="center", va="center",
            fontsize=7.5, color="#6B7280", style="italic")
    ax.plot([0.2, 4.8], [14.95, 14.95], color="#6D28D9", lw=1.8)

    BW = 2.8
    BX = 0.5
    # main (x-branch) column center
    MX = BX + BW / 2   # 1.9

    # ── Input ──────────────────────────────────────────────────
    _box(ax, BX, 14.32, BW, 0.42,
         "Input  x  :  (B, L, D)     D = d_model = C",
         *C["io"], fontsize=7.5, bold=True, tc="#1E3A8A")

    _arr(ax, MX, 14.32, MX, 13.72)
    _dim(ax, MX + 1.55, 14.02, "(B, L, D)")

    # ── in_proj ────────────────────────────────────────────────
    _box(ax, BX, 13.26, BW, 0.42,
         "in_proj  :  Linear(D → 2·E)\n(E = D × expand = 2D)",
         *C["mlp"], fontsize=7.5, tc="#064E3B")
    _dim(ax, MX + 1.55, 13.47, "(B, L, 2E)")

    # split
    _arr(ax, MX, 13.26, MX - 0.5, 12.87, color="#1D4ED8")
    _arr(ax, MX, 13.26, MX + 1.55, 12.87, color="#9D174D")
    _label(ax, MX, 13.10, "split last dim", fontsize=6.5, color="#374151")

    # x-branch label
    _label(ax, MX - 0.7, 12.75, "x_branch", fontsize=6.8, color="#1D4ED8")
    _dim(ax, MX - 0.7, 12.60, "(B, L, E)")

    # z-branch label
    _label(ax, MX + 1.78, 12.75, "z_branch", fontsize=6.8, color="#9D174D")
    _dim(ax, MX + 1.78, 12.60, "(B, L, E)")

    # z-branch vertical line
    ZX = 4.25   # z branch column x
    ax.plot([ZX, ZX], [12.54, 6.62],
            color="#9D174D", lw=1.0, ls="--", zorder=5)

    # ── Conv1d (depthwise) ─────────────────────────────────────
    _arr(ax, MX, 12.52, MX, 12.08)
    _box(ax, BX, 11.62, BW, 0.42,
         "Conv1d  (depthwise, kernel=d_conv=3, groups=E)",
         *C["conv"], fontsize=7.5, tc="#831843")
    _dim(ax, MX + 1.55, 11.83, "(B, L, E)")

    # ── SiLU ──────────────────────────────────────────────────
    _arr(ax, MX, 11.62, MX, 11.22)
    _box(ax, BX, 10.77, BW, 0.42,
         r"SiLU  ( x · $\sigma$(x) )",
         *C["act"], fontsize=8, tc="#9A3412")
    _dim(ax, MX + 1.55, 10.98, "(B, L, E)")

    # ── x_proj: generate Δ, B, C ──────────────────────────────
    _arr(ax, MX, 10.77, MX, 10.37)
    _box(ax, BX, 9.92, BW, 0.42,
         "x_proj  :  Linear(E → dt_rank + 2·N)\n"
         "→ split → dt_raw (dt_rank)  |  B (N)  |  C (N)",
         *C["mlp"], fontsize=7.2, tc="#064E3B")

    _dim(ax, MX + 1.55, 10.20, "dt_rank=ceil(D/16)")
    _dim(ax, MX + 1.55, 10.05, "N = d_state = 16")

    # ── dt_proj: Δ ─────────────────────────────────────────────
    _arr(ax, MX, 9.92, MX, 9.53)
    _box(ax, BX, 9.07, BW, 0.42,
         "dt_proj  :  Linear(dt_rank → E)  +  Softplus\n"
         r"→ $\Delta > 0$   (input-dependent timescale)",
         *C["mlp"], fontsize=7.2, tc="#064E3B")
    _dim(ax, MX + 1.55, 9.28, r"$\Delta$ : (B, L, E)")

    # ── SSM core ──────────────────────────────────────────────
    _section_bg(ax, 0.15, 6.72, 3.55, 2.45,
                fc="#FDF4FF", ec="#7E22CE", lw=1.2, alpha=0.55, ls="-")
    _label(ax, 1.95, 9.00, "Selective SSM Core", fontsize=8.5,
           color="#6D28D9", bold=True)

    _arr(ax, MX, 9.07, MX, 8.75)

    # A param
    ax.text(0.20, 8.70,
            "A  (E x N)\n(learned, fixed)",
            fontsize=7.5, color="#166534", va="top",
            bbox=dict(boxstyle="round,pad=0.12",
                      fc=C["param"][0], ec=C["param"][1], lw=0.8))
    _darr(ax, 1.22, 8.62, 1.55, 8.30, color="#166534", lw=0.9)

    # discretize equation
    ax.text(0.25, 8.52,
            r"$\bar{A} = \exp(\Delta \cdot A)$",
            fontsize=8.5, color="#4C1D95", va="center")
    ax.text(0.25, 8.22,
            r"$\bar{B} = \Delta \cdot B$",
            fontsize=8.5, color="#4C1D95", va="center")

    _box(ax, BX, 7.67, BW, 0.50,
         r"Selective scan:  $h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t$" + "\n" +
         r"                      $y_t = C_t\,h_t$",
         *C["ssm"], fontsize=8, tc="#581C87")

    _arr(ax, MX, 7.67, MX, 7.23)
    _box(ax, BX, 6.77, BW, 0.42,
         r"D-skip:  $y = y + D \cdot x$   (D learned, scalar per channel)",
         *C["param"], fontsize=7.5, tc="#166534")
    _dim(ax, MX + 1.55, 6.98, "y : (B, L, E)")

    # ── SiLU on z-branch + gate ───────────────────────────────
    _arr(ax, MX, 6.77, MX, 6.36)

    # draw gate
    _box(ax, BX, 5.92, BW, 0.40,
         r"Gate:  $y = y_{ssm} \odot SiLU(z_{branch})$",
         *C["gate"], fontsize=8, tc="#881337")

    # z-branch arrow into gate
    _arr(ax, ZX, 6.62, 3.30, 6.12, color="#9D174D")
    _label(ax, 3.60, 6.52, r"SiLU(z)", fontsize=7, color="#9D174D")
    _dim(ax, MX + 1.55, 6.18, "(B, L, E)")

    # ── out_proj ───────────────────────────────────────────────
    _arr(ax, MX, 5.92, MX, 5.52)
    _box(ax, BX, 5.07, BW, 0.42,
         "out_proj  :  Linear(E → D)",
         *C["mlp"], fontsize=7.5, tc="#064E3B")
    _dim(ax, MX + 1.55, 5.28, "(B, L, D)")

    # ── Output ──────────────────────────────────────────────────
    _arr(ax, MX, 5.07, MX, 4.65)
    _box(ax, BX, 4.20, BW, 0.42,
         "Output  y  :  (B, L, D)",
         *C["io"], fontsize=8.5, bold=True, tc="#1E3A8A")

    # ── Dimension legend ─────────────────────────────────────
    ax.text(0.20, 3.85,
            "Symbols:\n"
            "  D = d_model = C  (input channels)\n"
            "  E = D × expand = 2D  (inner dim)\n"
            "  N = d_state = 16  (SSM hidden dim)\n"
            "  L = H × W  (sequence length)\n"
            r"  $\sigma$ = sigmoid;  SiLU(x) = x·$\sigma$(x)",
            fontsize=7.8, color="#1F2937", va="top",
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

    ax.text(0.20, 1.95,
            "Key innovation — Selectivity:\n"
            r"  S4:  A, B, C  fixed (independent of input)" + "\n"
            r"  Mamba:  $\Delta$, B, C = Linear($x_t$)  (input-dependent)" + "\n"
            "  → model dynamically chooses what to remember",
            fontsize=7.8, color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="#FEFCE8", ec="#FDE047", lw=0.8))

    ax.add_patch(Rectangle((0, 0), 5, 16,
                            facecolor="none", edgecolor="#94A3B8",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output",
                    default="figures/ss2d_paper.png",
                    help="Output image path")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Resolution (300 recommended for paper)")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]},
        facecolor="white",
    )

    fig.suptitle(
        "Selective Scan 2D (SS2D) Architecture — As Implemented",
        fontsize=14, fontweight="bold", color="#0F172A", y=0.995,
    )

    draw_vssblock(ax1)
    draw_ss2d(ax2)
    draw_mamba_internals(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.995], pad=0.6, w_pad=0.3)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved  →  {out}   (dpi={args.dpi})")


if __name__ == "__main__":
    main()
