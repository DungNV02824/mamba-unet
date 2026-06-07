"""
Vẽ kiến trúc SS2D như đã implement trong code thực tế.

Panels:
  Panel 1 – class SS2D          (models/vss_block.py)
  Panel 2 – class VSSBlock      (models/vss_block.py) — 3 skip connections
  Panel 3 – Mamba internals     (mamba_ssm library, được gọi bởi SS2D)

Chạy:
    python scripts/draw_ss2d_actual.py
    python scripts/draw_ss2d_actual.py -o figures/ss2d_actual.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


# ─────────────────────────── helpers ────────────────────────────

def _box(ax, x, y, w, h, text,
         fc="#E3F2FD", ec="#1565C0", fontsize=8.5, tc="#0D47A1",
         bold=False, lw=1.5, radius=0.04):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=tc,
            fontweight=weight, linespacing=1.3, zorder=4)
    return x + w / 2, y + h / 2   # center


def _arrow(ax, x1, y1, x2, y2, color="#1565C0", lw=1.3, ms=11, z=5):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        shrinkA=2, shrinkB=2, zorder=z
    )
    ax.add_patch(arr)


def _line(ax, x1, y1, x2, y2, color="#C62828", lw=1.2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=4)


def _label(ax, x, y, text, fontsize=7.5, color="#37474F",
           ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=color, fontweight=weight, style=style, zorder=6)


def _tensor(ax, x, y, text, fontsize=7.0, color="#004D40"):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=fontsize, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.1", fc="#E0F2F1",
                      ec="#004D40", lw=0.7),
            zorder=7)


def _circle_op(ax, cx, cy, r=0.18, sym="+",
               fc="#FFF9C4", ec="#F57F17"):
    circ = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                       linewidth=1.5, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, sym, ha="center", va="center",
            fontsize=11, color="#E65100", fontweight="bold", zorder=4)


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — class SS2D  (vss_block.py)
# ═══════════════════════════════════════════════════════════════════

def draw_panel_ss2d(ax):
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    cx = 2.0
    bx = 0.45
    bw = 3.1

    # ── Title ──
    ax.text(2.0, 13.65, "class  SS2D",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1A237E")
    ax.text(2.0, 13.28, "models/vss_block.py",
            ha="center", va="center", fontsize=8, color="#546E7A",
            style="italic")
    ax.add_patch(Rectangle((0, 13.15), 4, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── Input ──
    _box(ax, bx, 12.35, bw, 0.55,
         "Input  x  :  (B, C, H, W)",
         fc="#E3F2FD", ec="#1565C0", fontsize=8.5, bold=True)
    _arrow(ax, cx, 12.35, cx, 11.77)
    _tensor(ax, 2.95, 12.06, "(B, C, H, W)")

    # ── rearrange flatten ──
    _box(ax, bx, 11.3, bw, 0.44,
         "rearrange  'b c h w  →  b (h w) c'\n(flatten không gian → sequence)",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 11.3, cx, 10.73)
    _tensor(ax, 2.95, 11.01, "(B, H·W, C)")

    # ── Mamba ──
    ax.add_patch(Rectangle((bx - 0.05, 9.8), bw + 0.1, 1.0,
                            facecolor="#EDE7F6", edgecolor="#4527A0",
                            linewidth=0.8, linestyle="--", alpha=0.4, zorder=1))
    _box(ax, bx, 9.88, bw, 0.82,
         "Mamba  (mamba_ssm)\n"
         "d_model = C   d_state = 16\n"
         "d_conv  = 3   expand  = 2",
         fc="#EDE7F6", ec="#4527A0", fontsize=8.2,
         tc="#1A237E", bold=False)
    ax.text(3.65, 10.27, "→ chi tiết\n   Panel 3",
            fontsize=7, color="#4527A0", va="center", ha="left", zorder=6)
    _arrow(ax, cx, 9.88, cx, 9.3)
    _tensor(ax, 2.95, 9.59, "(B, H·W, C)")

    # ── rearrange reshape ──
    _box(ax, bx, 8.83, bw, 0.44,
         "rearrange  'b (h w) c  →  b c h w'  (h=H, w=W)\n(khôi phục không gian 2D)",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 8.83, cx, 8.27)
    _tensor(ax, 2.95, 8.55, "(B, C, H, W)")

    # ── Output ──
    _box(ax, bx, 7.82, bw, 0.45,
         "Output  x_out  :  (B, C, H, W)",
         fc="#E8F5E9", ec="#2E7D32", fontsize=8.5, bold=True, tc="#1B5E20")

    # ── Code snippet ──
    ax.add_patch(Rectangle((0.1, 5.08), 3.8, 2.55,
                            facecolor="#ECEFF1", edgecolor="#607D8B",
                            linewidth=0.9, zorder=2))
    _label(ax, 0.22, 7.45, "Code thực tế (forward):",
           fontsize=7.8, color="#263238", ha="left", weight="bold")
    ax.text(0.2, 7.28,
            "def forward(self, x):\n"
            "    B, C, H, W = x.shape\n"
            "    x_flat = rearrange(\n"
            "        x, 'b c h w -> b (h w) c')\n"
            "    x_out = self.mamba(x_flat)\n"
            "    x_out = rearrange(\n"
            "        x_out,\n"
            "        'b (h w) c -> b c h w',\n"
            "        h=H, w=W)\n"
            "    return x_out",
            fontsize=7.2, color="#37474F", va="top",
            family="monospace", zorder=5)

    # ── Note single-direction scan ──
    ax.text(0.1, 4.85,
            "Lưu ý: SS2D trong code này dùng  1 hướng quét\n"
            "(raster scan: trái→phải, trên→dưới)\n"
            "Không có multi-direction scan như một số biến thể.",
            fontsize=7.5, color="#B71C1C", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFEBEE",
                      ec="#C62828", lw=0.8), zorder=5)

    # ── Init params ──
    ax.text(0.1, 3.68,
            "Init parameters:\n"
            "  d_model  = C      (kênh đầu vào)\n"
            "  d_state  = 16     (chiều hidden SSM)\n"
            "  d_conv   = 3      (kernel conv1d)\n"
            "  expand   = 2      (expansion factor)\n"
            "  → E = C × 2  (inner dimension)",
            fontsize=7.5, color="#263238", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc="#E8F5E9",
                      ec="#2E7D32", lw=0.8), zorder=5)

    ax.add_patch(Rectangle((0, 0), 4, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — class VSSBlock  (3 skip connections)
# ═══════════════════════════════════════════════════════════════════

def draw_panel_vssblock(ax):
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    cx   = 1.75   # main flow centre x
    bx   = 0.3    # box left x
    bw   = 2.9    # box width
    sx   = 3.72   # skip line x (right side)
    ar   = 0.17   # Add-circle radius
    sc   = "#C62828"  # skip / shortcut colour

    # ── Title ──
    ax.text(2.25, 13.65, "class  VSSBlock  —  3 skip connections",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#1A237E")
    ax.text(2.25, 13.28, "models/vss_block.py",
            ha="center", va="center", fontsize=8, color="#546E7A",
            style="italic")
    ax.add_patch(Rectangle((0, 13.15), 4.5, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── Input ──
    _box(ax, bx, 12.42, bw, 0.52,
         "Input  x  :  (B, C, H, W)",
         fc="#E3F2FD", ec="#1565C0", fontsize=8.5, bold=True)
    _arrow(ax, cx, 12.42, cx, 11.88)

    # ═══════════════════════════════
    # BRANCH 1 background
    ax.add_patch(Rectangle((0.18, 8.75), 3.38, 3.52,
                            facecolor="#EDE7F6", edgecolor="#4527A0",
                            linewidth=0.8, linestyle="--", alpha=0.18, zorder=1))
    _label(ax, 3.7, 12.14, "Branch 1\n(SS2D)", fontsize=7,
           color="#4527A0", ha="center", weight="bold")

    # skip_1 branch: from right edge of Input box
    _line(ax, bx + bw, 12.68, sx, 12.68, color=sc)   # horizontal
    _line(ax, sx, 12.68, sx, 8.97, color=sc)           # vertical down to Add1

    # Branch 1 operations
    _box(ax, bx, 11.42, bw, 0.42,
         "rearrange  'b c h w → b h w c'",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 11.42, cx, 11.0)

    _box(ax, bx, 10.58, bw, 0.38,
         "LayerNorm  (ln_1)   —  channel-last",
         fc="#F3E5F5", ec="#7B1FA2", fontsize=7.8)
    _arrow(ax, cx, 10.58, cx, 10.15)

    _box(ax, bx, 9.73, bw, 0.38,
         "rearrange  'b h w c → b c h w'",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 9.73, cx, 9.33)

    _box(ax, bx, 8.9, bw, 0.38,
         "SS2D  (Mamba wrapper)  →  Panel 1",
         fc="#EDE7F6", ec="#4527A0", fontsize=7.8)
    _arrow(ax, cx, 8.9, cx, 8.52)

    _box(ax, bx, 8.15, bw, 0.33,
         "DropPath",
         fc="#ECEFF1", ec="#607D8B", fontsize=7.5)
    _arrow(ax, cx, 8.15, cx, 7.8)

    # Add1 circle + skip arrow
    _circle_op(ax, cx, 7.62, ar)
    _label(ax, cx - 0.6, 7.62, "shortcut_1", fontsize=6.8, color=sc, ha="right")
    _arrow(ax, sx, 8.97, cx + ar, 7.62, color=sc)   # skip → Add1
    _arrow(ax, cx, 7.62 - ar, cx, 7.2)

    # ═══════════════════════════════
    # BRANCH 2 background
    ax.add_patch(Rectangle((0.18, 5.5), 3.38, 1.82,
                            facecolor="#E8F5E9", edgecolor="#2E7D32",
                            linewidth=0.8, linestyle="--", alpha=0.2, zorder=1))
    _label(ax, 3.7, 6.41, "Branch 2\n(DWConv)", fontsize=7,
           color="#2E7D32", ha="center", weight="bold")

    # skip_2 branch from Add1 level
    _line(ax, cx + ar, 7.62, sx, 7.62, color=sc)   # horizontal from Add1
    _line(ax, sx, 7.62, sx, 5.71, color=sc)

    # Branch 2 operations
    _box(ax, bx, 6.78, bw, 0.52,
         "DWConv2d (k=3, pad=1, groups=C)\n+ BatchNorm2d  +  GELU",
         fc="#E8F5E9", ec="#2E7D32", fontsize=7.8)
    _arrow(ax, cx, 6.78, cx, 6.4)

    _box(ax, bx, 6.03, bw, 0.33,
         "DropPath",
         fc="#ECEFF1", ec="#607D8B", fontsize=7.5)
    _arrow(ax, cx, 6.03, cx, 5.68)

    # Add2 circle + skip arrow
    _circle_op(ax, cx, 5.50, ar)
    _label(ax, cx - 0.6, 5.50, "shortcut_2", fontsize=6.8, color=sc, ha="right")
    _arrow(ax, sx, 5.71, cx + ar, 5.50, color=sc)
    _arrow(ax, cx, 5.50 - ar, cx, 5.1)

    # ═══════════════════════════════
    # BRANCH 3 background
    ax.add_patch(Rectangle((0.18, 1.75), 3.38, 3.47,
                            facecolor="#FFF9C4", edgecolor="#F9A825",
                            linewidth=0.8, linestyle="--", alpha=0.22, zorder=1))
    _label(ax, 3.7, 3.49, "Branch 3\n(MLP)", fontsize=7,
           color="#F57F17", ha="center", weight="bold")

    # skip_3 branch from Add2 level
    _line(ax, cx + ar, 5.50, sx, 5.50, color=sc)
    _line(ax, sx, 5.50, sx, 1.96, color=sc)

    # Branch 3 operations
    _box(ax, bx, 4.68, bw, 0.38,
         "rearrange  'b c h w → b h w c'",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 4.68, cx, 4.28)

    _box(ax, bx, 3.88, bw, 0.35,
         "LayerNorm  (ln_2)   —  channel-last",
         fc="#F3E5F5", ec="#7B1FA2", fontsize=7.8)
    _arrow(ax, cx, 3.88, cx, 3.5)

    _box(ax, bx, 3.12, bw, 0.35,
         "MLP:  Linear(C→4C)  →  GELU  →  Linear(4C→C)",
         fc="#FFF9C4", ec="#F9A825", fontsize=7.8)
    _arrow(ax, cx, 3.12, cx, 2.73)

    _box(ax, bx, 2.35, bw, 0.35,
         "rearrange  'b h w c → b c h w'  +  DropPath",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _arrow(ax, cx, 2.35, cx, 1.95)

    # Add3 circle + skip arrow
    _circle_op(ax, cx, 1.77, ar)
    _label(ax, cx - 0.6, 1.77, "shortcut_3", fontsize=6.8, color=sc, ha="right")
    _arrow(ax, sx, 1.96, cx + ar, 1.77, color=sc)
    _arrow(ax, cx, 1.77 - ar, cx, 1.32)

    # ── Output ──
    _box(ax, bx, 0.85, bw, 0.44,
         "Output  x  :  (B, C, H, W)",
         fc="#E8F5E9", ec="#2E7D32", fontsize=8.5, bold=True, tc="#1B5E20")

    # ── DropPath shared note ──
    ax.text(0.12, 0.62,
            "* Dùng chung 1 module DropPath cho cả 3 nhánh",
            fontsize=7, color="#607D8B", va="top", style="italic", zorder=5)

    ax.add_patch(Rectangle((0, 0), 4.5, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — Mamba internals  (mamba_ssm library)
# ═══════════════════════════════════════════════════════════════════

def draw_panel_mamba(ax):
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # branch centres
    cx_x = 1.3   # x-branch
    cx_z = 3.4   # z-branch (gate)
    bx_x = 0.22  # x-branch box left
    bw_x = 2.1   # x-branch box width
    bx_z = 2.45  # z-branch box left
    bw_z = 1.85

    # ── Title ──
    ax.text(2.25, 13.65, "Mamba — bên trong  mamba_ssm",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#1A237E")
    ax.text(2.25, 13.28, "Được gọi từ  SS2D.mamba(x_flat)",
            ha="center", va="center", fontsize=8, color="#546E7A",
            style="italic")
    ax.add_patch(Rectangle((0, 13.15), 4.5, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── Input ──
    _box(ax, 0.62, 12.42, 3.26, 0.52,
         "Input  x  :  (B, L, D)     L = H·W,  D = C",
         fc="#E3F2FD", ec="#1565C0", fontsize=8, bold=True)
    _arrow(ax, 2.25, 12.42, 2.25, 11.88)

    # ── in_proj split ──
    _box(ax, 0.62, 11.4, 3.26, 0.44,
         "in_proj  :  Linear(D  →  2·E)     E = D × 2\nsplit  →  x_branch,  z_branch",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)

    # split arrows
    _arrow(ax, 1.5,  11.4, cx_x, 10.9, color="#1565C0")
    _arrow(ax, 3.0,  11.4, cx_z, 10.9, color="#9C27B0")
    _label(ax, cx_x - 0.22, 11.16, "x_branch\n(B,L,E)",
           fontsize=7, color="#1565C0")
    _label(ax, cx_z + 0.22, 11.16, "z_branch\n(B,L,E)",
           fontsize=7, color="#9C27B0")

    # ────────── x-branch ──────────

    # conv1d
    _box(ax, bx_x, 10.45, bw_x, 0.42,
         "conv1d  depthwise  (k=3, groups=E)",
         fc="#E8EAF6", ec="#3949AB", fontsize=7.5)
    _arrow(ax, cx_x, 10.45, cx_x, 10.02)

    # SiLU
    _box(ax, bx_x, 9.60, bw_x, 0.38,
         "SiLU  ( x · σ(x) )",
         fc="#FBE9E7", ec="#BF360C", fontsize=7.8)
    _arrow(ax, cx_x, 9.60, cx_x, 9.18)

    # x_proj
    _box(ax, bx_x, 8.75, bw_x, 0.40,
         "x_proj : Linear(E → dt_rank + 2·N)\n→ split:  Δ,  B_ssm,  C_ssm",
         fc="#E0F2F1", ec="#00695C", fontsize=7.2)
    ax.text(0.04, 8.95,
            "Δ:(B,L,r)\nB:(B,L,N)\nC:(B,L,N)",
            fontsize=6.5, color="#004D40", va="center", style="italic", zorder=6)
    _arrow(ax, cx_x, 8.75, cx_x, 8.32)

    # dt_proj
    _box(ax, bx_x, 7.88, bw_x, 0.40,
         "dt_proj : Linear(dt_rank → E)\n+ Softplus(Δ)  →  Δ ∈ ℝ⁺",
         fc="#E0F2F1", ec="#00695C", fontsize=7.2)
    _arrow(ax, cx_x, 7.88, cx_x, 7.45)

    # SSM Core
    ax.add_patch(Rectangle((0.17, 5.95), 2.26, 1.7,
                            facecolor="#FFF9C4", edgecolor="#F9A825",
                            linewidth=1.0, linestyle="--", alpha=0.55, zorder=1))
    _label(ax, 1.3, 7.46, "SSM Core", fontsize=8,
           color="#E65100", weight="bold")
    _box(ax, bx_x, 6.98, bw_x, 0.44,
         "Ā = exp(Δ·A_log)    B̄ = ZOH(Δ, A, B)\nselective_scan(Ā, B̄, C_ssm, x_branch)",
         fc="#FFF9C4", ec="#F9A825", fontsize=7.0)
    ax.text(0.2, 6.75,
            "A_log: param (E,N)  learnable\n"
            "h_t = Ā·h_{t-1} + B̄·x_t\n"
            "y_t  = C_t · h_t",
            fontsize=6.5, color="#37474F", va="top",
            family="monospace", zorder=5)
    _arrow(ax, cx_x, 6.98, cx_x, 6.58)
    _arrow(ax, cx_x, 6.58, cx_x, 5.93)

    # D skip
    _box(ax, bx_x, 5.50, bw_x, 0.38,
         "D (skip):  y  +=  D · x_branch",
         fc="#FFFDE7", ec="#F9A825", fontsize=7.5)
    _arrow(ax, cx_x, 5.50, cx_x, 5.1)

    # ────────── gate multiply ──────────
    _circle_op(ax, cx_x, 4.92, 0.17, "×")

    # z-branch: vertical line then SiLU + gate
    _line(ax, cx_z, 10.88, cx_z, 5.28, color="#9C27B0")
    _box(ax, bx_z, 5.08, bw_z, 0.38,
         "SiLU(z_branch)\n(gate activation)",
         fc="#F3E5F5", ec="#7B1FA2", fontsize=7.5, tc="#4A148C")
    _arrow(ax, bx_z, 5.27, cx_x + 0.17, 4.92, color="#9C27B0")

    _arrow(ax, cx_x, 4.92 - 0.17, cx_x, 4.48)

    # out_proj
    _box(ax, bx_x, 4.08, bw_x, 0.36,
         "out_proj  :  Linear(E  →  D)",
         fc="#FFF3E0", ec="#E65100", fontsize=7.8)
    _tensor(ax, 2.38, 4.26, "(B, L, D)")
    _arrow(ax, cx_x, 4.08, cx_x, 3.68)

    # Output
    _box(ax, bx_x, 3.28, bw_x, 0.36,
         "Output  y  :  (B, L, D)",
         fc="#E8F5E9", ec="#2E7D32", fontsize=8, bold=True, tc="#1B5E20")

    # ── Dimension legend ──
    ax.text(0.18, 2.95,
            "Ký hiệu:\n"
            "  D = d_model = C       (kênh đầu vào)\n"
            "  E = D × expand = 2D   (inner dim)\n"
            "  N = d_state  = 16     (SSM hidden)\n"
            "  L = H × W             (sequence len)\n"
            "  r = dt_rank = ⌈D/16⌉  (rank Δ)\n"
            "  σ = sigmoid;  SiLU(x) = x·σ(x)",
            fontsize=7.5, color="#263238", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc="#F9FBE7",
                      ec="#827717", lw=0.8), zorder=5)

    ax.add_patch(Rectangle((0, 0), 4.5, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Vẽ kiến trúc SS2D theo code thực tế")
    ap.add_argument("-o", "--output",
                    default="figures/ss2d_actual.png",
                    help="Đường dẫn file ảnh output")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(17, 10.5),
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]},
        facecolor="#ECEFF1"
    )

    fig.suptitle(
        "Kiến trúc SS2D  —  Theo code thực tế  (models/vss_block.py)",
        fontsize=14, fontweight="bold", color="#0D47A1", y=0.985
    )

    draw_panel_ss2d(ax1)
    draw_panel_vssblock(ax2)
    draw_panel_mamba(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.8)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved → {out}")


if __name__ == "__main__":
    main()
