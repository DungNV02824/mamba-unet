"""
Vẽ sơ đồ chi tiết SS2D (Selective Scan 2D) trong Mamba-UNet.
Bao gồm:
  • Panel trái  – Luồng tổng quan SS2D (2D → flatten → Mamba → reshape 2D)
  • Panel giữa – Bên trong Mamba: x_proj, conv1d, SiLU, SSM (A/B/C/D), gate, out_proj
  • Panel phải  – Chi tiết SSM: phương trình trạng thái rời rạc + selective scan

Chạy:
    python scripts/draw_ss2d_detailed.py
    python scripts/draw_ss2d_detailed.py -o figures/ss2d_detailed.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse, FancyArrow
from matplotlib.lines import Line2D
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
            fontweight=weight, linespacing=1.25, zorder=4)
    return x + w / 2, y + h / 2   # center


def _arrow(ax, x1, y1, x2, y2,
           color="#1565C0", lw=1.3, ms=11, z=5, style="-|>"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color,
        shrinkA=2, shrinkB=2, zorder=z
    )
    ax.add_patch(arr)


def _dashed_arrow(ax, x1, y1, x2, y2, color="#B71C1C", lw=1.2, ms=10):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color,
        connectionstyle="arc3,rad=0",
        linestyle="dashed",
        shrinkA=2, shrinkB=2, zorder=5
    )
    ax.add_patch(arr)


def _label(ax, x, y, text, fontsize=7.5, color="#37474F", ha="center", va="center",
           style="normal", weight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            color=color, style=style, fontweight=weight, zorder=6)


def _section_bg(ax, x, y, w, h, color="#F3E5F5", alpha=0.4, lw=1.0, ec="#7B1FA2"):
    ax.add_patch(Rectangle((x, y), w, h,
                            facecolor=color, edgecolor=ec,
                            linewidth=lw, linestyle="--",
                            alpha=alpha, zorder=1))


def _circle_op(ax, cx, cy, r, symbol, fc="#FFF9C4", ec="#F57F17", fontsize=11):
    """Vẽ vòng tròn toán tử (+, ×, …)"""
    circ = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                       linewidth=1.4, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, symbol, ha="center", va="center",
            fontsize=fontsize, color="#E65100", zorder=4, fontweight="bold")


def _tensor_label(ax, x, y, shape_text, color="#004D40", fontsize=7):
    ax.text(x, y, shape_text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.1", fc="#E0F2F1", ec="#004D40", lw=0.8),
            zorder=6)


def _draw_2d_grid(ax, cx, cy, rows=4, cols=4, cell=0.045,
                  color="#1565C0", fc="#BBDEFB", label=""):
    """Vẽ lưới 2D nhỏ đại diện feature map"""
    ox = cx - cols * cell / 2
    oy = cy - rows * cell / 2
    for r in range(rows):
        for c in range(cols):
            rect = Rectangle(
                (ox + c * cell, oy + r * cell),
                cell, cell,
                facecolor=fc, edgecolor=color,
                linewidth=0.5, zorder=3
            )
            ax.add_patch(rect)
    if label:
        ax.text(cx, oy - 0.02, label, ha="center", va="top",
                fontsize=6.5, color="#0D47A1", style="italic", zorder=5)


def _draw_token_seq(ax, cx, cy, n=8, cell_w=0.05, cell_h=0.06,
                    color="#1B5E20", fc="#C8E6C9", label=""):
    """Vẽ chuỗi token 1D"""
    total_w = n * cell_w
    ox = cx - total_w / 2
    for i in range(n):
        rect = Rectangle(
            (ox + i * cell_w, cy - cell_h / 2),
            cell_w, cell_h,
            facecolor=fc, edgecolor=color,
            linewidth=0.5, zorder=3
        )
        ax.add_patch(rect)
    ax.text(ox + total_w + 0.01, cy, "...", fontsize=9, color=color,
            va="center", zorder=5)
    if label:
        ax.text(cx, cy - cell_h / 2 - 0.02, label, ha="center", va="top",
                fontsize=6.5, color="#1B5E20", style="italic", zorder=5)


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — Tổng quan SS2D
# ═══════════════════════════════════════════════════════════════════

def draw_panel_overview(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # Title
    ax.text(2.0, 13.6, "SS2D — Tổng quan luồng dữ liệu",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1A237E")
    ax.add_patch(Rectangle((0.0, 13.3), 4.0, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── Input feature map 2D ──
    _draw_2d_grid(ax, 2.0, 12.7, rows=4, cols=4, cell=0.07,
                  fc="#BBDEFB", label="")
    _box(ax, 0.5, 12.3, 3.0, 0.55,
         "Input Feature Map\n(B, C, H, W)",
         fc="#E3F2FD", ec="#1565C0", fontsize=8.5, bold=True)

    _arrow(ax, 2.0, 12.3, 2.0, 11.7)
    _tensor_label(ax, 2.8, 12.0, "(B, C, H, W)")

    # ── Rearrange 2D → 1D ──
    _box(ax, 0.5, 11.2, 3.0, 0.45,
         "Rearrange  (b c h w) → (b (h w) c)\nFlatten spatial → sequence",
         fc="#FFF3E0", ec="#E65100", fontsize=7.5)

    _arrow(ax, 2.0, 11.2, 2.0, 10.65)
    _tensor_label(ax, 2.9, 10.93, "(B, H·W, C)")

    # ── Token sequence visual ──
    _draw_token_seq(ax, 2.0, 10.42, n=9, cell_w=0.055, cell_h=0.07,
                    fc="#C8E6C9", label="")
    _label(ax, 2.0, 10.22, "Sequence of H·W tokens, mỗi token dim=C",
           fontsize=7, color="#2E7D32")

    _arrow(ax, 2.0, 10.16, 2.0, 9.6)
    _tensor_label(ax, 2.9, 9.88, "(B, L, C)  L=H·W")

    # ── Mamba SSM block ──
    _section_bg(ax, 0.3, 8.85, 3.4, 0.9, color="#EDE7F6", ec="#4527A0")
    _box(ax, 0.5, 8.95, 3.0, 0.70,
         "Mamba  (SSM — Selective State Space Model)\n"
         "d_model=C · d_state=16 · d_conv=3 · expand=2",
         fc="#EDE7F6", ec="#4527A0", fontsize=8, bold=True, tc="#1A237E")
    _label(ax, 3.5, 9.31, "← chi tiết\n  Panel 2",
           fontsize=7, color="#4527A0", ha="left")

    _arrow(ax, 2.0, 8.95, 2.0, 8.38)
    _tensor_label(ax, 2.9, 8.67, "(B, L, C)  L=H·W")

    # ── Rearrange 1D → 2D ──
    _box(ax, 0.5, 7.88, 3.0, 0.45,
         "Rearrange  (b (h w) c) → (b c h w)\nRestore spatial structure",
         fc="#FFF3E0", ec="#E65100", fontsize=7.5)

    _arrow(ax, 2.0, 7.88, 2.0, 7.35)
    _tensor_label(ax, 2.9, 7.62, "(B, C, H, W)")

    # ── Output ──
    _draw_2d_grid(ax, 2.0, 7.08, rows=4, cols=4, cell=0.07, fc="#C8E6C9")
    _box(ax, 0.5, 6.65, 3.0, 0.55,
         "Output Feature Map\n(B, C, H, W)",
         fc="#E8F5E9", ec="#2E7D32", fontsize=8.5, bold=True, tc="#1B5E20")

    # ── Add back in VSSBlock (context) ──
    ax.add_patch(Rectangle((0.25, 5.85), 3.5, 0.55,
                            facecolor="#FCE4EC", edgecolor="#C62828",
                            linewidth=1.0, linestyle="--",
                            alpha=0.6, zorder=2))
    _label(ax, 2.0, 6.12,
           "VSSBlock: shortcut + drop_path(SS2D(LN(x)))",
           fontsize=7.5, color="#C62828", weight="bold")
    _arrow(ax, 2.0, 6.65, 2.0, 6.42, color="#C62828")

    # ── Math note ──
    ax.text(0.15, 5.55,
            "Ghi chú toán học:\n"
            " •  L = H × W  (flatten không gian)\n"
            " •  Mỗi token $x_t \\in \\mathbb{R}^C$\n"
            " •  Mamba xử lý chuỗi causal: $h_t = A\\,h_{t-1} + B\\,x_t$\n"
            " •  Output: $y_t = C\\,h_t + D\\,x_t$  (rồi qua gate)",
            fontsize=7.8, color="#263238", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F1F8E9",
                      ec="#558B2F", lw=0.8))

    # ── Scan direction note ──
    ax.text(0.15, 4.12,
            "Lưu ý hướng quét:\n"
            " Flatten (h w) → hàng từ trái→phải, trên→dưới\n"
            " (raster scan order)\n"
            " → token 0: pixel (0,0), token 1: pixel (0,1), …",
            fontsize=7.5, color="#37474F", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="#E8EAF6",
                      ec="#3949AB", lw=0.8))

    # ── Pixel index illustration ──
    cxg, cyg = 2.5, 3.18
    g_cell = 0.10
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            fc_cell = "#1565C0" if idx == 0 else (
                "#42A5F5" if idx < 5 else "#BBDEFB")
            tc_cell = "white" if idx < 5 else "#0D47A1"
            rect = Rectangle(
                (cxg + c * g_cell - 0.15, cyg + (2 - r) * g_cell - 0.15),
                g_cell, g_cell,
                facecolor=fc_cell, edgecolor="#1565C0",
                linewidth=0.6, zorder=3
            )
            ax.add_patch(rect)
            ax.text(cxg + c * g_cell - 0.15 + g_cell / 2,
                    cyg + (2 - r) * g_cell - 0.15 + g_cell / 2,
                    str(idx),
                    ha="center", va="center",
                    fontsize=6.5, color=tc_cell, zorder=4)
    _arrow(ax, 2.77, 3.04, 3.5, 3.04, color="#1565C0")
    ax.text(3.55, 3.04, "[0,1,2,3,4,5,6,7,8]",
            va="center", fontsize=6.8, color="#1565C0", zorder=5)

    # border
    ax.add_patch(Rectangle((0, 0), 4, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — Bên trong Mamba
# ═══════════════════════════════════════════════════════════════════

def draw_panel_mamba_internals(ax):
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # Title
    ax.text(2.25, 13.6, "Bên trong Mamba Block",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1A237E")
    ax.add_patch(Rectangle((0.0, 13.3), 4.5, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── Input ──
    _box(ax, 0.75, 12.7, 3.0, 0.45,
         "Input  x  :  (B, L, D)    D = d_model = C",
         fc="#E3F2FD", ec="#1565C0", fontsize=8, bold=True)
    cx_main = 2.25

    # ── in_proj (Linear expand) ──
    _arrow(ax, cx_main, 12.7, cx_main, 12.12)
    _box(ax, 0.75, 11.65, 3.0, 0.42,
         "in_proj  :  Linear(D → 2·D·expand)\n(split → x_branch, z_branch)",
         fc="#FFF3E0", ec="#E65100", fontsize=7.5)
    _tensor_label(ax, 3.7, 11.87, "(B,L,2·E)\nE=D·expand")

    # split arrow
    _label(ax, cx_main, 11.37,
           "split along last dim", fontsize=7, color="#E65100")
    # x branch
    _arrow(ax, 1.1, 11.65, 0.85, 10.98, color="#1565C0")
    _label(ax, 0.48, 11.32, "x\n(B,L,E)", fontsize=7, color="#1565C0")
    # z branch
    _arrow(ax, 3.4, 11.65, 3.65, 10.98, color="#9C27B0")
    _label(ax, 3.88, 11.32, "z\n(B,L,E)", fontsize=7, color="#9C27B0")

    # ── Conv1D (depthwise, d_conv=3) ──
    _box(ax, 0.2, 10.52, 2.0, 0.42,
         "Conv1D  (depthwise)\nkernel_size=3, groups=E",
         fc="#E8EAF6", ec="#3949AB", fontsize=7.2)
    _tensor_label(ax, 0.35, 10.38, "(B,L,E)")

    # ── SiLU activation ──
    _arrow(ax, 1.2, 10.52, 1.2, 10.03)
    _box(ax, 0.2, 9.60, 2.0, 0.40,
         "SiLU  ( x · σ(x) )",
         fc="#FBE9E7", ec="#BF360C", fontsize=8)

    # ── x_proj: B, C, Δ ──
    _arrow(ax, 1.2, 9.60, 1.2, 9.08)
    _box(ax, 0.2, 8.62, 2.0, 0.42,
         "x_proj  :  Linear(E → dt_rank+2·N)\n→ split: Δ, B_ssm, C_ssm",
         fc="#E0F2F1", ec="#00695C", fontsize=7.2)
    _tensor_label(ax, 0.35, 8.55, "Δ:(B,L,dt)\nB:(B,L,N)\nC:(B,L,N)")

    # ── dt_proj (Δ → E) ──
    _arrow(ax, 1.2, 8.62, 1.2, 8.13)
    _box(ax, 0.2, 7.67, 2.0, 0.42,
         "dt_proj  :  Linear(dt_rank → E)\n+ Softplus(Δ)  →  Δ ∈ ℝ⁺",
         fc="#E0F2F1", ec="#00695C", fontsize=7.2)

    # ── SSM ──
    _section_bg(ax, 0.08, 6.62, 2.32, 1.35, color="#FFF9C4", ec="#F9A825")
    _label(ax, 1.22, 7.87, "SSM Core", fontsize=8, color="#E65100", weight="bold")
    _arrow(ax, 1.2, 7.67, 1.2, 7.3)
    _box(ax, 0.2, 6.72, 2.0, 0.55,
         "Selective Scan\nA (E×N learned)\nΔ,B_ssm,C_ssm (input-dependent)",
         fc="#FFF9C4", ec="#F9A825", fontsize=7.2)
    _label(ax, 0.12, 7.0,
           "y = SSM(Δ,A,B,C) · x", fontsize=7, color="#F57F17", ha="left")
    _label(ax, 3.2, 7.26, "← Panel 3\n  (detail)", fontsize=7,
           color="#F57F17", ha="left")

    _arrow(ax, 1.2, 6.72, 1.2, 6.3)
    _box(ax, 0.2, 5.88, 2.0, 0.38,
         "D (skip): y += D · x",
         fc="#FFFDE7", ec="#F9A825", fontsize=7.5)

    _arrow(ax, 1.2, 5.88, 1.2, 5.38)
    # SiLU on SSM output
    _box(ax, 0.2, 4.96, 2.0, 0.38,
         "SiLU  ( y_ssm · σ(y_ssm) )",
         fc="#FBE9E7", ec="#BF360C", fontsize=7.5)

    # ── Gate multiply ──
    _arrow(ax, 1.2, 4.96, 1.2, 4.55)
    _circle_op(ax, 1.2, 4.35, 0.18, "×", fc="#FFF9C4", ec="#F57F17")
    # z branch to gate
    _arrow(ax, 3.65, 10.98, 3.65, 4.35, color="#9C27B0", lw=1.2)
    _box(ax, 2.8, 4.75, 1.6, 0.42,
         "SiLU(z)\ngate branch",
         fc="#F3E5F5", ec="#7B1FA2", fontsize=7.2, tc="#4A148C")
    _arrow(ax, 2.8, 4.96, 1.38, 4.35, color="#9C27B0")

    # ── out_proj ──
    _arrow(ax, 1.2, 4.17, 1.2, 3.68)
    _box(ax, 0.2, 3.26, 2.0, 0.38,
         "out_proj  :  Linear(E → D)",
         fc="#FFF3E0", ec="#E65100", fontsize=7.5)
    _tensor_label(ax, 0.35, 3.15, "(B, L, D)")

    # ── Output ──
    _arrow(ax, 1.2, 3.26, 1.2, 2.78)
    _box(ax, 0.2, 2.38, 2.0, 0.38,
         "Output  y  :  (B, L, D)",
         fc="#E8F5E9", ec="#2E7D32", fontsize=8, bold=True, tc="#1B5E20")

    # ── Residual (internal) ──
    ax.annotate("",
                xy=(0.2, 2.57), xytext=(0.05, 12.9),
                arrowprops=dict(arrowstyle="-|>", color="#C62828",
                                lw=1.2, linestyle="dashed",
                                connectionstyle="arc3,rad=0"),
                zorder=5)
    _label(ax, 0.0, 7.75, "Internal\nresidual\n(optional)", fontsize=6.5,
           color="#C62828", ha="left")

    # ── Dimension legend ──
    ax.text(0.15, 1.98,
            "Ký hiệu:\n"
            "  D  = d_model (= C, số kênh đầu vào)\n"
            "  E  = D × expand  (expand=2)\n"
            "  N  = d_state (=16, chiều ẩn SSM)\n"
            "  L  = H × W (độ dài chuỗi)\n"
            "  dt_rank = ceil(D/16)  (rank của Δ)\n"
            "  σ  = sigmoid;  SiLU(x)=x·σ(x)",
            fontsize=7.5, color="#263238", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="#F9FBE7",
                      ec="#827717", lw=0.8))

    ax.add_patch(Rectangle((0, 0), 4.5, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — Chi tiết SSM (phương trình + selective scan)
# ═══════════════════════════════════════════════════════════════════

def draw_panel_ssm_detail(ax):
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    # Title
    ax.text(2.25, 13.6, "Chi tiết SSM (Selective State Space)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1A237E")
    ax.add_patch(Rectangle((0.0, 13.3), 4.5, 0.02,
                            facecolor="#1A237E", edgecolor="none"))

    # ── 1. Phương trình liên tục (S4 / Mamba continuous) ──
    _section_bg(ax, 0.15, 12.1, 4.2, 1.35, color="#E8EAF6", ec="#3949AB")
    _label(ax, 2.25, 13.28, "① Phương trình SSM liên tục",
           fontsize=9, color="#1A237E", weight="bold")

    ax.text(0.35, 13.05,
            r"$h'(t) = A\,h(t) + B\,x(t)$",
            fontsize=10, color="#1A237E", va="center")
    ax.text(0.35, 12.72,
            r"$y(t)   = C\,h(t) + D\,x(t)$",
            fontsize=10, color="#1A237E", va="center")
    ax.text(0.35, 12.38,
            r"$A \in \mathbb{R}^{N \times N},\;"
            r"B \in \mathbb{R}^{N \times 1},\;"
            r"C \in \mathbb{R}^{1 \times N}$",
            fontsize=8, color="#37474F", va="center")

    # ── 2. Rời rạc hóa (ZOH) ──
    _section_bg(ax, 0.15, 10.55, 4.2, 1.42, color="#E0F2F1", ec="#00695C")
    _label(ax, 2.25, 11.80, "② Rời rạc hóa (Zero-Order Hold, Δ adaptive)",
           fontsize=9, color="#004D40", weight="bold")

    ax.text(0.35, 11.58,
            r"$\bar{A} = \exp(\Delta \cdot A)$",
            fontsize=10, color="#004D40", va="center")
    ax.text(0.35, 11.22,
            r"$\bar{B} = (\Delta \cdot A)^{-1}(\bar{A}-I)\cdot\Delta \cdot B$",
            fontsize=9.5, color="#004D40", va="center")
    ax.text(0.35, 10.86,
            r"$\Delta \in \mathbb{R}^{+E}$ — input-dependent (selective!)",
            fontsize=8, color="#37474F", va="center")

    # ── 3. Recurrence rời rạc ──
    _section_bg(ax, 0.15, 9.0, 4.2, 1.38, color="#FFF8E1", ec="#F9A825")
    _label(ax, 2.25, 10.22, "③ Recurrence rời rạc (mỗi bước thời gian t)",
           fontsize=9, color="#E65100", weight="bold")

    ax.text(0.35, 9.98,
            r"$h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t$",
            fontsize=10.5, color="#E65100", va="center")
    ax.text(0.35, 9.62,
            r"$y_t = C_t\,h_t + D\,x_t$",
            fontsize=10.5, color="#E65100", va="center")
    ax.text(0.35, 9.28,
            r"$h_t \in \mathbb{R}^N$ — hidden state,  "
            r"$C_t$ phụ thuộc đầu vào",
            fontsize=8, color="#37474F", va="center")

    # ── 4. Sơ đồ unrolled recurrence ──
    _section_bg(ax, 0.15, 6.75, 4.2, 2.1, color="#FCE4EC", ec="#C62828")
    _label(ax, 2.25, 8.68, "④ Sơ đồ Unrolled Selective Scan",
           fontsize=9, color="#B71C1C", weight="bold")

    # states h
    hy = 7.95
    hxs = [0.55, 1.55, 2.55, 3.55]
    for i, hx in enumerate(hxs):
        lbl = f"$h_{{{i}}}$"
        _box(ax, hx - 0.22, hy - 0.18, 0.44, 0.36,
             lbl, fc="#FFCDD2", ec="#C62828", fontsize=9, tc="#B71C1C")

    # arrows h_t → h_{t+1}
    for i in range(len(hxs) - 1):
        _arrow(ax, hxs[i] + 0.22, hy,
               hxs[i + 1] - 0.22, hy, color="#C62828", lw=1.3)
        _label(ax, (hxs[i] + hxs[i + 1]) / 2, hy + 0.22,
               r"$\bar{A}$", fontsize=9, color="#C62828")

    # inputs x and B labels
    xy_in = 7.35
    for i, hx in enumerate(hxs):
        _box(ax, hx - 0.22, xy_in - 0.18, 0.44, 0.36,
             f"$x_{{{i}}}$", fc="#E3F2FD", ec="#1565C0", fontsize=9, tc="#0D47A1")
        _arrow(ax, hx, xy_in + 0.18, hx, hy - 0.18, color="#1565C0")
        _label(ax, hx + 0.14, (xy_in + hy) / 2, r"$\bar{B}_t$",
               fontsize=8, color="#1565C0")

    # outputs y
    yy_out = 8.60
    for i, hx in enumerate(hxs):
        _box(ax, hx - 0.22, yy_out - 0.18, 0.44, 0.36,
             f"$y_{{{i}}}$", fc="#E8F5E9", ec="#2E7D32", fontsize=9, tc="#1B5E20")
        _arrow(ax, hx, hy + 0.18, hx, yy_out - 0.18, color="#2E7D32")
        _label(ax, hx + 0.14, (hy + yy_out) / 2, r"$C_t$",
               fontsize=8, color="#2E7D32")

    _label(ax, 4.1, 8.0, "…", fontsize=14, color="#C62828")

    # ── 5. Selectivity (Δ, B, C phụ thuộc x) ──
    _section_bg(ax, 0.15, 5.3, 4.2, 1.3, color="#E8F5E9", ec="#2E7D32")
    _label(ax, 2.25, 6.44, "⑤ Tính Selective — điểm khác biệt vs S4",
           fontsize=9, color="#1B5E20", weight="bold")

    ax.text(0.3, 6.22,
            "S4 (classic):  A, B, C  cố định, không phụ thuộc input\n"
            "Mamba:         Δ, B, C = Linear(x_t)  →  phụ thuộc x_t\n"
            "                → model tự chọn thông tin nào cần nhớ",
            fontsize=8, color="#263238", va="top")

    # ── 6. Parallel scan (training) vs recurrence (inference) ──
    _section_bg(ax, 0.15, 3.75, 4.2, 1.4, color="#FFF3E0", ec="#E65100")
    _label(ax, 2.25, 4.98, "⑥ Training vs Inference",
           fontsize=9, color="#E65100", weight="bold")

    ax.text(0.3, 4.78,
            "Training  : Parallel associative scan  O(L log L)\n"
            "            — CUDA kernel (causal_conv1d + mamba_ssm)\n"
            "Inference : Recurrence step-by-step  O(L)\n"
            "            — chỉ cần lưu h_t (không cần full sequence)",
            fontsize=7.8, color="#37474F", va="top")

    # ── 7. Kích thước trong code ──
    _section_bg(ax, 0.15, 1.85, 4.2, 1.72, color="#E3F2FD", ec="#1565C0")
    _label(ax, 2.25, 3.40, "⑦ Kích thước tensor cụ thể (C=96, H=W=64)",
           fontsize=9, color="#1A237E", weight="bold")

    ax.text(0.3, 3.18,
            "Input x:       (B, 64×64=4096, 96)\n"
            "in_proj:        → (B, 4096, 192×2=384)\n"
            "x/z branch:    (B, 4096, 192)  each\n"
            "conv1d:         (B, 4096, 192)\n"
            "x_proj:         → Δ:(B,4096,6) B_ssm:(B,4096,16) C_ssm:(B,4096,16)\n"
            "hidden h:       (B, 192, 16)  per step\n"
            "out_proj:       (B, 4096, 96)",
            fontsize=7.5, color="#0D47A1", va="top",
            family="monospace")

    ax.add_patch(Rectangle((0, 0), 4.5, 14,
                            facecolor="none", edgecolor="#90A4AE",
                            linewidth=1.5, zorder=0))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output",
                    default="figures/ss2d_detailed.png",
                    help="Đường dẫn file ảnh output")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(16, 10.5),
        gridspec_kw={"width_ratios": [1, 1.1, 1.1]},
        facecolor="#ECEFF1"
    )

    fig.suptitle(
        "SS2D — Selective Scan 2D  (Chi tiết hoàn chỉnh)",
        fontsize=14, fontweight="bold", color="#0D47A1", y=0.985
    )

    draw_panel_overview(ax1)
    draw_panel_mamba_internals(ax2)
    draw_panel_ssm_detail(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=0.8)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved → {out}")


if __name__ == "__main__":
    main()
