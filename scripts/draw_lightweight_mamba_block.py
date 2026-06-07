"""
Vẽ sơ đồ Lightweight Mamba Block (Conv 1×1 → LN → Mamba → SiLU → Conv 1×1 + residual).

Chạy:
    python scripts/draw_lightweight_mamba_block.py
    python scripts/draw_lightweight_mamba_block.py -o figures/my_diagram.png

Cần: matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#E3F2FD",
    ec: str = "#1565C0",
) -> tuple[float, float, float, float]:
    """Trả về (x, y, w, h) — góc dưới-trái."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#0D47A1",
        linespacing=1.15,
    )
    return (x, y, w, h)


def _arrow(ax, x1, y1, x2, y2, color="#424242", lw=1.2, z=2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(arr)


def _dashed_polyline(ax, pts, color="#C62828", lw=1.5, z=1):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=(0, (5, 4)), zorder=z, solid_capstyle="round")


def draw_diagram(
    out_path: Path,
    dpi: int = 200,
    figsize: tuple[float, float] = (7.5, 10.0),
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        5,
        12.1,
        "Lightweight Mamba Block",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#212121",
    )
    ax.text(
        5,
        11.55,
        "B, C, H, W — một nhánh residual",
        ha="center",
        va="center",
        fontsize=9,
        color="#616161",
    )

    w, h = 3.2, 0.65
    xc = 5.0 - w / 2
    gap = 0.45

    # Cột chính (giữa)
    y = 10.2
    b_in = _box(ax, xc, y, w, h, "Input  x  (B, C, H, W)", fc="#FFF8E1", ec="#F57F17")
    y -= h + gap
    b_c1 = _box(ax, xc, y, w, h, "Conv2d 1×1  (in_proj)")
    y -= h + gap
    b_ln = _box(ax, xc, y, w, h, "LayerNorm  (theo kênh C)")
    y -= h + gap
    b_m = _box(ax, xc, y, w, h, "Mamba / SS2D\n(B, H·W, C)")
    y -= h + gap
    b_a = _box(ax, xc, y, w, h, "SiLU")
    y -= h + gap
    b_c2 = _box(ax, xc, y, w, h, "Conv2d 1×1  (out_proj)")
    y -= h + gap
    b_dp = _box(ax, xc, y, w, h, "DropPath  (train)", fc="#F3E5F5", ec="#7B1FA2")
    y -= h + gap
    b_add = _box(ax, xc, y, w, h, "⊕  Residual", fc="#E8F5E9", ec="#2E7D32")
    y -= h + gap
    b_out = _box(ax, xc, y, w, h, "Output  (B, C, H, W)", fc="#FFF8E1", ec="#F57F17")

    def bottom_center(b):
        x, y0, bw, bh = b
        return x + bw / 2, y0

    def top_center(b):
        x, y0, bw, bh = b
        return x + bw / 2, y0 + bh

    # Luồng chính: Input → … → DropPath → ⊕ → Output (nhánh tính toán)
    main_chain = [b_in, b_c1, b_ln, b_m, b_a, b_c2, b_dp, b_add, b_out]
    for i in range(len(main_chain) - 1):
        x1, y1 = bottom_center(main_chain[i])
        x2, y2 = top_center(main_chain[i + 1])
        _arrow(ax, x1, y1 - 0.02, x2, y2 + 0.02)

    # Shortcut (identity): tách tại **đáy Input** (cùng điểm với mũi tên xuống Conv1×1)
    x_main, y_fork = bottom_center(b_in)
    y_add_cy = b_add[1] + b_add[3] / 2
    x_left = max(0.85, xc - 1.35)

    _dashed_polyline(
        ax,
        [(x_main, y_fork), (x_left, y_fork), (x_left, y_add_cy)],
        color="#C62828",
        lw=1.55,
        z=1,
    )
    _arrow(
        ax,
        x_left + 0.04,
        y_add_cy,
        b_add[0] + 0.06,
        y_add_cy,
        color="#C62828",
        lw=1.55,
        z=3,
    )
    ax.text(
        x_left - 0.12,
        (y_fork + y_add_cy) / 2,
        "Shortcut\n(identity)",
        ha="right",
        va="center",
        fontsize=8,
        color="#C62828",
        fontweight="medium",
        zorder=4,
    )

    ax.text(
        5,
        0.35,
        "Permute (B,C,H,W) ↔ (B,H,W,C) quanh LayerNorm — trong code: einops.rearrange",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#757575",
        style="italic",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Đã lưu: {out_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Vẽ sơ đồ Lightweight Mamba Block")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures") / "lightweight_mamba_block.png",
        help="Đường dẫn file ảnh (.png / .pdf)",
    )
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    draw_diagram(args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
