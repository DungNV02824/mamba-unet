"""
Vẽ sơ đồ Mamba-UNet theo phong cách hình mẫu (VM-UNet / paper):
chữ U, Encoder–Decoder khung nét đứt, stage nền xanh lá, bottleneck xanh nhạt,
kích thước H/4×… ở lề, Skip Connection có nhãn.
Decoder mỗi stage: Patch Expanding → Concat(+skip) → Linear 1×1 → VSS (khớp models/mamba_unet.DecoderStage).

Chạy:
    python scripts/draw_mamba_unet_architecture.py
    python scripts/draw_mamba_unet_architecture.py -o figures/mamba_unet_architecture.png

Cần: matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def _white_block(ax, x, y, w, h, text, ec="#1565C0", fontsize=8):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        facecolor="white",
        edgecolor=ec,
        linewidth=1.35,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0D47A1",
        linespacing=1.08,
    )
    return (x, y, w, h)


def _green_stage_bg(ax, x, y, w, h, pad=0.08):
    ax.add_patch(
        Rectangle(
            (x - pad, y - pad),
            w + 2 * pad,
            h + 2 * pad,
            facecolor="#C8E6C9",
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )


def _arrow(ax, x1, y1, x2, y2, color="#1565D8", lw=1.35):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(arr)


def _skip_arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.5,
        color="#1565D8",
        linestyle="-",
        shrinkA=3,
        shrinkB=3,
        zorder=3,
    )
    ax.add_patch(arr)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 + 0.15
    ax.text(
        mx,
        my,
        "Skip Connection",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#0D47A1",
        fontweight="medium",
        zorder=4,
    )


def draw_diagram(out_path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 11), dpi=dpi)
    ax.set_xlim(0, 13)
    ax.set_ylim(1.5, 14.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        6.5,
        13.15,
        "Mamba-UNet Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0D47A1",
    )

    x_enc = 2.15
    w = 2.35
    bh = 0.42
    gap = 0.14
    x_dec = 8.5
    x_mid_bn = 5.75
    w_bn = 3.5

    enc_frame = Rectangle(
        (1.05, 2.05),
        4.35,
        10.15,
        fill=False,
        edgecolor="#1565C0",
        linewidth=1.4,
        linestyle=(0, (6, 4)),
        zorder=1,
    )
    dec_frame = Rectangle(
        (7.6, 2.05),
        4.35,
        10.15,
        fill=False,
        edgecolor="#1565C0",
        linewidth=1.4,
        linestyle=(0, (6, 4)),
        zorder=1,
    )
    ax.add_patch(enc_frame)
    ax.add_patch(dec_frame)
    ax.text(3.2, 11.95, "Encoder", ha="center", fontsize=10, color="#1565C0", fontweight="bold")
    ax.text(9.75, 11.95, "Decoder", ha="center", fontsize=10, color="#1565C0", fontweight="bold")

    def cy(b):
        x, y, bw, bh_ = b
        return x + bw / 2, y + bh_ / 2

    # ===== Encoder: từ trên xuống =====
    y = 11.35
    b_img = _white_block(ax, x_enc, y, w, bh * 0.95, r"Image  $H \times W \times 1$")
    y -= bh * 0.95 + gap
    b_pp = _white_block(ax, x_enc, y, w, bh, "Patch Partition")
    y -= bh + gap

    # Stage 1: Linear + VSS×2
    h_s1 = bh * 2 + gap + 0.12
    y_s1_bot = y - h_s1
    _green_stage_bg(ax, x_enc, y_s1_bot, w, h_s1)
    y_b = y - bh
    b_lin = _white_block(ax, x_enc, y_b, w, bh, "Linear Embedding")
    y_b -= bh + gap
    b_vss1 = _white_block(ax, x_enc, y_b, w, bh, "VSS Block  $\\times 2$")
    y = y_s1_bot - gap
    ax.text(
        0.35,
        y_s1_bot + h_s1 / 2,
        r"$\frac{H}{4} \times \frac{W}{4} \times C$",
        ha="left",
        va="center",
        fontsize=9,
        color="#333",
    )

    # Stage 2
    h_s2 = h_s1
    y_s2_bot = y - h_s2
    _green_stage_bg(ax, x_enc, y_s2_bot, w, h_s2)
    y_b = y - bh
    b_m2 = _white_block(ax, x_enc, y_b, w, bh, "Patch Merging")
    y_b -= bh + gap
    b_vss2 = _white_block(ax, x_enc, y_b, w, bh, "VSS Block  $\\times 2$")
    y = y_s2_bot - gap
    ax.text(
        0.35,
        y_s2_bot + h_s2 / 2,
        r"$\frac{H}{8} \times \frac{W}{8} \times 2C$",
        ha="left",
        va="center",
        fontsize=9,
        color="#333",
    )

    # Stage 3
    h_s3 = h_s2
    y_s3_bot = y - h_s3
    _green_stage_bg(ax, x_enc, y_s3_bot, w, h_s3)
    y_b = y - bh
    b_m3 = _white_block(ax, x_enc, y_b, w, bh, "Patch Merging")
    y_b -= bh + gap
    b_vss3 = _white_block(ax, x_enc, y_b, w, bh, "VSS Block  $\\times 2$")
    y = y_s3_bot - gap
    ax.text(
        0.35,
        y_s3_bot + h_s3 / 2,
        r"$\frac{H}{16} \times \frac{W}{16} \times 4C$",
        ha="left",
        va="center",
        fontsize=9,
        color="#333",
    )

    # Stage 4: chỉ Patch Merging (theo hình mẫu)
    b_m4 = _white_block(ax, x_enc, y - bh, w, bh, "Patch Merging")
    y_bn_top = b_m4[1] - gap - 0.55
    ax.text(
        0.35,
        b_m4[1] + bh / 2,
        r"$\rightarrow \frac{H}{32} \times \frac{W}{32} \times 8C$",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333",
    )

    # Bottleneck (nền xanh nhạt)
    h_bn = 0.95
    y_bn = y_bn_top - h_bn
    ax.add_patch(
        Rectangle(
            (x_mid_bn - 0.1, y_bn),
            w_bn + 0.2,
            h_bn,
            facecolor="#BBDEFB",
            edgecolor="#1976D2",
            linewidth=1.3,
            alpha=0.85,
            zorder=1,
        )
    )
    bx = x_mid_bn + 0.25
    bw = (w_bn - 0.5) / 2 - 0.1
    _white_block(ax, bx, y_bn + 0.22, bw, bh * 0.9, "VSS Block  $\\times 1$", fontsize=7.8)
    _white_block(ax, bx + bw + 0.15, y_bn + 0.22, bw, bh * 0.9, "VSS Block  $\\times 1$", fontsize=7.8)
    ax.text(
        x_mid_bn + w_bn / 2,
        y_bn + 0.06,
        r"$\frac{H}{32} \times \frac{W}{32} \times 8C$",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#0D47A1",
    )

    # Mũi tên encoder dọc
    for a, b in [
        (b_img, b_pp),
        (b_pp, b_lin),
        (b_vss1, b_m2),
        (b_vss2, b_m3),
        (b_vss3, b_m4),
    ]:
        x1, y1 = a[0] + a[2] / 2, a[1]
        x2, y2 = b[0] + b[2] / 2, b[1] + b[3]
        _arrow(ax, x1, y1 - 0.02, x2, y2 + 0.02)
    for a, b in [(b_lin, b_vss1), (b_m2, b_vss2), (b_m3, b_vss3)]:
        x1, y1 = a[0] + a[2] / 2, a[1]
        x2, y2 = b[0] + b[2] / 2, b[1] + b[3]
        _arrow(ax, x1, y1 - 0.02, x2, y2 + 0.02)
    x1, y1 = b_m4[0] + b_m4[2] / 2, b_m4[1]
    x2, y2 = x_mid_bn + w_bn / 2, y_bn + h_bn
    _arrow(ax, x1, y1 - 0.02, x2, y2 + 0.04)

    # ===== Decoder: căn tâm Concat theo VSS encoder cùng tầng; skip ngang (bản cũ gọn)
    y_d3_c = cy(b_vss3)[1]
    y_d2_c = cy(b_vss2)[1]
    y_d1_c = cy(b_vss1)[1]

    bh_d = bh * 0.82
    g_d = max(gap, 0.16)

    def decoder_stage_boxes(y_concat_center, depth_txt):
        """Expand → Concat(+skip) → Linear 1×1 → VSS (khớp DecoderStage)."""
        y_cat_lo = y_concat_center - bh_d / 2
        y_exp_lo = y_cat_lo - g_d - bh_d
        y_lin_lo = y_cat_lo + bh_d + g_d
        y_vss_lo = y_lin_lo + bh_d + g_d
        h_stage = (y_vss_lo + bh_d) - y_exp_lo
        _green_stage_bg(ax, x_dec, y_exp_lo, w, h_stage, pad=0.06)
        b_de = _white_block(ax, x_dec, y_exp_lo, w, bh_d, "Patch Expanding", fontsize=7.8)
        b_cat = _white_block(
            ax,
            x_dec,
            y_cat_lo,
            w,
            bh_d,
            "Concat\n(+ skip)",
            fontsize=7.4,
        )
        b_lin = _white_block(
            ax,
            x_dec,
            y_lin_lo,
            w,
            bh_d,
            "Linear 1×1\n(Conv2d)",
            fontsize=7.4,
        )
        b_dv = _white_block(
            ax,
            x_dec,
            y_vss_lo,
            w,
            bh_d,
            f"VSS Block  {depth_txt}",
            fontsize=7.8,
        )
        return b_de, b_cat, b_lin, b_dv

    b_de3, b_cat3, b_lin3, b_dv3 = decoder_stage_boxes(y_d3_c, r"$\times 2$")
    ax.text(
        12.65,
        y_d3_c,
        r"$\frac{H}{16} \times \frac{W}{16} \times 4C$",
        ha="right",
        va="center",
        fontsize=9,
        color="#333",
    )

    b_de2, b_cat2, b_lin2, b_dv2 = decoder_stage_boxes(y_d2_c, r"$\times 2$")
    ax.text(
        12.65,
        y_d2_c,
        r"$\frac{H}{8} \times \frac{W}{8} \times 2C$",
        ha="right",
        va="center",
        fontsize=9,
        color="#333",
    )

    b_de1, b_cat1, b_lin1, b_dv1 = decoder_stage_boxes(y_d1_c, r"$\times 2$")
    ax.text(
        12.65,
        y_d1_c,
        r"$\frac{H}{4} \times \frac{W}{4} \times C$",
        ha="right",
        va="center",
        fontsize=9,
        color="#333",
    )

    cx_d = x_dec + w / 2
    y_f = b_dv1[1] + b_dv1[3] + gap
    b_def = _white_block(ax, x_dec, y_f, w, bh, "Patch Expanding")
    y_f += bh + gap
    b_lp = _white_block(ax, x_dec, y_f, w, bh, "Linear Projection")
    y_f += bh + gap
    b_seg = _white_block(ax, x_dec, y_f, w, bh * 0.9, r"Seg  $H \times W \times Class$", fontsize=8)

    # BN → Patch Expanding (đáy stage decoder 3)
    bn_cx = x_mid_bn + w_bn / 2
    bn_top = y_bn + h_bn
    _arrow(ax, bn_cx, bn_top + 0.02, cx_d, b_de3[1] - 0.02)

    def chain_vertical(bottom_box, *upper_boxes):
        """Mũi tên từ đỉnh khối dưới tới đáy khối trên (theo thứ tự đáy → đỉnh)."""
        prev = bottom_box
        for nxt in upper_boxes:
            _arrow(
                ax,
                cx_d,
                prev[1] + prev[3] + 0.02,
                cx_d,
                nxt[1] - 0.02,
            )
            prev = nxt

    chain_vertical(b_de3, b_cat3, b_lin3, b_dv3)
    _arrow(ax, cx_d, b_dv3[1] + b_dv3[3] + 0.02, cx_d, b_de2[1] - 0.02)
    chain_vertical(b_de2, b_cat2, b_lin2, b_dv2)
    _arrow(ax, cx_d, b_dv2[1] + b_dv2[3] + 0.02, cx_d, b_de1[1] - 0.02)
    chain_vertical(b_de1, b_cat1, b_lin1, b_dv1)
    _arrow(ax, cx_d, b_dv1[1] + b_dv1[3] + 0.02, cx_d, b_def[1] - 0.02)
    _arrow(ax, cx_d, b_def[1] + b_def[3] + 0.02, cx_d, b_lp[1] - 0.02)
    _arrow(ax, cx_d, b_lp[1] + b_lp[3] + 0.02, cx_d, b_seg[1] - 0.02)

    x_skip_l = x_enc + w + 0.05
    x_skip_r = x_dec - 0.05
    _skip_arrow(ax, x_skip_l, cy(b_vss3)[1], x_skip_r, cy(b_cat3)[1])
    _skip_arrow(ax, x_skip_l, cy(b_vss2)[1], x_skip_r, cy(b_cat2)[1])
    _skip_arrow(ax, x_skip_l, cy(b_vss1)[1], x_skip_r, cy(b_cat1)[1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Đã lưu: {out_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Vẽ sơ đồ Mamba-UNet (kiểu paper)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures") / "mamba_unet_architecture.png",
    )
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    draw_diagram(args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
