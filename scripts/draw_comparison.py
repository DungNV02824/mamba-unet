"""
So sánh kiến trúc VMamba-UNet gốc vs mô hình tùy chỉnh (4.73M tham số)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#F8F9FA')

# ─────────────────────────────────────────────────────────────────────────────
# TIÊU ĐỀ CHÍNH
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'So sánh kiến trúc: VMamba-UNet gốc  vs  Mô hình tùy chỉnh',
         ha='center', va='top', fontsize=18, fontweight='bold', color='#1a1a2e')

# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1: So sánh tổng quan kiến trúc UNet  (ax_orig, ax_custom)
# ═════════════════════════════════════════════════════════════════════════════
ax_orig   = fig.add_axes([0.03, 0.52, 0.44, 0.42])
ax_custom = fig.add_axes([0.53, 0.52, 0.44, 0.42])

def draw_unet(ax, title, embed_dim, depths, color_enc, color_dec, color_btn,
              param_total, C_label):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    dims  = [embed_dim * (2**i) for i in range(4)]
    stage_labels = [f'{depths[i]}×VSS' for i in range(4)]

    # ── Tiêu đề ──────────────────────────────────────────────────────────────
    ax.text(5, 9.7, title, ha='center', va='top', fontsize=13,
            fontweight='bold', color='#1a1a2e')
    ax.text(5, 9.3, f'Tổng tham số: {param_total}',
            ha='center', va='top', fontsize=11,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#e63946', ec='none'))

    # ── Encoder blocks ────────────────────────────────────────────────────────
    enc_y   = [8.3, 7.0, 5.7, 4.4]
    enc_x   = 2.5
    enc_w   = 3.0
    enc_h   = 0.85

    for i in range(4):
        fc = color_enc if i < 3 else color_btn
        box = FancyBboxPatch((enc_x - enc_w/2, enc_y[i] - enc_h/2),
                             enc_w, enc_h,
                             boxstyle='round,pad=0.08',
                             facecolor=fc, edgecolor='#333', linewidth=1.2)
        ax.add_patch(box)
        label_stage = 'Bottleneck' if i == 3 else f'Encoder Stage {i+1}'
        ax.text(enc_x, enc_y[i] + 0.18, label_stage,
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        ax.text(enc_x, enc_y[i] - 0.18, f'{stage_labels[i]}  |  C={dims[i]}',
                ha='center', va='center', fontsize=8, color='white')

    # ── Mũi tên encoder ───────────────────────────────────────────────────────
    for i in range(3):
        ax.annotate('', xy=(enc_x, enc_y[i+1] + enc_h/2),
                    xytext=(enc_x, enc_y[i] - enc_h/2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
        ax.text(enc_x + 1.7, (enc_y[i] + enc_y[i+1])/2,
                f'PatchMerge\n×2↓  C:{dims[i]}→{dims[i+1]}',
                ha='left', va='center', fontsize=7, color='#555',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f0f0f0', ec='#aaa', lw=0.8))

    # ── Decoder blocks ────────────────────────────────────────────────────────
    dec_y = [5.7, 7.0, 8.3]
    dec_x = 7.5

    for i in range(3):
        stage_idx = 2 - i
        box = FancyBboxPatch((dec_x - enc_w/2, dec_y[i] - enc_h/2),
                             enc_w, enc_h,
                             boxstyle='round,pad=0.08',
                             facecolor=color_dec, edgecolor='#333', linewidth=1.2)
        ax.add_patch(box)
        ax.text(dec_x, dec_y[i] + 0.18, f'Decoder Stage {stage_idx+1}',
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        ax.text(dec_x, dec_y[i] - 0.18,
                f'{stage_labels[stage_idx]}  |  C={dims[stage_idx]}',
                ha='center', va='center', fontsize=8, color='white')

    # ── Mũi tên decoder ───────────────────────────────────────────────────────
    for i in range(2):
        ax.annotate('', xy=(dec_x, dec_y[i+1] - enc_h/2),
                    xytext=(dec_x, dec_y[i] + enc_h/2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
        ax.text(dec_x - 1.8, (dec_y[i] + dec_y[i+1])/2,
                f'PatchExpand\n×2↑',
                ha='right', va='center', fontsize=7, color='#555',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f0f0f0', ec='#aaa', lw=0.8))

    # ── Skip connections ──────────────────────────────────────────────────────
    skip_pairs = [(0, 2), (1, 1), (2, 0)]
    skip_colors = ['#2ecc71', '#f39c12', '#9b59b6']
    for (ei, di), sc in zip(skip_pairs, skip_colors):
        y_e = enc_y[ei]
        y_d = dec_y[di]
        y_mid = (y_e + y_d) / 2
        # nếu cùng y thì vẽ thẳng, khác y thì vẽ cung
        ax.annotate('', xy=(dec_x - enc_w/2, y_d),
                    xytext=(enc_x + enc_w/2, y_e),
                    arrowprops=dict(arrowstyle='->', color=sc, lw=1.5,
                                   connectionstyle='arc3,rad=-0.2'))
        ax.text(5.0, (y_e+y_d)/2 + 0.05, 'Skip',
                ha='center', va='center', fontsize=6.5, color=sc, fontstyle='italic')

    # ── Input / Output ────────────────────────────────────────────────────────
    ax.text(enc_x, 9.2, '↑  Input  512×512×1', ha='center', va='bottom',
            fontsize=7.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', fc='#dfe6e9', ec='#aaa'))
    ax.annotate('', xy=(enc_x, enc_y[0] + enc_h/2 + 0.05),
                xytext=(enc_x, 9.05),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))

    ax.text(dec_x, 9.2, 'Output  512×512×Class  ↑', ha='center', va='bottom',
            fontsize=7.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', fc='#dfe6e9', ec='#aaa'))
    ax.annotate('', xy=(dec_x, 9.05),
                xytext=(dec_x, dec_y[2] + enc_h/2 + 0.05),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))

    # Bottleneck ↔ Decoder
    ax.annotate('', xy=(dec_x - enc_w/2, dec_y[0]),
                xytext=(enc_x + enc_w/2, enc_y[3]),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.8,
                                connectionstyle='arc3,rad=0.3'))

    # ── C_label ──────────────────────────────────────────────────────────────
    ax.text(0.2, 0.3, C_label, ha='left', va='bottom', fontsize=8,
            color='#555', style='italic')


# Vẽ 2 cột
draw_unet(ax_orig,
          title='VMamba-UNet  GỐC',
          embed_dim=96,
          depths=[2, 2, 9, 2],
          color_enc='#2980b9',
          color_dec='#27ae60',
          color_btn='#8e44ad',
          param_total='~27M',
          C_label='C=96  →  2C=192  →  4C=384  →  8C=768')

draw_unet(ax_custom,
          title='Mô hình tùy chỉnh  (4.73M)',
          embed_dim=32,
          depths=[2, 2, 2, 1],
          color_enc='#e67e22',
          color_dec='#16a085',
          color_btn='#c0392b',
          param_total='4.73M  (↓83%)',
          C_label='C=32  →  2C=64  →  4C=128  →  8C=256')

# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2: So sánh VSSBlock  (bên trái gốc, bên phải custom)
# ═════════════════════════════════════════════════════════════════════════════
ax_vss_orig   = fig.add_axes([0.03, 0.14, 0.27, 0.35])
ax_vss_custom = fig.add_axes([0.36, 0.14, 0.27, 0.35])
ax_ss2d       = fig.add_axes([0.69, 0.14, 0.28, 0.35])

# ── Hàm vẽ VSS block dạng flowchart ─────────────────────────────────────────
def draw_vssblock(ax, title, layers, colors, subtitle=''):
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.3, len(layers) + 1.2)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    ax.text(2, len(layers) + 0.9, title, ha='center', va='top',
            fontsize=10, fontweight='bold', color='#1a1a2e')
    if subtitle:
        ax.text(2, len(layers) + 0.45, subtitle, ha='center', va='top',
                fontsize=8, color='#e63946', fontstyle='italic')

    bw, bh = 2.6, 0.5
    bx = 2 - bw/2

    for i, (lbl, fc) in enumerate(zip(layers, colors)):
        y = len(layers) - 1 - i
        box = FancyBboxPatch((bx, y - bh/2 + 0.05), bw, bh,
                             boxstyle='round,pad=0.06',
                             facecolor=fc, edgecolor='#333', linewidth=1)
        ax.add_patch(box)
        ax.text(2, y + 0.05, lbl, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')

        if i < len(layers) - 1:
            ax.annotate('', xy=(2, y - bh/2 + 0.08),
                        xytext=(2, y + bh/2 - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))

    # Skip annotations
    skip_y_top = len(layers) - 0.5
    skip_y_bot = -0.05
    ax.annotate('', xy=(0.15, skip_y_bot),
                xytext=(0.15, skip_y_top),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5,
                                connectionstyle='arc3,rad=0.0'))
    ax.text(0.0, (skip_y_top + skip_y_bot)/2, 'Skip\n×3',
            ha='center', va='center', fontsize=7, color='#e74c3c', fontweight='bold')

# VSSBlock gốc — dùng 4 hướng SS2D
layers_orig = [
    'LayerNorm',
    'SS2D  (4 hướng quét)\n× 4 Mamba modules',
    '+ Residual  (Skip 1)',
    'DWConv 3×3  |  BN  |  GELU',
    '+ Residual  (Skip 2)',
    'LayerNorm',
    'MLP  (4× expand)',
    '+ Residual  (Skip 3)',
]
colors_orig = ['#2980b9','#1a6b9a','#27ae60',
               '#8e44ad','#27ae60',
               '#2980b9','#c0392b','#27ae60']

draw_vssblock(ax_vss_orig, 'VSSBlock  GỐC', layers_orig, colors_orig,
              subtitle='SS2D: 4 Mamba modules  →  nhiều tham số hơn')

# VSSBlock tùy chỉnh — 1 hướng SS2D
layers_custom = [
    'LayerNorm',
    'SS2D  (1 hướng quét)\n× 1 Mamba module',
    '+ Residual  (Skip 1)',
    'DWConv 3×3  |  BN  |  GELU',
    '+ Residual  (Skip 2)',
    'LayerNorm',
    'MLP  (4× expand)',
    '+ Residual  (Skip 3)',
]
colors_custom = ['#e67e22','#c0392b','#27ae60',
                 '#8e44ad','#27ae60',
                 '#e67e22','#c0392b','#27ae60']

draw_vssblock(ax_vss_custom, 'VSSBlock  TÙY CHỈNH', layers_custom, colors_custom,
              subtitle='SS2D: 1 Mamba module  →  ít tham số hơn ~4×')

# ── So sánh SS2D 4 hướng vs 1 hướng ──────────────────────────────────────────
ax_ss2d.set_xlim(0, 5)
ax_ss2d.set_ylim(0, 10)
ax_ss2d.axis('off')
ax_ss2d.set_facecolor('#F8F9FA')
ax_ss2d.text(2.5, 9.7, 'SS2D: 4 hướng vs 1 hướng',
             ha='center', va='top', fontsize=10, fontweight='bold', color='#1a1a2e')

# Grid ảnh minh họa
grid_size = 4
cell = 0.38
start_x_orig = 0.4
start_x_cust = 2.9

arrows_orig = [
    ([(0,3),(1,3),(2,3),(3,3)], '#e74c3c', '→'),
    ([(3,3),(2,3),(1,3),(0,3)], '#3498db', '←'),
    ([(0,3),(0,2),(0,1),(0,0)], '#2ecc71', '↓'),
    ([(0,0),(0,1),(0,2),(0,3)], '#f39c12', '↑'),
]
arrows_cust = [
    ([(0,3),(1,3),(2,3),(3,3),(0,2),(1,2),(2,2),(3,2)], '#e74c3c', 'row→'),
]

def draw_grid_arrows(ax, sx, sy, cell, grid_size, arrows, title, color):
    ax.text(sx + grid_size*cell/2, sy + grid_size*cell + 0.35, title,
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)
    # grid
    for r in range(grid_size):
        for c in range(grid_size):
            rect = plt.Rectangle((sx + c*cell, sy + r*cell), cell, cell,
                                  facecolor='#dfe6e9', edgecolor='#b2bec3', lw=0.7)
            ax.add_patch(rect)
    # arrows minh họa hướng quét
    direction_info = []
    dir_y = sy - 0.25
    for arr_cells, ac, symbol in arrows:
        direction_info.append((ac, symbol))
    for idx, (ac, symbol) in enumerate(direction_info):
        ax.text(sx + idx * (grid_size*cell / len(direction_info)) + 0.1,
                dir_y, symbol,
                ha='left', va='top', fontsize=9, color=ac, fontweight='bold')

draw_grid_arrows(ax_ss2d, start_x_orig, 5.5, cell, grid_size,
                 arrows_orig, 'Gốc: 4 hướng quét', '#2980b9')

draw_grid_arrows(ax_ss2d, start_x_cust, 5.5, cell, grid_size,
                 arrows_cust, 'Custom: 1 hướng', '#e67e22')

# Minh họa mũi tên quét thực sự
for i, (clr, sym, dx, dy, sx0, sy0) in enumerate([
    ('#e74c3c', '→', 1, 0, start_x_orig, 5.5+3*cell+cell/2),
    ('#3498db', '←', -1, 0, start_x_orig+3*cell, 5.5+2*cell+cell/2),
    ('#2ecc71', '↓', 0, -1, start_x_orig+cell/2, 5.5+3*cell),
    ('#f39c12', '↑', 0, 1, start_x_orig+2*cell+cell/2, 5.5),
]):
    ex = sx0 + dx * 3 * cell
    ey = sy0 + dy * 3 * cell
    ax_ss2d.annotate('', xy=(ex, ey), xytext=(sx0, sy0),
                     arrowprops=dict(arrowstyle='->', color=clr, lw=2.0))

# Custom: 1 hướng raster
ax_ss2d.annotate('', xy=(start_x_cust + 3*cell, 5.5 + 3*cell + cell/2),
                 xytext=(start_x_cust, 5.5 + 3*cell + cell/2),
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.0))

# Labels
ax_ss2d.text(start_x_orig + grid_size*cell/2, 5.1,
             '4 Mamba modules\n~4× tham số SS2D',
             ha='center', va='top', fontsize=7.5, color='#2980b9',
             bbox=dict(boxstyle='round,pad=0.3', fc='#d6eaf8', ec='#2980b9', lw=0.8))

ax_ss2d.text(start_x_cust + grid_size*cell/2, 5.1,
             '1 Mamba module\n~1× tham số SS2D',
             ha='center', va='top', fontsize=7.5, color='#e67e22',
             bbox=dict(boxstyle='round,pad=0.3', fc='#fdebd0', ec='#e67e22', lw=0.8))

# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3: Bảng tóm tắt + Bar chart tham số
# ═════════════════════════════════════════════════════════════════════════════
ax_bar  = fig.add_axes([0.03, 0.02, 0.44, 0.10])
ax_info = fig.add_axes([0.53, 0.02, 0.44, 0.10])

# Bar chart
categories = ['embed_dim\n(C)', 'Tổng depths\n(VSS blocks)', 'Hướng\nquét SS2D', 'Tổng\ntham số (M)']
orig_vals   = [96, 15, 4, 27]
cust_vals   = [32,  7, 1,  4.73]
x = np.arange(len(categories))
w = 0.35

b1 = ax_bar.bar(x - w/2, orig_vals,   w, label='Gốc VMamba', color='#2980b9', alpha=0.85)
b2 = ax_bar.bar(x + w/2, cust_vals,   w, label='Tùy chỉnh',   color='#e67e22', alpha=0.85)

for bar in b1:
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(bar.get_height()), ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(b2, cust_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=8, fontweight='bold', color='#c0392b')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(categories, fontsize=8)
ax_bar.set_ylabel('Giá trị', fontsize=8)
ax_bar.set_title('So sánh các siêu tham số chính', fontsize=9, fontweight='bold')
ax_bar.legend(fontsize=8)
ax_bar.set_facecolor('#F8F9FA')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# Bảng tóm tắt
ax_info.axis('off')
ax_info.set_facecolor('#F8F9FA')
table_data = [
    ['Thành phần',        'Gốc VMamba',   'Tùy chỉnh',    'Giảm'],
    ['embed_dim (C)',      '96',           '32',            '3× nhỏ hơn'],
    ['depths',            '[2,2,9,2]=15', '[2,2,2,1]=7',  '2.1× ít block'],
    ['SS2D hướng quét',   '4 hướng',      '1 hướng',       '4× ít Mamba'],
    ['Tổng tham số',      '~27M',         '4.73M',         '↓ 83%'],
]
col_colors = [['#2c3e50']*4,
              ['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
              ['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
              ['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
              ['#d6eaf8','#d6eaf8','#fdebd0','#fadbd8'],]

table = ax_info.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        colors_row = [['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
                      ['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
                      ['#d6eaf8','#d6eaf8','#fdebd0','#d5f5e3'],
                      ['#eaf2ff','#eaf2ff','#fef9e7','#fadbd8'],]
        cell.set_facecolor(colors_row[r-1][c])
    cell.set_edgecolor('#bdc3c7')

plt.savefig('figures/comparison_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
print("Đã lưu: figures/comparison_architecture.png")
plt.show()
