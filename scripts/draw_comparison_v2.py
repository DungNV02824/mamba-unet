"""
So sánh kiến trúc VMamba-UNet gốc vs mô hình tùy chỉnh (4.73M tham số)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.gridspec as gridspec

BG = '#f5f6fa'

# ─── Primitive helpers ───────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, fc, ec='#444', lw=1.0, r=0.06):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={r}', facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    ))

def varrow(ax, x, y_from, y_to, color='#666', lw=1.2):
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def harrow(ax, x_from, x_to, y, color='#666', lw=1.4):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def carrow(ax, x1, y1, x2, y2, color='#e74c3c', lw=1.5, rad=-0.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))

# ─── UNet diagram ────────────────────────────────────────────────────────────

def draw_unet(ax, title, embed_dim, depths, param_text, c_enc, c_dec, c_btn):
    dims = [embed_dim * (2**i) for i in range(4)]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG)

    EX, DX = 2.4, 7.6          # encoder / decoder x-center
    BW, BH = 3.2, 0.80         # block width / height
    SY = [8.5, 6.9, 5.3]       # stage y-centers (top = high-res)
    BY = 3.5                   # bottleneck y-center

    # Title
    ax.text(5, 9.75, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1a1a2e')
    ax.text(5, 9.18, f'Tổng tham số: {param_text}',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor=c_btn, edgecolor='none'))

    # Input
    varrow(ax, EX, SY[0]+BH/2+0.50, SY[0]+BH/2+0.03)
    ax.text(EX, SY[0]+BH/2+0.55, 'Input  512×512×1',
            ha='center', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.20', fc='#ecf0f1', ec='#95a5a6', lw=0.8))

    # Encoder stages
    for i in range(3):
        y = SY[i]
        rbox(ax, EX, y, BW, BH, c_enc)
        ax.text(EX, y+0.21, f'Encoder Stage {i+1}',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        ax.text(EX, y-0.21, f'{depths[i]}×VSS  |  C={dims[i]}',
                ha='center', va='center', fontsize=7.5, color='#f0e8ff')
        ny = SY[i+1] if i < 2 else BY
        varrow(ax, EX, y-BH/2-0.02, ny+BH/2+0.02)
        ax.text(EX+1.82, (y+ny)/2, f'PatchMerge↓\n{dims[i]}→{dims[i+1]}',
                ha='left', va='center', fontsize=6.5, color='#555',
                bbox=dict(boxstyle='round,pad=0.14', fc='#f0f0f0', ec='#ccc', lw=0.6))

    # Bottleneck
    rbox(ax, 5.0, BY, BW+0.5, BH, c_btn)
    ax.text(5.0, BY+0.21, 'Bottleneck',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax.text(5.0, BY-0.21, f'{depths[3]}×VSS  |  C={dims[3]}',
            ha='center', va='center', fontsize=7.5, color='#f0e8ff')

    # Bottleneck → Decoder stage 3
    carrow(ax, 5.0+(BW+0.5)/2+0.02, BY, DX-BW/2-0.02, SY[2], rad=-0.18)

    # Decoder stages
    for i in range(3):
        si = 2 - i
        y = SY[si]
        rbox(ax, DX, y, BW, BH, c_dec)
        ax.text(DX, y+0.21, f'Decoder Stage {si+1}',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        ax.text(DX, y-0.21, f'{depths[si]}×VSS  |  C={dims[si]}',
                ha='center', va='center', fontsize=7.5, color='#f0e8ff')
        if i < 2:
            py = SY[si-1]
            varrow(ax, DX, y+BH/2+0.02, py-BH/2-0.02)
            ax.text(DX-1.82, (y+py)/2, 'PatchExpand↑',
                    ha='right', va='center', fontsize=6.5, color='#555',
                    bbox=dict(boxstyle='round,pad=0.14', fc='#f0f0f0', ec='#ccc', lw=0.6))

    # Output
    varrow(ax, DX, SY[0]+BH/2+0.03, SY[0]+BH/2+0.50)
    ax.text(DX, SY[0]+BH/2+0.55, 'Output  512×512×Class',
            ha='center', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.20', fc='#ecf0f1', ec='#95a5a6', lw=0.8))

    # Skip connections
    for i, sc in enumerate(['#2ecc71', '#f39c12', '#9b59b6']):
        y = SY[i]
        harrow(ax, EX+BW/2+0.02, DX-BW/2-0.02, y, color=sc, lw=1.7)
        ax.text(5.0, y+0.12, 'Skip', ha='center', va='bottom',
                fontsize=6.5, color=sc, style='italic')

    # Dim label
    ax.text(5.0, 0.55,
            f'C={dims[0]}  →  2C={dims[1]}  →  4C={dims[2]}  →  8C={dims[3]}',
            ha='center', va='center', fontsize=8, color='#555', style='italic',
            bbox=dict(boxstyle='round,pad=0.18', fc='#f8f8f8', ec='#ddd', lw=0.7))

# ─── VSS Block flowchart ─────────────────────────────────────────────────────

def draw_vssblock(ax, title, ss2d_text, c_ss2d):
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.3, 10)
    ax.axis('off')
    ax.set_facecolor(BG)

    ax.text(2.5, 9.75, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1a1a2e')

    BX, BW = 2.5, 2.9
    GAP = 0.85
    start_y = 8.85

    blocks = [
        ('Input  (B, C, H, W)',             '#7f8c8d', 0.48),
        ('LayerNorm',                        '#2980b9', 0.46),
        (ss2d_text,                          c_ss2d,   0.62),
        ('⊕  Residual  (Skip 1)',            '#27ae60', 0.42),
        ('DWConv 3×3  |  BN  |  GELU',       '#8e44ad', 0.48),
        ('⊕  Residual  (Skip 2)',            '#27ae60', 0.42),
        ('LayerNorm',                        '#2980b9', 0.46),
        ('MLP  (Linear×4 → GELU → Linear)',  '#c0392b', 0.48),
        ('⊕  Residual  (Skip 3)',            '#27ae60', 0.42),
        ('Output  (B, C, H, W)',             '#7f8c8d', 0.48),
    ]

    ys = [start_y - i * GAP for i in range(len(blocks))]

    for i, (txt, fc, bh) in enumerate(blocks):
        y = ys[i]
        rbox(ax, BX, y, BW, bh, fc)
        ax.text(BX, y, txt, ha='center', va='center',
                fontsize=7.2, color='white', fontweight='bold')
        if i < len(blocks) - 1:
            bh_next = blocks[i+1][2]
            ax.annotate('', xy=(BX, ys[i+1]+bh_next/2+0.02),
                        xytext=(BX, y-bh/2-0.02),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=0.9))

    # Skip L-lines on the left
    SX = 0.25
    for src, dst in [(0, 3), (3, 5), (5, 8)]:
        y_s, y_d = ys[src], ys[dst]
        lx = BX - BW/2
        ax.plot([lx, SX, SX, lx+0.10],
                [y_s, y_s, y_d, y_d],
                color='#e74c3c', lw=1.1, ls='--', alpha=0.85, zorder=2)
        ax.annotate('', xy=(lx, y_d), xytext=(lx+0.10, y_d),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.1))

    ax.text(0.13, (ys[0]+ys[8])/2, 'Residual\n×3',
            ha='center', va='center', fontsize=6.5, color='#e74c3c',
            fontweight='bold', rotation=90)

# ─── Comparison table panel ──────────────────────────────────────────────────

def draw_table(ax):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG)

    ax.text(2.5, 9.6, 'Bảng so sánh', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1a1a2e')

    headers = ['Thành phần', 'Gốc', 'Tùy chỉnh']
    rows = [
        ['embed_dim',      '96',          '32'],
        ['depths',         '[2,2,9,2]',   '[2,2,2,1]'],
        ['VSS blocks',     '15',          '7'],
        ['SS2D hướng',     '4',           '1'],
        ['img_size',       '512',         '512'],
        ['Tham số',        '~27M',        '4.73M'],
    ]
    row_fc = ['#eaf4fb', '#f9ebea', '#eafaf1', '#fdf2f8', '#fdfefe', '#fadbd8']
    col_xs = [0.18, 2.72, 4.05]
    HEADER_Y = 8.9
    ROW_H = 0.75

    # Header
    ax.add_patch(FancyBboxPatch((0.05, HEADER_Y-0.24), 4.9, 0.60,
                                boxstyle='round,pad=0.04', fc='#2c3e50', ec='none'))
    for j, h in enumerate(headers):
        ax.text(col_xs[j], HEADER_Y+0.06, h, ha='left', va='center',
                fontsize=8, fontweight='bold', color='white')

    for i, row in enumerate(rows):
        y = HEADER_Y - (i+1)*ROW_H
        ax.add_patch(FancyBboxPatch((0.05, y-0.27), 4.9, 0.60,
                                    boxstyle='round,pad=0.04',
                                    fc=row_fc[i], ec='#dee2e6', lw=0.6))
        is_last = (i == len(rows)-1)
        for j, val in enumerate(row):
            color = '#e74c3c' if (is_last and j > 0) else '#2c3e50'
            fw = 'bold' if is_last else 'normal'
            ax.text(col_xs[j], y+0.03, val, ha='left', va='center',
                    fontsize=7.8, color=color, fontweight=fw)

    # Impact summary
    ax.add_patch(FancyBboxPatch((0.05, 0.15), 4.9, 2.55,
                                boxstyle='round,pad=0.08',
                                fc='white', ec='#2c3e50', lw=1.0))
    ax.text(2.5, 2.58, 'Lý do giảm tham số', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2c3e50')

    impacts = [
        ('embed_dim  96→32',      '÷ 9× params',   '#2980b9'),
        ('depths  15→7 blocks',   '÷ 2.1× blocks', '#8e44ad'),
        ('SS2D  4→1 hướng quét',  '÷ 4× Mamba',    '#e67e22'),
        ('Tổng cộng',             '27M → 4.73M',   '#e74c3c'),
    ]
    for j, (name, eff, tc) in enumerate(impacts):
        y = 2.18 - j*0.44
        ax.text(0.22, y, name, ha='left', va='center', fontsize=7.5, color='#555')
        ax.text(4.78, y, eff, ha='right', va='center',
                fontsize=7.5, color=tc, fontweight='bold')
        if j < len(impacts)-1:
            ax.plot([0.18, 4.82], [y-0.20, y-0.20], color='#ecf0f1', lw=0.7)
    ax.plot([0.18, 4.82], [2.18-2*0.44-0.20, 2.18-2*0.44-0.20], color='#bdc3c7', lw=0.9)

# ─── SS2D comparison panel ───────────────────────────────────────────────────

def draw_ss2d(ax):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG)

    ax.text(2.5, 9.75, 'SS2D: So sánh hướng quét', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1a1a2e')

    CS = 0.38   # cell size
    N  = 4      # grid n×n
    GW = N * CS

    def draw_grid(ox, oy, title, tc):
        for r in range(N):
            for c in range(N):
                ax.add_patch(Rectangle((ox+c*CS, oy+r*CS), CS, CS,
                                       fc='#dfe6e9', ec='#b2bec3', lw=0.5, zorder=2))
        ax.text(ox+GW/2, oy+N*CS+0.14, title,
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=tc)

    # ── Original: 4 scan directions ──────────────────────────────────────────
    OX, OY = 0.35, 5.6
    draw_grid(OX, OY, 'Gốc: 4 hướng quét', '#2980b9')
    GH = N * CS
    scan = [
        (OX-0.08,    OY+GH-CS/2, OX+GW+0.08, OY+GH-CS/2, '#e74c3c'),  # →
        (OX+GW+0.08, OY+CS/2,    OX-0.08,    OY+CS/2,    '#3498db'),  # ←
        (OX+CS/2,    OY+GH+0.08, OX+CS/2,    OY-0.08,    '#2ecc71'),  # ↓
        (OX+GW-CS/2, OY-0.08,    OX+GW-CS/2, OY+GH+0.08, '#f39c12'),  # ↑
    ]
    for fx, fy, tx, ty, c in scan:
        ax.annotate('', xy=(tx, ty), xytext=(fx, fy),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8, mutation_scale=10))
    legends = [('→ Trái→Phải', '#e74c3c'), ('← Phải→Trái', '#3498db'),
               ('↓ Trên→Dưới', '#2ecc71'), ('↑ Dưới→Trên', '#f39c12')]
    for j, (lbl, lc) in enumerate(legends):
        ax.text(OX+GW+0.20, OY+GH-j*0.38, lbl,
                ha='left', va='center', fontsize=6.5, color=lc, fontweight='bold')
    ax.text(OX+GW/2, OY-0.15, '4 Mamba modules',
            ha='center', va='top', fontsize=7.5, color='#2980b9', fontweight='bold')

    # ── Custom: 1 scan direction ──────────────────────────────────────────────
    CX2, CY2 = 0.35, 2.4
    draw_grid(CX2, CY2, 'Custom: 1 hướng (raster)', '#e67e22')
    ax.annotate('', xy=(CX2+GW+0.08, CY2+GH-CS/2), xytext=(CX2-0.08, CY2+GH-CS/2),
                arrowprops=dict(arrowstyle='-|>', color='#e74c3c', lw=1.8, mutation_scale=10))
    for r in range(N):
        yr = CY2 + (N-1-r)*CS + CS/2
        ax.plot([CX2, CX2+GW], [yr, yr], color='#e74c3c', lw=0.9, ls=':')
    ax.text(CX2+GW/2, CY2-0.15, '1 Mamba module  (↓ ~4× tham số)',
            ha='center', va='top', fontsize=7.5, color='#e67e22', fontweight='bold')

    # ── Visual param comparison bar ───────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.08, 0.12), 4.84, 1.88,
                                boxstyle='round,pad=0.08', fc='white', ec='#bdc3c7', lw=0.9))
    ax.text(2.5, 1.88, 'Tổng tham số  (tỉ lệ tương đối)',
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2c3e50')
    # Bar: original
    bar_max_w = 4.3
    orig_w = bar_max_w
    cust_w = bar_max_w * (4.73 / 27)
    bx0 = 0.25
    ax.add_patch(Rectangle((bx0, 1.38), orig_w, 0.30, fc='#2980b9', ec='none', zorder=3))
    ax.text(bx0+orig_w+0.05, 1.53, '~27M', ha='left', va='center',
            fontsize=8, color='#2980b9', fontweight='bold')
    ax.add_patch(Rectangle((bx0, 0.92), cust_w, 0.30, fc='#e67e22', ec='none', zorder=3))
    ax.text(bx0+cust_w+0.05, 1.07, '4.73M', ha='left', va='center',
            fontsize=8, color='#e67e22', fontweight='bold')
    ax.text(bx0-0.02, 1.53, 'Gốc', ha='right', va='center', fontsize=7, color='#2c3e50')
    ax.text(bx0-0.02, 1.07, 'Custom', ha='right', va='center', fontsize=7, color='#2c3e50')

# ════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 22), facecolor=BG)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.05, 1.0],
    width_ratios=[1.0, 1.0, 0.86],
    hspace=0.06, wspace=0.05,
    left=0.02, right=0.98, top=0.96, bottom=0.02
)

fig.text(0.5, 0.978,
         'So sánh kiến trúc: VMamba-UNet gốc  vs  Mô hình tùy chỉnh (4.73M tham số)',
         ha='center', va='top', fontsize=15, fontweight='bold', color='#1a1a2e')

ax_orig  = fig.add_subplot(gs[0, 0])
ax_cust  = fig.add_subplot(gs[0, 1])
ax_tbl   = fig.add_subplot(gs[0, 2])
ax_vss_o = fig.add_subplot(gs[1, 0])
ax_vss_c = fig.add_subplot(gs[1, 1])
ax_ss2d  = fig.add_subplot(gs[1, 2])

draw_unet(ax_orig, 'VMamba-UNet  GỐC',
          embed_dim=96, depths=[2, 2, 9, 2], param_text='~27M',
          c_enc='#2980b9', c_dec='#27ae60', c_btn='#8e44ad')

draw_unet(ax_cust, 'Mô hình tùy chỉnh',
          embed_dim=32, depths=[2, 2, 2, 1], param_text='4.73M  (↓83%)',
          c_enc='#e67e22', c_dec='#16a085', c_btn='#c0392b')

draw_table(ax_tbl)

draw_vssblock(ax_vss_o, 'VSSBlock  GỐC',
              'SS2D  (4 hướng)\n× 4 Mamba modules', c_ss2d='#1a5276')

draw_vssblock(ax_vss_c, 'VSSBlock  TÙY CHỈNH',
              'SS2D  (1 hướng)\n× 1 Mamba module', c_ss2d='#a04000')

draw_ss2d(ax_ss2d)

out = 'figures/comparison_architecture.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
