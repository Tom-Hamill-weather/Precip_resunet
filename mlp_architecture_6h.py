#!/usr/bin/env python3
"""
mlp_architecture_6h.py – architecture diagram of GammaMixtureMLP.

Architecture: 38 → 72 → 144 → 72 → 36 → 12 → 6
Run:  python mlp_architecture_6h.py
Out:  mlp_architecture_6h.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox


def rbox(ax, cx, cy, w, h, fc, ec='white', lw=1.8, zo=4):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.07',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo))


def arrow(ax, x0, y0, x1, y1, c='#2C3E50', lw=1.8, zo=2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0'),
                zorder=zo)


def t(ax, x, y, s, color='white', fs=9, ha='center', va='center',
      bold=False, zo=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal', zorder=zo)


def main():
    FW, FH = 22.0, 9.0
    fig = plt.figure(figsize=(FW, FH), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
    ax.set_facecolor('none')
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor('white')

    C_H  = '#2874A6'   # expansion  – blue
    C_HD = '#1A5276'   # contraction – dark blue
    C_O  = '#1A7A43'   # output      – green
    BW = 1.55          # box width
    CY = 4.8

    # Heights scaled so 144 → 3.6; visually distinguishable but not extreme
    H = {144: 3.60, 72: 2.20, 36: 1.40, 12: 0.95, 6: 0.80}

    layers = [
        ( 4.8,  '38 → 72',   72,  'BN + ReLU', C_H ),
        ( 7.5,  '72 → 144', 144,  'BN + ReLU', C_H ),
        (10.2,  '144 → 72',  72,  'BN + ReLU', C_HD),
        (12.7,  '72 → 36',   36,  'BN + ReLU', C_HD),
        (15.0,  '36 → 12',   12,  'BN + ReLU', C_HD),
        (17.1,  '12 → 6',     6,  '',          C_O ),
    ]
    xs = [cx for cx, *_ in layers]
    hs = [H[n] for _, _, n, *_ in layers]

    # draw layer boxes + labels
    for (cx, lbl, n, act, fc), h in zip(layers, hs):
        rbox(ax, cx, CY, BW, h, fc)
        dy = max(h * 0.13, 0.18)
        t(ax, cx, CY + dy, 'Linear', fs=17, bold=True)
        t(ax, cx, CY - dy, lbl,      fs=15)
        if act:
            t(ax, cx, CY - h/2 - 0.58, act, color='#555', fs=16.9)
        t(ax, cx, CY + h/2 + 0.38, str(n), color='#333', fs=16, bold=True)

    # arrows between layers
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + BW/2 + 0.07, CY, xs[i+1] - BW/2 - 0.07, CY)

    # ── input feature panel ─────────────────────────────────────────────────
    feat = [
        ('p₀',  'fraction zero',   '#717D7E'),
        ('w',   'mix. weight',     '#717D7E'),
        ('α₁',  'shape, comp. 1',  '#2874A6'),
        ('θ₁',  'scale, comp. 1',  '#2874A6'),
        ('α₂',  'shape, comp. 2',  '#1A5276'),
        ('θ₂',  'scale, comp. 2',  '#1A5276'),
        ('cos', 'day-of-yr',       '#8E44AD'),
        ('sin', 'day-of-yr',       '#8E44AD'),
    ]
    PW, PR = 2.10, 0.58
    px = 2.10
    pt = CY + len(feat) * PR / 2   # y-centre of top row

    t(ax, px, pt + 0.68, 'Input features',            color='#1A1A1A', fs=19.0, bold=True)
    t(ax, px, pt + 0.28, '(6 vars × 6 h + 2 = 38)',   color='#555',    fs=15.7)

    pl = px - PW/2   # panel left edge
    for i, (sym, desc, fc) in enumerate(feat):
        ry = pt - (i + 0.5) * PR
        rbox(ax, px, ry, PW, PR * 0.88, fc, lw=1.0)
        sym_fs = 15.0 if len(sym) > 2 else 19.0
        desc_x = pl + (0.85 if len(sym) > 2 else 0.60)
        t(ax, pl + 0.22, ry, sym,  ha='left', fs=sym_fs, bold=True)
        t(ax, desc_x,    ry, desc, ha='left', fs=13.4)

    arrow(ax, px + PW/2 + 0.08, CY, xs[0] - BW/2 - 0.08, CY)

    # ── output parameter panel ──────────────────────────────────────────────
    out_p = [
        ('p₀',  'P(y=0)  [sigmoid]',   '#717D7E'),
        ('w',   'mix. wt  [sigmoid]',  '#717D7E'),
        ('α₁',  'shape₁  [softplus]',  '#2874A6'),
        ('θ₁',  'scale₁  [softplus]',  '#2874A6'),
        ('α₂',  'shape₂  [softplus]',  '#1A5276'),
        ('θ₂',  'scale₂  [softplus]',  '#1A5276'),
    ]
    OW = 2.70
    ox = 20.00
    ot = CY + len(out_p) * PR / 2

    t(ax, ox, ot + 0.68, 'Output parameters', color='#1A1A1A', fs=19.0, bold=True)
    t(ax, ox, ot + 0.28, '6-h Gamma mixture', color='#555',    fs=15.7)

    ol = ox - OW/2   # output panel left edge
    for i, (sym, desc, fc) in enumerate(out_p):
        ry = ot - (i + 0.5) * PR
        rbox(ax, ox, ry, OW, PR * 0.88, fc, lw=1.0)
        t(ax, ol + 0.22, ry, sym,  ha='left', fs=19.0, bold=True)
        t(ax, ol + 0.70, ry, desc, ha='left', fs=13.4)

    arrow(ax, xs[-1] + BW/2 + 0.08, CY, ox - OW/2 - 0.08, CY)

    # bottom note
    t(ax, FW/2, 1.80,
      'Hidden layers: Linear → BatchNorm1d → ReLU',
      color='#444', fs=21)

    # bbox_inches='tight' does not help here: Axes.get_tightbbox() always
    # folds in the axes' own full [0,0,1,1] window extent, so the "tight"
    # crop is just the full FW x FH canvas.  Instead, compute the true
    # content bbox directly from the drawn artists (boxes, text, arrows)
    # and crop to that, with a small explicit pad.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in ax.get_children():
        if not artist.get_visible():
            continue
        try:
            bb = artist.get_window_extent(renderer)
        except (NotImplementedError, AttributeError):
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append(bb)
    tight_px = Bbox.union(boxes)
    tight_in = tight_px.transformed(fig.dpi_scale_trans.inverted())
    pad_in = 0.10
    tight_in = Bbox.from_extents(tight_in.x0 - pad_in, tight_in.y0 - pad_in,
                                  tight_in.x1 + pad_in, tight_in.y1 + pad_in)

    fig.savefig('mlp_architecture_6h.png', dpi=150,
                bbox_inches=tight_in, facecolor='white')
    print('Saved: mlp_architecture_6h.png')


if __name__ == '__main__':
    main()
