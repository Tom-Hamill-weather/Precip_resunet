#!/usr/bin/env python3
"""
mlp_architecture_6h.py – architecture diagram of GammaMixtureMLP.

Architecture: 36 → 72 → 144 → 72 → 36 → 12 → 6
Run:  python mlp_architecture_6h.py
Out:  mlp_architecture_6h.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


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
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('white')

    C_H  = '#2874A6'   # expansion  – blue
    C_HD = '#1A5276'   # contraction – dark blue
    C_O  = '#1A7A43'   # output      – green
    BW = 1.55          # box width
    CY = 4.8

    # Heights scaled so 144 → 3.6; visually distinguishable but not extreme
    H = {144: 3.60, 72: 2.20, 36: 1.40, 12: 0.95, 6: 0.80}

    layers = [
        ( 4.8,  '36 → 72',   72,  'BN + ReLU', C_H ),
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
            t(ax, cx, CY - h/2 - 0.58, act, color='#555', fs=13)
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
    ]
    PW, PR = 2.10, 0.58
    px = 2.10
    pt = CY + len(feat) * PR / 2   # y-centre of top row

    t(ax, px, pt + 0.68, 'Input features',      color='#1A1A1A', fs=17, bold=True)
    t(ax, px, pt + 0.28, '(6 vars × 6 h = 36)', color='#555',    fs=14)

    pl = px - PW/2   # panel left edge
    for i, (sym, desc, fc) in enumerate(feat):
        ry = pt - (i + 0.5) * PR
        rbox(ax, px, ry, PW, PR * 0.88, fc, lw=1.0)
        t(ax, pl + 0.22, ry, sym,  ha='left', fs=17, bold=True)
        t(ax, pl + 0.60, ry, desc, ha='left', fs=12)

    t(ax, px, pt - len(feat) * PR - 0.38,
      '← t−5 … t  (consecutive hours)', color='#555', fs=13)

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

    t(ax, ox, ot + 0.68, 'Output parameters', color='#1A1A1A', fs=17, bold=True)
    t(ax, ox, ot + 0.28, '6-h Gamma mixture', color='#555',    fs=14)

    ol = ox - OW/2   # output panel left edge
    for i, (sym, desc, fc) in enumerate(out_p):
        ry = ot - (i + 0.5) * PR
        rbox(ax, ox, ry, OW, PR * 0.88, fc, lw=1.0)
        t(ax, ol + 0.22, ry, sym,  ha='left', fs=17, bold=True)
        t(ax, ol + 0.70, ry, desc, ha='left', fs=12)

    arrow(ax, xs[-1] + BW/2 + 0.08, CY, ox - OW/2 - 0.08, CY)

    # annotation: max-width layer
    cx144, h144 = xs[1], hs[1]
    ax.annotate('max expansion\n(144 neurons)',
                xy=(cx144, CY + h144/2 + 0.06),
                xytext=(cx144 - 0.4, CY + h144/2 + 1.40),
                ha='center', va='bottom', fontsize=14, color='#2874A6',
                arrowprops=dict(arrowstyle='->', color='#2874A6', lw=1.2),
                zorder=5)

    # bottom note
    t(ax, FW/2, 1.80,
      'Hidden layers: Linear → BatchNorm1d → ReLU\n'
      'Post-forward reordering ensures α₁θ₁ ≤ α₂θ₂  '
      '(comp. 1 = lighter precipitation)',
      color='#444', fs=21)

    fig.savefig('mlp_architecture_6h.png', dpi=150,
                bbox_inches='tight', pad_inches=0.15, facecolor='white')
    print('Saved: mlp_architecture_6h.png')


if __name__ == '__main__':
    main()
