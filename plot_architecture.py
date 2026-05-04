"""
plot_architecture.py

Generates a matplotlib diagram of the Attention Residual U-Net (AttnResUNet)
used in pytorch_train_resunet_gamma_mixture.py.

Output: architecture_diagram.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ─── Layout ───────────────────────────────────────────────────────────────────
FW, FH   = 26, 17
BG       = '#ECEFF1'

X_ENC    = 3.5          # encoder column centre-x
X_BRIDGE = 11.5         # bridge centre-x
X_DEC    = 19.5         # decoder column centre-x
BW       = 5.0          # encoder / decoder block width
BW_B     = 5.0          # bridge block width
BH       = 1.5          # encoder / decoder block height
BH_IO    = 1.8          # input / output block height

Y_TOP    = 15.2         # centre-y of input and output blocks
Y_LEV    = [13.0, 11.0, 9.0, 7.0]   # U-Net resolution levels 0–3
Y_BRIDGE = 4.5

# ─── Colours ──────────────────────────────────────────────────────────────────
C_IN   = '#00695C'
C_ENC  = '#1565C0'
C_BRDG = '#4A148C'
C_DEC  = '#1B5E20'
C_ATTN = '#E65100'
C_OUT  = '#B71C1C'
C_ARR  = '#37474F'
C_SKIP = '#78909C'

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def box(cx, cy, label, sub='', color=C_ENC, w=BW, h=BH, fs=19, sfs=17,
        lbl_dy=None, sub_dy=None):
    """Rounded rectangle; sub may contain \\n for line-wrapping."""
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.07',
                        fc=color, ec='white', lw=2.0, zorder=3)
    ax.add_patch(r)
    _lbl_dy = lbl_dy if lbl_dy is not None else (0.27 if sub else 0)
    _sub_dy = sub_dy if sub_dy is not None else -0.28
    ax.text(cx, cy + _lbl_dy, label, ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold', zorder=4)
    if sub:
        ax.text(cx, cy + _sub_dy, sub, ha='center', va='center',
                fontsize=sfs, color='#BDBDBD', zorder=4,
                multialignment='center')


def arr(x1, y1, x2, y2, color=C_ARR, lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=24,
                                connectionstyle='arc3,rad=0.0'), zorder=5)


def skip(y_lev):
    """Dashed skip connection with orange attention-gate circle at midpoint."""
    xe = X_ENC + BW / 2
    xd = X_DEC - BW / 2
    xm = (xe + xd) / 2
    ax.plot([xe, xm - 0.48], [y_lev, y_lev],
            color=C_SKIP, lw=1.8, ls='--', zorder=2)
    circ = Circle((xm, y_lev), 0.42, fc=C_ATTN, ec='white', lw=1.8, zorder=6)
    ax.add_patch(circ)
    ax.text(xm, y_lev, 'A', ha='center', va='center',
            fontsize=16, color='white', fontweight='bold', zorder=7)
    arr(xm + 0.48, y_lev, xd, y_lev, color=C_SKIP, lw=1.8)


# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FW / 2, FH - 0.22,
        'Attention Residual U-Net  ·  2-Component Gamma Mixture Precipitation Model',
        ha='center', va='top', fontsize=27, color='#263238')

# ══════════════════════════════════════════════════════════════════════════════
# Input block
# ══════════════════════════════════════════════════════════════════════════════
box(X_ENC, Y_TOP, 'Input   7 × 96 × 96',
    '① GRAF   ② terrain   ③ GFS-RH\n④ GRAF×terrain   ⑤ GRAF×RH\n⑥ ∂/∂lon   ⑦ ∂/∂lat',
    color=C_IN, w=BW + 0.5, h=BH_IO, fs=20, sfs=16, lbl_dy=0.42)
arr(X_ENC, Y_TOP - BH_IO/2, X_ENC, Y_LEV[0] + BH/2)

# ══════════════════════════════════════════════════════════════════════════════
# Encoder column
# ══════════════════════════════════════════════════════════════════════════════
enc = [
    ('inc',   '96 × 96 · 64 ch',  'ResBlock  7 → 64'),
    ('down1', '48 × 48 · 128 ch', 'MaxPool  →  ResBlock\n64 → 128'),
    ('down2', '24 × 24 · 256 ch', 'MaxPool  →  ResBlock\n128 → 256'),
    ('down3', '12 × 12 · 512 ch', 'MaxPool  →  ResBlock\n256 → 512'),
]
for i, (name, dims, detail) in enumerate(enc):
    box(X_ENC, Y_LEV[i], f'{name}:  {dims}', detail, color=C_ENC)
    if i < 3:
        arr(X_ENC, Y_LEV[i] - BH/2, X_ENC, Y_LEV[i+1] + BH/2)

# Enc[3] bottom → Bridge left (diagonal)
arr(X_ENC, Y_LEV[3] - BH/2, X_BRIDGE - BW_B/2, Y_BRIDGE)

# ══════════════════════════════════════════════════════════════════════════════
# Bridge
# ══════════════════════════════════════════════════════════════════════════════
box(X_BRIDGE, Y_BRIDGE,
    'Bridge:  6 × 6 · 1024 ch',
    'MaxPool  →  ResBlock  512 → 1024',
    color=C_BRDG, w=BW_B, h=BH)

# Bridge right → Dec[0] bottom (diagonal)
arr(X_BRIDGE + BW_B/2, Y_BRIDGE, X_DEC, Y_LEV[3] - BH/2)

# ══════════════════════════════════════════════════════════════════════════════
# Decoder column  (up1 at level 3, up4 at level 0)
# ══════════════════════════════════════════════════════════════════════════════
dec = [
    ('up1', '12 × 12 · 512 ch', 'ConvTranspose  →  cat\n→  ResBlock  1024 → 512'),
    ('up2', '24 × 24 · 256 ch', 'ConvTranspose  →  cat\n→  ResBlock  512 → 256'),
    ('up3', '48 × 48 · 128 ch', 'ConvTranspose  →  cat\n→  ResBlock  256 → 128'),
    ('up4', '96 × 96 ·  64 ch', 'ConvTranspose  →  cat\n→  ResBlock  128 → 64'),
]
for i, (name, dims, detail) in enumerate(dec):
    y = Y_LEV[3 - i]
    box(X_DEC, y, f'{name}:  {dims}', detail, color=C_DEC)
    if i < 3:
        arr(X_DEC, Y_LEV[3-i] + BH/2, X_DEC, Y_LEV[3-i-1] - BH/2)

# Dec[0] top → Output block
arr(X_DEC, Y_LEV[0] + BH/2, X_DEC, Y_TOP - BH_IO/2)

# ══════════════════════════════════════════════════════════════════════════════
# Skip connections with Attention Gates
# ══════════════════════════════════════════════════════════════════════════════
for y in Y_LEV:
    skip(y)

# ══════════════════════════════════════════════════════════════════════════════
# Output block
# ══════════════════════════════════════════════════════════════════════════════
box(X_DEC, Y_TOP,
    'outc:  96 × 96 · 6 ch  (raw logits)',
    'Conv 1×1   64 → 6',
    color=C_OUT, w=BW + 0.5, h=BH_IO, fs=20, sfs=16)

# ══════════════════════════════════════════════════════════════════════════════
# Per-pixel output parameters  (right margin, same y range as output block)
# ══════════════════════════════════════════════════════════════════════════════
px      = X_DEC + (BW + 0.5)/2 + 0.38    # left edge of param boxes
pw      = FW - 0.20 - px                  # width of param boxes
ph      = 0.92                             # height of each param box
pg      = 1.12                             # centre-to-centre gap
py_top  = Y_TOP                            # first param box aligned with output block

ax.text(px + pw/2, py_top + ph/2 + 0.40,
        'Per-pixel outputs\n(6 per pixel)',
        ha='center', va='bottom', fontsize=18, fontweight='bold', color='#263238')

params = [
    ('①  p₀',    'fraction_zero',   'sigmoid( logit[0] )'),
    ('②  w',      'mixing weight',   'sigmoid( logit[1] )'),
    ('③  α₁',     'shape₁  (light)', 'shape_min + softplus( logit[2] )'),
    ('④  θ₁',     'scale₁  (light)', 'scale_min + softplus( logit[3] )'),
    ('⑤  α₂',     'shape₂  (heavy)', 'α₁ + softplus( logit[4] ) + 0.5\n[hard ordering constraint]'),
    ('⑥  θ₂',     'scale₂  (heavy)', 'scale_min + softplus( logit[5] )'),
]
for j, (sym, name, act) in enumerate(params):
    yj = py_top - j * pg
    r = FancyBboxPatch((px, yj - ph/2), pw, ph,
                        boxstyle='round,pad=0.05',
                        fc=C_OUT, ec='white', lw=1.4, alpha=0.90, zorder=3)
    ax.add_patch(r)
    ax.text(px + 0.20, yj + 0.18, f'{sym}:  {name}',
            ha='left', va='center', fontsize=16, color='white',
            fontweight='bold', zorder=4)
    ax.text(px + 0.20, yj - 0.18, act,
            ha='left', va='center', fontsize=13, color='#FFCDD2', zorder=4,
            multialignment='left')

# Horizontal arrow from output block right edge to param list
arr(X_DEC + (BW+0.5)/2, Y_TOP, px - 0.05, Y_TOP, color=C_OUT, lw=2.0)

# ══════════════════════════════════════════════════════════════════════════════
# ResidualBlock inset  (bottom-left, clear of bridge)
# ══════════════════════════════════════════════════════════════════════════════
ri_x, ri_y = 0.5, 4.7
ri_w, ri_h, ri_gap = 4.6, 0.72, 1.12

ax.text(ri_x + ri_w/2, ri_y + 0.65, 'ResidualBlock',
        ha='center', fontsize=17, fontweight='bold', color='#263238')
layers = ['Conv3×3 – BN – ReLU', 'Conv3×3 – BN']
for k, lbl in enumerate(layers):
    yk = ri_y - k * ri_gap
    r = FancyBboxPatch((ri_x, yk - ri_h/2), ri_w, ri_h,
                        boxstyle='round,pad=0.05',
                        fc=C_ENC, ec='white', lw=1.5, zorder=3)
    ax.add_patch(r)
    ax.text(ri_x + ri_w/2, yk, lbl, ha='center', va='center',
            fontsize=18, color='white', zorder=4)
    if k == 0:
        arr(ri_x + ri_w/2, yk - ri_h/2,
            ri_x + ri_w/2, yk - ri_gap + ri_h/2, lw=1.4)

yr_add = ri_y - 2 * ri_gap
r = FancyBboxPatch((ri_x, yr_add - ri_h/2), ri_w, ri_h,
                    boxstyle='round,pad=0.05',
                    fc='#455A64', ec='white', lw=1.5, zorder=3)
ax.add_patch(r)
ax.text(ri_x + ri_w/2, yr_add, 'Add + ReLU', ha='center', va='center',
        fontsize=18, color='white', zorder=4)
arr(ri_x + ri_w/2, ri_y - ri_gap - ri_h/2,
    ri_x + ri_w/2, yr_add + ri_h/2, lw=1.4)

# Shortcut (residual) arrow on right side
ax.annotate('', xy=(ri_x + ri_w + 0.15, yr_add + ri_h/2 + 0.05),
            xytext=(ri_x + ri_w + 0.15, ri_y + ri_h/2),
            arrowprops=dict(arrowstyle='->', color=C_ENC, lw=2.0,
                            mutation_scale=24,
                            connectionstyle='arc3,rad=-0.35'), zorder=5)
ax.text(ri_x + ri_w + 0.82, (ri_y + yr_add)/2, 'shortcut',
        va='center', ha='center', fontsize=14, color=C_ENC, rotation=270)

# ══════════════════════════════════════════════════════════════════════════════
# AttentionGate inset  (bottom-right of bridge, clear of diagonal arrows)
# ══════════════════════════════════════════════════════════════════════════════
ai_x, ai_y = 16.5, 4.7
ai_w, ai_h, ai_gap = 6.5, 0.72, 1.12

ax.text(ai_x + ai_w/2, ai_y + 0.65, 'AttentionGate',
        ha='center', fontsize=17, fontweight='bold', color='#263238')
att_steps = [
    'g′ = BN( Conv1×1( g ) )',
    'x′ = BN( Conv1×1( x ) )',
    'ψ  = Sigmoid( BN( Conv1×1( ReLU( g′ + x′ ) ) ) )',
    'output = x  ×  ψ',
]
for k, lbl in enumerate(att_steps):
    yk = ai_y - k * ai_gap
    r = FancyBboxPatch((ai_x, yk - ai_h/2), ai_w, ai_h,
                        boxstyle='round,pad=0.05',
                        fc=C_ATTN, ec='white', lw=1.5, alpha=0.92, zorder=3)
    ax.add_patch(r)
    ax.text(ai_x + 0.20, yk, lbl, ha='left', va='center',
            fontsize=17, color='white', zorder=4)
    if k < len(att_steps) - 1:
        arr(ai_x + ai_w/2, yk - ai_h/2,
            ai_x + ai_w/2, yk - ai_gap + ai_h/2, lw=1.3)

# Input labels
ax.text(ai_x - 0.18, ai_y, 'g =\ndecoder\nupsampled',
        ha='right', va='center', fontsize=14, color='#546E7A', style='italic')
ax.text(ai_x - 0.18, ai_y - ai_gap, 'x =\nencoder\nskip',
        ha='right', va='center', fontsize=14, color='#546E7A', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════════
legend = [
    (C_ENC,  'Encoder block'),
    (C_BRDG, 'Bridge'),
    (C_DEC,  'Decoder block'),
    (C_ATTN, 'Attention Gate (A)'),
    (C_IN,   'Input'),
    (C_OUT,  'Output / params'),
]
xl, yl = 0.4, 0.50
for k, (col, lbl) in enumerate(legend):
    xk = xl + k * 4.25
    circ = Circle((xk + 0.22, yl), 0.20, fc=col, ec='white', lw=1.2, zorder=4)
    ax.add_patch(circ)
    ax.text(xk + 0.55, yl, lbl, va='center', fontsize=16, color='#263238')

ax.plot([0.4, 0.92], [0.15, 0.15], color=C_SKIP, lw=1.8, ls='--')
ax.text(1.05, 0.15,
        'dashed = skip connection  (attended by  A,  then concatenated with upsampled features)',
        va='center', fontsize=15, color='#263238')

plt.tight_layout(pad=0.1)
plt.savefig('architecture_diagram.png', dpi=200, bbox_inches='tight')
print('Saved architecture_diagram.png')
plt.close()
