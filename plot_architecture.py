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
FW, FH = 22, 14
BG     = '#ECEFF1'

X_ENC    = 3.5
X_BRIDGE = 11.0
X_DEC    = 18.5
BW       = 5.0    # encoder / decoder block width
BW_B     = 4.5    # bridge block width
BH       = 0.85   # block height

Y_TOP    = 13.3   # y for input / output blocks
Y_LEV    = [11.8, 10.0, 8.2, 6.4]  # levels 0–3 (shared by enc and dec)
Y_BRIDGE = 4.4

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

def box(cx, cy, label, sub='', color=C_ENC, w=BW, h=BH, fs=8.5, sfs=6.9):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.07',
                        fc=color, ec='white', lw=1.8, zorder=3)
    ax.add_patch(r)
    y0 = cy + (0.12 if sub else 0)
    ax.text(cx, y0, label, ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold', zorder=4)
    if sub:
        ax.text(cx, cy - 0.20, sub, ha='center', va='center',
                fontsize=sfs, color='#BDBDBD', zorder=4)


def arr(x1, y1, x2, y2, color=C_ARR, lw=1.6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=5)


def skip(y_lev):
    """Dashed skip connection with orange attention-gate circle at midpoint."""
    xe = X_ENC + BW / 2
    xd = X_DEC - BW / 2
    xm = (xe + xd) / 2
    ax.plot([xe, xm - 0.35], [y_lev, y_lev],
            color=C_SKIP, lw=1.5, ls='--', zorder=2)
    circ = Circle((xm, y_lev), 0.30, fc=C_ATTN, ec='white', lw=1.5, zorder=6)
    ax.add_patch(circ)
    ax.text(xm, y_lev, 'A', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=7)
    arr(xm + 0.35, y_lev, xd, y_lev, color=C_SKIP, lw=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(FW / 2, FH - 0.2,
        'Attention Residual U-Net  ·  2-Component Gamma Mixture Precipitation Model',
        ha='center', va='top', fontsize=13, fontweight='bold', color='#263238')

# ═══════════════════════════════════════════════════════════════════════════════
# Input
# ═══════════════════════════════════════════════════════════════════════════════
box(X_ENC, Y_TOP, 'Input   7 × 96 × 96',
    '① GRAF  ② terrain  ③ GFS-RH  ④ GRAF×terrain  ⑤ GRAF×RH  ⑥ ∂/∂lon  ⑦ ∂/∂lat',
    color=C_IN, w=BW + 0.4, fs=9, sfs=7.0)
arr(X_ENC, Y_TOP - BH / 2, X_ENC, Y_LEV[0] + BH / 2)

# ═══════════════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════════════
enc = [
    ('inc',   '96 × 96 · 64 ch',  'ResBlock  7 → 64'),
    ('down1', '48 × 48 · 128 ch', 'MaxPool  →  ResBlock  64 → 128'),
    ('down2', '24 × 24 · 256 ch', 'MaxPool  →  ResBlock  128 → 256'),
    ('down3', '12 × 12 · 512 ch', 'MaxPool  →  ResBlock  256 → 512'),
]
for i, (name, dims, detail) in enumerate(enc):
    box(X_ENC, Y_LEV[i], f'{name}:  {dims}', detail, color=C_ENC)
    if i < 3:
        arr(X_ENC, Y_LEV[i] - BH / 2, X_ENC, Y_LEV[i + 1] + BH / 2)

# Enc bottom → Bridge (diagonal)
arr(X_ENC, Y_LEV[3] - BH / 2, X_BRIDGE - BW_B / 2, Y_BRIDGE)

# ═══════════════════════════════════════════════════════════════════════════════
# Bridge
# ═══════════════════════════════════════════════════════════════════════════════
box(X_BRIDGE, Y_BRIDGE,
    'Bridge:  6 × 6 · 1024 ch',
    'MaxPool  →  ResBlock  512 → 1024',
    color=C_BRDG, w=BW_B)

# Bridge → Dec bottom (diagonal)
arr(X_BRIDGE + BW_B / 2, Y_BRIDGE, X_DEC, Y_LEV[3] - BH / 2)

# ═══════════════════════════════════════════════════════════════════════════════
# Decoder  (up1 at level 3, up4 at level 0)
# ═══════════════════════════════════════════════════════════════════════════════
dec = [
    ('up1', '12 × 12 · 512 ch', 'ConvTranspose  →  cat  →  ResBlock  1024 → 512'),
    ('up2', '24 × 24 · 256 ch', 'ConvTranspose  →  cat  →  ResBlock   512 → 256'),
    ('up3', '48 × 48 · 128 ch', 'ConvTranspose  →  cat  →  ResBlock   256 → 128'),
    ('up4', '96 × 96 ·  64 ch', 'ConvTranspose  →  cat  →  ResBlock   128 →  64'),
]
for i, (name, dims, detail) in enumerate(dec):
    y = Y_LEV[3 - i]
    box(X_DEC, y, f'{name}:  {dims}', detail, color=C_DEC)
    if i < 3:
        arr(X_DEC, Y_LEV[3 - i] + BH / 2, X_DEC, Y_LEV[3 - i - 1] - BH / 2)

# Dec top → Output block
arr(X_DEC, Y_LEV[0] + BH / 2, X_DEC, Y_TOP - BH / 2)

# ═══════════════════════════════════════════════════════════════════════════════
# Skip connections + Attention Gates
# ═══════════════════════════════════════════════════════════════════════════════
for y in Y_LEV:
    skip(y)

# ═══════════════════════════════════════════════════════════════════════════════
# Output block
# ═══════════════════════════════════════════════════════════════════════════════
box(X_DEC, Y_TOP,
    'outc:  96 × 96 · 6 ch  (raw logits)',
    'Conv 1×1   64 → 6',
    color=C_OUT, w=BW + 0.4, fs=9, sfs=7.2)

# ═══════════════════════════════════════════════════════════════════════════════
# Per-pixel output parameters  (bottom-right block)
# ═══════════════════════════════════════════════════════════════════════════════
px, py_top = 13.6, 3.75
pw, ph_gap = 8.0, 0.54
ph_row = 0.44

ax.text(px + pw / 2, py_top + 0.20,
        'Per-pixel outputs  (6 channels, applied per pixel)',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#263238')

params = [
    ('①  p₀',      'fraction_zero',          'sigmoid( logit[0] )'),
    ('②  w',        'mixing weight',           'sigmoid( logit[1] )'),
    ('③  α₁',       'shape₁  (light comp.)',   'shape_min + softplus( logit[2] )'),
    ('④  θ₁',       'scale₁  (light comp.)',   'scale_min + softplus( logit[3] )'),
    ('⑤  α₂',       'shape₂  (heavy comp.)',   'α₁ + softplus( logit[4] ) + 0.5   [hard ordering]'),
    ('⑥  θ₂',       'scale₂  (heavy comp.)',   'scale_min + softplus( logit[5] )'),
]
for j, (sym, name, act) in enumerate(params):
    yj = py_top - j * ph_gap
    r = FancyBboxPatch((px, yj - ph_row / 2), pw, ph_row,
                        boxstyle='round,pad=0.04',
                        fc=C_OUT, ec='white', lw=1.2, alpha=0.88, zorder=3)
    ax.add_patch(r)
    ax.text(px + 0.18, yj + 0.06, f'{sym}:  {name}',
            ha='left', va='center', fontsize=7.8, color='white',
            fontweight='bold', zorder=4)
    ax.text(px + 0.18, yj - 0.12, act,
            ha='left', va='center', fontsize=6.8, color='#FFCDD2', zorder=4)

# Arrow from output block down to parameter list
ax.annotate('', xy=(X_DEC, py_top + 0.22),
            xytext=(X_DEC, Y_TOP - BH / 2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=C_OUT, lw=1.5,
                            connectionstyle='arc3,rad=0.0'), zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# ResidualBlock inset  (bottom-left)
# ═══════════════════════════════════════════════════════════════════════════════
ri_x, ri_y = 0.4, 3.6
ri_w, ri_h = 3.4, 0.44
ri_gap = 0.64
ax.text(ri_x + ri_w / 2, ri_y + 0.55, 'ResidualBlock',
        ha='center', fontsize=8.5, fontweight='bold', color='#263238')

layers = ['Conv3×3 – BN – ReLU', 'Conv3×3 – BN']
for k, lbl in enumerate(layers):
    yk = ri_y - k * ri_gap
    r = FancyBboxPatch((ri_x, yk - ri_h / 2), ri_w, ri_h,
                        boxstyle='round,pad=0.04',
                        fc=C_ENC, ec='white', lw=1.2, zorder=3)
    ax.add_patch(r)
    ax.text(ri_x + ri_w / 2, yk, lbl, ha='center', va='center',
            fontsize=7, color='white', zorder=4)
    if k == 0:
        arr(ri_x + ri_w / 2, yk - ri_h / 2,
            ri_x + ri_w / 2, yk - ri_gap + ri_h / 2, lw=1.2)

# Add + ReLU
yr = ri_y - 2 * ri_gap
r = FancyBboxPatch((ri_x, yr - ri_h / 2), ri_w, ri_h,
                    boxstyle='round,pad=0.04',
                    fc='#455A64', ec='white', lw=1.2, zorder=3)
ax.add_patch(r)
ax.text(ri_x + ri_w / 2, yr, 'Add + ReLU', ha='center', va='center',
        fontsize=7, color='white', zorder=4)
arr(ri_x + ri_w / 2, ri_y - ri_gap - ri_h / 2, ri_x + ri_w / 2, yr + ri_h / 2, lw=1.2)

# Shortcut arrow (right side, curved)
ax.annotate('', xy=(ri_x + ri_w + 0.12, yr + ri_h / 2 + 0.04),
            xytext=(ri_x + ri_w + 0.12, ri_y + ri_h / 2),
            arrowprops=dict(arrowstyle='->', color=C_ENC, lw=1.5,
                            connectionstyle='arc3,rad=-0.35'), zorder=5)
ax.text(ri_x + ri_w + 0.62, (ri_y + yr) / 2, 'shortcut',
        va='center', ha='center', fontsize=6.5, color=C_ENC, rotation=270)

# ═══════════════════════════════════════════════════════════════════════════════
# AttentionGate inset  (bottom-centre)
# ═══════════════════════════════════════════════════════════════════════════════
ai_x, ai_y = 4.5, 3.6
ai_w, ai_h = 4.5, 0.44
ai_gap = 0.62
ax.text(ai_x + ai_w / 2, ai_y + 0.55, 'AttentionGate',
        ha='center', fontsize=8.5, fontweight='bold', color='#263238')

att_steps = [
    'g′ = BN( Conv1×1( g ) )',
    'x′ = BN( Conv1×1( x ) )',
    'ψ  = Sigmoid( BN( Conv1×1( ReLU(g′ + x′) ) ) )',
    'output = x  ×  ψ',
]
for k, lbl in enumerate(att_steps):
    yk = ai_y - k * ai_gap
    r = FancyBboxPatch((ai_x, yk - ai_h / 2), ai_w, ai_h,
                        boxstyle='round,pad=0.04',
                        fc=C_ATTN, ec='white', lw=1.2, alpha=0.90, zorder=3)
    ax.add_patch(r)
    ax.text(ai_x + 0.15, yk, lbl, ha='left', va='center',
            fontsize=6.8, color='white', zorder=4)
    if k < len(att_steps) - 1:
        arr(ai_x + ai_w / 2, yk - ai_h / 2,
            ai_x + ai_w / 2, yk - ai_gap + ai_h / 2, lw=1.1)

# Input labels for attention gate inset
ax.text(ai_x - 0.12, ai_y,
        'g = decoder\n(upsampled)', ha='right', va='center',
        fontsize=6.5, color='#546E7A', style='italic')
ax.text(ai_x - 0.12, ai_y - ai_gap,
        'x = encoder\nskip', ha='right', va='center',
        fontsize=6.5, color='#546E7A', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# Legend  (bottom strip)
# ═══════════════════════════════════════════════════════════════════════════════
legend = [
    (C_ENC,  'Encoder block'),
    (C_BRDG, 'Bridge'),
    (C_DEC,  'Decoder block'),
    (C_ATTN, 'Attention Gate (A)'),
    (C_IN,   'Input'),
    (C_OUT,  'Output / params'),
]
xl, yl = 0.4, 0.42
for k, (col, lbl) in enumerate(legend):
    xk = xl + k * 3.55
    circ = Circle((xk + 0.18, yl), 0.16, fc=col, ec='white', lw=1.0, zorder=4)
    ax.add_patch(circ)
    ax.text(xk + 0.44, yl, lbl, va='center', fontsize=7.5, color='#263238')

# Skip connection legend entry
ax.plot([0.4, 0.85], [0.12, 0.12], color=C_SKIP, lw=1.5, ls='--')
ax.text(0.97, 0.12, '  dashed = skip connection  (attended by A, then concatenated)',
        va='center', fontsize=7.5, color='#263238')

plt.tight_layout(pad=0.1)
plt.savefig('architecture_diagram.png', dpi=200, bbox_inches='tight')
print('Saved architecture_diagram.png')
plt.close()
