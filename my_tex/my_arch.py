import sys
# This path must point to your PlotNeuralNet installation directory
sys.path.append('/opt/anaconda3/bin/PlotNeuralNet')

from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('.'), 
    to_cor(),
    to_begin(),

    # --- CUSTOM COLOR DEFINITION ---
    # Defining a Teal color for the Attention Gates
    "\\definecolor{AttentionColor}{rgb}{0.0, 0.5, 0.5}",

    # --- TITLE ---
    "\\node at (15, 9, 0) [scale=4, font=\\bfseries] {GRAF Attention Residual U-Net};",

    # --- ENCODER ---
    to_Conv("inc", 64, 5, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="Res-Inc"),
    
    to_Pool("pool1", offset="(0.7,0,0)", to="(inc-east)"),
    to_Conv("down1", 128, 64, offset="(1.7,0,0)", to="(pool1-east)", height=32, depth=32, width=3),

    to_Pool("pool2", offset="(0.7,0,0)", to="(down1-east)"),
    to_Conv("down2", 256, 128, offset="(1.7,0,0)", to="(pool2-east)", height=24, depth=24, width=4),

    to_Pool("pool3", offset="(0.7,0,0)", to="(down2-east)"),
    to_Conv("down3", 512, 256, offset="(1.7,0,0)", to="(pool3-east)", height=16, depth=16, width=5),

    # --- BRIDGE ---
    to_Pool("pool4", offset="(0.7,0,0)", to="(down3-east)"),
    to_Conv("bridge", 1024, 512, offset="(1.7,0,0)", to="(pool4-east)", height=8, depth=8, width=8, caption="Bridge"),

    # --- DECODER with Attention Gates ---
    # We use a LaTeX "hack" in the caption to change the fill color since the Python function lacks the argument.
    to_UnPool("up1", offset="(3.0,0,0)", to="(bridge-east)", height=16, depth=16, width=8),
    "\\def\\ConvColor{AttentionColor}", # Switch color to AttentionColor
    to_Conv("att1", 512, 512, offset="(1.8,0,0)", to="(up1-east)", height=16, depth=16, width=5, caption="Attn1"),
    to_connection("down3", "att1"), 
    "\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}", # Switch back to default Yellow
    to_Conv("conv_dec1", 512, 1024, offset="(1.8,0,0)", to="(att1-east)", height=16, depth=16, width=5),

    to_UnPool("up2", offset="(3.0,0,0)", to="(conv_dec1-east)", height=24, depth=24, width=5),
    "\\def\\ConvColor{AttentionColor}",
    to_Conv("att2", 256, 256, offset="(1.8,0,0)", to="(up2-east)", height=24, depth=24, width=4),
    to_connection("down2", "att2"),
    "\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}",
    to_Conv("conv_dec2", 256, 512, offset="(1.8,0,0)", to="(att2-east)", height=24, depth=24, width=4),

    to_UnPool("up3", offset="(3.0,0,0)", to="(conv_dec2-east)", height=32, depth=32, width=4),
    "\\def\\ConvColor{AttentionColor}",
    to_Conv("att3", 128, 128, offset="(1.8,0,0)", to="(up3-east)", height=32, depth=32, width=3),
    to_connection("down1", "att3"),
    "\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}",
    to_Conv("conv_dec3", 128, 256, offset="(1.8,0,0)", to="(att3-east)", height=32, depth=32, width=3),

    to_UnPool("up4", offset="(3.5,0,0)", to="(conv_dec3-east)", height=40, depth=40, width=3),
    "\\def\\ConvColor{AttentionColor}",
    to_Conv("att4", 64, 64, offset="(2.2,0,0)", to="(up4-east)", height=40, depth=40, width=2),
    to_connection("inc", "att4"),
    "\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}",
    to_Conv("conv_dec4", 64, 128, offset="(2.2,0,0)", to="(att4-east)", height=40, depth=40, width=2),

    # --- OUTPUT ---
    to_Conv("outc", 101, 64, offset="(4.5,0,0)", to="(conv_dec4-east)", height=40, depth=40, width=1, caption="Output"),

    # --- LEGEND ---
    to_Conv("legend_conv", "", "", offset="(5,-7,0)", to="(inc-anchor)", height=3, depth=3, width=2, caption="\\Large Conv/ResBlock"),
    "\\def\\ConvColor{AttentionColor}",
    to_Conv("legend_att", "", "", offset="(12,0,0)", to="(legend_conv-east)", height=3, depth=3, width=2, caption="\\Large Attention Gate"),
    "\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}",
    to_Pool("legend_pool", offset="(12,0,0)", to="(legend_att-east)", height=3, depth=3, width=2, caption="\\Large Max Pool"),
    to_UnPool("legend_unpool", offset="(12,0,0)", to="(legend_pool-east)", height=3, depth=3, width=2, caption="\\Large Upsample"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
