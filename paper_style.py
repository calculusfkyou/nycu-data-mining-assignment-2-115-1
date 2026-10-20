"""Paper figure shared styling adhering to Nature / ACL presentation standards.

Based on Nature Methods / ACL guidelines:
- 8 pt labels, 7 pt ticks, 0.8 pt axis line width
- Serif Times New Roman matching ACL template
- High contrast, colorblind-friendly palette
- Both PDF (vector) and PNG (300 DPI) outputs
"""

import matplotlib.pyplot as plt

# Nature-inspired palette
BLUE_MAIN = "#0F4D92"
BLUE_2 = "#3775BA"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
RED = "#B64342"
ORANGE = "#E66101"
REF_GRAY = "#767676"
LIGHT_GRAY = "#D9D9D9"
INK = "#1a1a1a"
INK2 = "#4d4d4d"

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)


def panel_labels_row(fig, axes, labels):
    """Bold panel labels aligned at the top left of each axis box."""
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    top = max(b.y1 for b in boxes) + 3 / 72 / fig.get_figheight()
    for box, label in zip(boxes, labels):
        fig.text(
            box.x0 - 4 / 72 / fig.get_figwidth(),
            top,
            label,
            fontsize=8,
            weight="bold",
            va="bottom",
            ha="right",
        )


def save(fig, path_without_suffix):
    """Save both vector PDF and high-res PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(
            f"{path_without_suffix}.{ext}",
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
