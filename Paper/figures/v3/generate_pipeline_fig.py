"""Generates compact publication-quality system architecture pipeline vector figure."""

from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "figure.dpi": 300,
})

FIG_PATH = Path("Paper/figures/v3/pipeline.pdf")


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 2.5))
    ax.axis("off")

    c_data = "#e8f0fe"
    c_feat = "#fce8e6"
    c_norm = "#fef7e0"
    c_model = "#e6f4ea"
    c_out = "#f3e8fd"
    border_color = "#333333"

    def draw_box(x, y, w, h, title, subtitle, color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=color, edgecolor=border_color, linewidth=1.1
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, title, weight="bold", ha="center", va="center", fontsize=8.8)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=7.6, color="#333333")

    draw_box(0.01, 0.28, 0.21, 0.60, "SWE-bench Candidates", "PRs Passing Issue Tests\nN=2,663 across 7 LLMs\nNormals: 2,564 | Regr: 99 (3.7%)", c_data)
    draw_box(0.26, 0.28, 0.22, 0.60, "Multimodal Features", "R0-Struct (12-D Churn/Gap)\nMiniLM Terminal Logs (384-D)\nCodeBERT Patch Diffs (768-D)", c_feat)
    draw_box(0.51, 0.28, 0.20, 0.60, "Variance Equalization", "Fit scales on y=0 train\nalpha_m = 1 / sqrt(Var_m)\nPrevents log dominance", c_norm)
    draw_box(0.74, 0.28, 0.25, 0.60, "CM-BAE Bottleneck AE", "1164 -> 256 -> 64 -> 256 -> 1164\nHuber Loss on y=0 Normals\nAdditive Score s(x) = sum s_m", c_model)

    arrow_props = dict(facecolor="#444444", edgecolor="#444444", width=1.2, headwidth=5, headlength=5)
    ax.annotate("", xy=(0.26, 0.58), xytext=(0.22, 0.58), arrowprops=arrow_props)
    ax.annotate("", xy=(0.51, 0.58), xytext=(0.48, 0.58), arrowprops=arrow_props)
    ax.annotate("", xy=(0.74, 0.58), xytext=(0.71, 0.58), arrowprops=arrow_props)

    triage_rect = patches.FancyBboxPatch(
        (0.18, 0.02), 0.64, 0.18,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=c_out, edgecolor=border_color, linewidth=1.0, linestyle="--"
    )
    ax.add_patch(triage_rect)
    ax.text(
        0.50, 0.11,
        r"$\bf{CI/CD\ Triage\ Decision:}$ Platt Calibration $\hat{p}(y=1|s)$ $\rightarrow$ Cost-Optimal Cutoff $\tau^*$ ($C_{\text{escaped}}=\$1,200$, $C_{\text{audit}}=\$30$)",
        ha="center", va="center", fontsize=8.0, color="#222222"
    )
    ax.annotate("", xy=(0.50, 0.20), xytext=(0.86, 0.28),
                arrowprops=dict(facecolor="#555555", edgecolor="#555555", width=1.0, headwidth=4, headlength=4, connectionstyle="arc3,rad=-0.20"))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.95)

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, bbox_inches="tight")
    plt.close()
    print(f"Saved compact pipeline architecture figure to {FIG_PATH}")


if __name__ == "__main__":
    draw_pipeline()
