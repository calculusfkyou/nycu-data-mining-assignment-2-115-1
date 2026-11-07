"""Format saved v3 results for fixed-width, full-size report panels (no training)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Paper/generated/v3"


def panel(name, header, rows, caption, label, columns):
    text = (
        r"\begin{center}\small\setlength{\tabcolsep}{5pt}" + "\n"
        + r"\begin{tabular}{" + columns + "}\n"
        + r"\toprule" + "\n" + header + r" \\" + "\n"
        + r"\midrule" + "\n" + "\n".join(rows) + "\n"
        + r"\bottomrule\end{tabular}" + "\n"
        + r"\reportcaption{table}{" + caption + "}{" + label + "}\n"
        + r"\end{center}" + "\n"
    )
    (OUT / name).write_text(text, encoding="utf-8")


def main():
    data = json.loads((ROOT / "Model/benchmark_results_v3.json").read_text())
    rows = []
    for x in data["summary"]:
        method = x["model"].split(": ", 1)[-1]
        rows.append(
            f"{method} & {x['pr_auc_mean']:.3f} $\\pm$ {x['pr_auc_std']:.3f}"
            f" & {x['f2_mean']:.3f} & {x['roc_auc_mean']:.3f}"
            f" & {100*x['recall_mean']:.1f}\\% & {100*x['fpr_mean']:.1f}\\%" + r" \\"
        )
    panel("layout_main.tex", r"Method & AP $\uparrow$ & $F_2$ & ROC-AUC & Recall & FPR",
          rows, "Five-fold benchmark (seed 42). AP is mean $\\pm$ fold standard deviation. "
          "Threshold metrics use validation-selected operating points. CM-BAE is the proposed model, not the best performer.",
          "tab:main_benchmark", "lccccc")
    ab = json.loads((ROOT / "Model/ablation_results_v3.json").read_text())
    rows = [f"{name.replace(' + ', '+')} & {v['pr_auc_mean']:.3f} $\\pm$ {v['pr_auc_std']:.3f} & {v['f2_mean']:.3f}" + r" \\"
            for name, v in ab["modality_ablation"].items()]
    panel("layout_ablation.tex", r"Variant & AP $\uparrow$ & $F_2$ $\uparrow$", rows,
          "Saved modality ablations. Structural-only and raw fusion outperform the full equalized model.",
          "tab:ablation", "lcc")
    # Preserve the saved EDA values; remove its float so it stays on the data page.
    src = (OUT / "v3_stats.tex").read_text()
    table = src[src.index(r"\begin{tabular}"):src.index(r"\end{tabular}")+len(r"\end{tabular}")]
    table = table.replace(r"\textbf{Candidate Feature}", "Feature").replace(r"\textbf{Deceptive ($y=1$)}", r"Regression ($y=1$)")
    table = table.replace(r"\textbf{Wasserstein $W$}", "$W$").replace(r"\textbf{KS $D$}", "$D$")
    (OUT / "layout_stats.tex").write_text(
        r"\begin{center}\small\setlength{\tabcolsep}{5pt}" + "\n" + table + "\n"
        + r"\reportcaption{table}{Fold-0 training statistics: median [IQR], standardized Wasserstein distance $W$, and KS statistic $D$. Clean $n=1,540$; regression $n=61$. Stars denote unadjusted KS tests ($^{**}p<0.01$, $^{***}p<0.001$).}{tab:eda_stats}"
        + "\n" + r"\end{center}" + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
