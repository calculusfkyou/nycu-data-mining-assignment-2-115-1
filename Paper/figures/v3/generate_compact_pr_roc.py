"""Generates compact PR and ROC curve figure from saved models/data."""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Model.baselines_v3 import AllNormalBaseline, PatchChurnBaseline, UnverifiedEditHeuristic
from Model.cm_bae_v3 import CMBAE
from Model.tier2_controls_v3 import InductiveECOD, StandardizedDualKNN, TabularIsolationForest
from Model.tier3_supervised_v3 import SupervisedMultimodalMLP, SupervisedXGBoost
from Model.experiment_v3 import STRUCT_COLS

FEATURES_PATH = Path("Dataset/verified/candidate_features.csv")
MINILM_PATH = Path("Dataset/verified/embeddings/minilm_logs.npy")
CODEBERT_PATH = Path("Dataset/verified/embeddings/codebert_diffs.npy")
SPLITS_PATH = Path("Dataset/verified/splits.json")
FIG_PATH = Path("Paper/figures/v3/pr_roc_curves.pdf")


def main():
    df = pd.read_csv(FEATURES_PATH)
    minilm = np.load(MINILM_PATH)
    codebert = np.load(CODEBERT_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    X_struct = df[STRUCT_COLS].values.astype(np.float32)
    y_all = df["label"].values.astype(int)

    # Use 2 folds for high-fidelity curve representation
    curve_data = {
        "All-Normal": {"y_true": [], "y_score": []},
        "3-Sigma Churn": {"y_true": [], "y_score": []},
        "Unverified-Edit": {"y_true": [], "y_score": []},
        "Inductive ECOD": {"y_true": [], "y_score": []},
        "Isolation Forest": {"y_true": [], "y_score": []},
        "Dual k-NN": {"y_true": [], "y_score": []},
        "CM-BAE (Primary)": {"y_true": [], "y_score": []},
        "Supervised XGBoost": {"y_true": [], "y_score": []},
        "Supervised MLP": {"y_true": [], "y_score": []},
    }

    for fold in splits["folds"][:3]:
        train_idx = fold["train_indices"]
        train_norm_idx = fold["train_normal_indices"]
        val_idx = fold["val_indices"]
        test_idx = fold["test_indices"]
        y_test = y_all[test_idx]

        scaler = StandardScaler()
        X_s_norm = scaler.fit_transform(X_struct[train_norm_idx])
        X_s_train = scaler.transform(X_struct[train_idx])
        X_s_val = scaler.transform(X_struct[val_idx])
        X_s_test = scaler.transform(X_struct[test_idx])
        X_fused_train = np.hstack([X_s_train, minilm[train_idx], codebert[train_idx]])
        X_fused_test = np.hstack([X_s_test, minilm[test_idx], codebert[test_idx]])

        # T1
        s_all_norm = np.zeros(len(test_idx))
        m_churn = PatchChurnBaseline().fit(df.iloc[train_norm_idx])
        s_churn = m_churn.score(df.iloc[test_idx])
        m_unv = UnverifiedEditHeuristic().fit(df.iloc[train_norm_idx])
        s_unv = m_unv.score(df.iloc[test_idx])

        # T2
        m_ecod = InductiveECOD().fit(X_s_norm)
        s_ecod = m_ecod.score(X_s_test)
        m_if = TabularIsolationForest(random_state=42).fit(X_s_norm)
        s_if = m_if.score(X_s_test)
        m_knn = StandardizedDualKNN(k=15).fit(minilm[train_norm_idx], codebert[train_norm_idx])
        s_knn = m_knn.score(minilm[test_idx], codebert[test_idx])

        # CM-BAE
        cm_bae = CMBAE(bottleneck_dim=64, epochs=25, seed=42)
        cm_bae.fit(X_struct[train_norm_idx], minilm[train_norm_idx], codebert[train_norm_idx])
        cm_bae.calibrate(X_struct[val_idx], minilm[val_idx], codebert[val_idx], y_all[val_idx])
        p_bae = cm_bae.predict_proba(X_struct[test_idx], minilm[test_idx], codebert[test_idx])

        # T3
        m_xgb = SupervisedXGBoost(scale_pos_weight=25.9, seed=42).fit(X_fused_train, y_all[train_idx])
        s_xgb = m_xgb.predict_proba(X_fused_test)
        m_mlp = SupervisedMultimodalMLP(input_dim=X_fused_train.shape[1], epochs=15, seed=42).fit(X_fused_train, y_all[train_idx])
        s_mlp = m_mlp.predict_proba(X_fused_test)

        preds_dict = {
            "All-Normal": s_all_norm,
            "3-Sigma Churn": s_churn,
            "Unverified-Edit": s_unv,
            "Inductive ECOD": s_ecod,
            "Isolation Forest": s_if,
            "Dual k-NN": s_knn,
            "CM-BAE (Primary)": p_bae,
            "Supervised XGBoost": s_xgb,
            "Supervised MLP": s_mlp,
        }

        for k, v in preds_dict.items():
            curve_data[k]["y_true"].append(y_test)
            curve_data[k]["y_score"].append(v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.7))

    colors = {
        "All-Normal": "#888888",
        "3-Sigma Churn": "#999999",
        "Unverified-Edit": "#7570b3",
        "Inductive ECOD": "#1b9e77",
        "Isolation Forest": "#e6ab02",
        "Dual k-NN": "#e7298a",
        "CM-BAE (Primary)": "#d95f02",
        "Supervised XGBoost": "#2b5c8f",
        "Supervised MLP": "#1f78b4",
    }

    linestyles = {
        "All-Normal": ":",
        "3-Sigma Churn": "-.",
        "Unverified-Edit": "-.",
        "Inductive ECOD": "--",
        "Isolation Forest": "--",
        "Dual k-NN": "--",
        "CM-BAE (Primary)": "-",
        "Supervised XGBoost": "-",
        "Supervised MLP": "-",
    }

    for model in curve_data.keys():
        y_true = np.concatenate(curve_data[model]["y_true"])
        y_score = np.concatenate(curve_data[model]["y_score"])

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        label_str = f"{model} ({ap:.3f})"
        axes[0].plot(
            rec, prec, label=label_str, color=colors[model], linestyle=linestyles[model],
            linewidth=2.2 if "CM-BAE" in model else 1.1,
            zorder=10 if "CM-BAE" in model else 3
        )

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_val = roc_auc_score(y_true, y_score)
        axes[1].plot(
            fpr, tpr, label=f"{model} ({roc_val:.3f})", color=colors[model], linestyle=linestyles[model],
            linewidth=2.0 if "CM-BAE" in model else 1.1,
            zorder=10 if "CM-BAE" in model else 3
        )

    prevalence = float(np.mean(y_all))
    axes[0].axhline(y=prevalence, color="gray", linestyle=":", label=f"Random ({prevalence:.3f})")
    axes[0].set_title("(a) Precision-Recall Curves (1:27 Imbalance)")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.92)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])

    axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", label="Chance (0.500)")
    axes[1].set_title("(b) Receiver Operating Characteristic (ROC)")
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])

    plt.tight_layout()
    plt.savefig(FIG_PATH, bbox_inches="tight")
    plt.savefig(FIG_PATH.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved compact PR/ROC curves to {FIG_PATH} and {FIG_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
