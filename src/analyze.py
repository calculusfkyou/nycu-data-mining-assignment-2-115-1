"""§5 pipeline figure, §7 significance, §8 failure cases, heatmaps, H1/H2 checks and practical impact.

Runs after the stage 6 / 7 test evaluation and changes no decision: models are refitted with the same
configs and seeds only to recover heatmaps (their test scores are asserted equal to stage 6).

Usage:  python src/analyze.py        # writes results/stage8_analysis/*.csv and fig_*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from data import CLASSES, IMG_SIZE, ROOT, VisADataset
from eda import GROUP
from features import load_features
from methods import PaDiM, PatchCore
import run

RESULTS = ROOT / "results" / "stage8_analysis"
MAIN = ROOT / "results" / "stage6_main"
SEEDS = (0, 1, 2)
N_BOOT = 1000
PI = 0.01


def scores(obj, tier, seed):
    z = np.load(MAIN / "scores" / f"{obj}_{tier}_s{seed}.npz", allow_pickle=True)
    return z["label"], z["score"], float(z["tau"]), z["image"]


def affected():
    modes = pd.read_csv(ROOT / "results" / "stage4_eda" / "eda_acquisition_modes.csv")
    return modes.loc[modes.affected, "object"].tolist()


# ---------------------------------------------------------------- §7 significance
def significance():
    """Paired bootstrap of the class-averaged PR-AUC gap (T3 - T2): resample test images within each class
    and label (prevalence kept), PR-AUC averaged over the 3 seeds; plus a Wilcoxon test over the 12 classes
    and a paired t-test over seeds."""
    rng = np.random.default_rng(0)
    data = {o: {t: [scores(o, t, s) for s in SEEDS] for t in ("T2_padim", "T3_patchcore")} for o in CLASSES}
    gaps = {o: [] for o in CLASSES}
    for _ in range(N_BOOT):
        for o in CLASSES:
            y = data[o]["T2_padim"][0][0]
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
            ap = {t: np.mean([average_precision_score(y[idx], d[1][idx]) for d in data[o][t]]) for t in data[o]}
            gaps[o].append(ap["T3_patchcore"] - ap["T2_padim"])
    gaps = pd.DataFrame(gaps)                                       # (N_BOOT, 12)
    m = pd.read_csv(MAIN / "metrics.csv")
    per_class = m.groupby(["object", "tier"]).pr_auc.mean().unstack()
    per_seed = m[m.tier.isin(["T2_padim", "T3_patchcore"])].groupby(["seed", "tier"]).pr_auc.mean().unstack()
    rows = []
    for subset, objs in [("12 classes", CLASSES), ("9 classes", [o for o in CLASSES if o not in affected()])]:
        g = gaps[objs].mean(1)
        pc = per_class.loc[objs]
        rows.append({"subset": subset, "comparison": "T3 - T2 PR-AUC",
                     "point": (pc.T3_patchcore - pc.T2_padim).mean(),
                     "boot_ci_low": g.quantile(0.025), "boot_ci_high": g.quantile(0.975),
                     "wilcoxon_p": wilcoxon(pc.T3_patchcore, pc.T2_padim).pvalue,
                     "classes_T3_better": int((pc.T3_patchcore > pc.T2_padim).sum()), "n_classes": len(objs)})
        rows.append({"subset": subset, "comparison": "T2 - T1 PR-AUC",
                     "point": (pc.T2_padim - pc.T1_template).mean(),
                     "wilcoxon_p": wilcoxon(pc.T2_padim, pc.T1_template).pvalue,
                     "classes_T2_better": int((pc.T2_padim > pc.T1_template).sum()), "n_classes": len(objs)})
    rows.append({"subset": "12 classes", "comparison": "T3 - T2 PR-AUC, paired t-test over seeds",
                 "point": (per_seed.T3_patchcore - per_seed.T2_padim).mean(),
                 "ttest_p": ttest_rel(per_seed.T3_patchcore, per_seed.T2_padim).pvalue})
    pd.DataFrame(rows).to_csv(RESULTS / "significance.csv", index=False)


# ---------------------------------------------------------------- heatmaps (refit, seed 0)
def masks_and_images(frame):
    ds_m, ds_i = VisADataset(frame), VisADataset(frame, normalize=False)
    masks = np.stack([ds_m[i][2][0].numpy() > 0 for i in range(len(frame))])
    images = np.stack([ds_i[i][0].permute(1, 2, 0).numpy() for i in range(len(frame))])
    return masks, images


def up(maps):
    t = torch.from_numpy(np.asarray(maps, np.float32))[:, None]
    return F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[:, 0].numpy()


def heatmap_analysis():
    """Per class: T2 / T3 seed-0 heatmaps on test -> pixel AUROC and whether the peak lies in the mask (+8 px);
    T2 patch vs global-pooled detection by defect size (H2, all 3 seeds)."""
    pix, peak, h2, keep = [], [], [], {}
    for obj in CLASSES:
        _, train = load_features(obj, run.L234, splits=["train"])
        fv, val = load_features(obj, run.L234, splits=["val"])
        frame, test = load_features(obj, run.L234, splits=["test"])
        y_val = (fv.label == "anomaly").to_numpy().astype(int)
        y = (frame.label == "anomaly").to_numpy().astype(int)
        masks, images = masks_and_images(frame)
        area = masks.reshape(len(masks), -1).mean(1)
        grown = np.stack([binary_dilation(m, iterations=8) for m in masks])

        maps = {}
        for tier, cls in (("T2_padim", PaDiM), ("T3_patchcore", PatchCore)):
            s, heat = cls(seed=0, **run.CONFIG[tier]).fit(train).score(test)
            ref_y, ref_s, tau, ref_img = scores(obj, tier, 0)
            assert (ref_img == frame.image.to_numpy()).all() and np.allclose(s, ref_s, rtol=1e-4, atol=1e-4), (obj, tier)
            h = up(heat)
            maps[tier] = (s, tau, h)
            pix.append({"object": obj, "tier": tier, "pixel_auroc": roc_auc_score(masks.ravel(), h.ravel())})
            flat = h.reshape(len(h), -1).argmax(1)
            hit = grown.reshape(len(h), -1)[np.arange(len(h)), flat]
            peak.append({"object": obj, "tier": tier, "n_anomaly": int(y.sum()),
                         "peak_in_mask_pct": 100 * hit[y == 1].mean()})
        keep[obj] = {"frame": frame, "y": y, "masks": masks, "images": images, "area": area, "maps": maps}

        for seed in SEEDS:  # H2: patch-level (full) vs global pooled, detection by defect size
            for variant, kw in (("patch-level (full)", {}), ("global pooled", {"pooled": True})):
                m = PaDiM(eps=0.01, seed=seed, **run.CONFIG["T2_padim"], **kw).fit(train)
                tau = run.best_f2_tau(y_val, m.score(val)[0])
                s = m.score(test)[0]
                for i in np.flatnonzero(y == 1):
                    h2.append({"object": obj, "seed": seed, "variant": variant, "area": area[i],
                               "detected": bool(s[i] >= tau), "s": s[i], "tau": tau})
        print(f"{obj}: heatmaps done", flush=True)

    pd.DataFrame(pix).to_csv(RESULTS / "pixel_auroc.csv", index=False)
    peak = pd.DataFrame(peak)
    peak.to_csv(RESULTS / "peak_in_mask.csv", index=False)

    h2 = pd.DataFrame(h2)
    edges = np.quantile(h2.area.unique(), [0, 1 / 3, 2 / 3, 1])
    h2["area_bin"] = pd.cut(h2.area, edges, include_lowest=True,
                            labels=[f"small (<{100 * edges[1]:.2f}%)", f"medium ({100 * edges[1]:.2f}-{100 * edges[2]:.2f}%)",
                                    f"large (>{100 * edges[2]:.2f}%)"])
    rows = []
    for subset, part in [("12 classes", h2), ("9 classes", h2[~h2.object.isin(affected())])]:
        r = part.groupby(["area_bin", "variant", "seed"], observed=True).detected.mean().groupby(
            ["area_bin", "variant"], observed=True).agg(["mean", "std"]).reset_index()
        r.insert(0, "subset", subset)
        r["n_anomalies"] = part.groupby(["area_bin", "variant"], observed=True).size().values // len(SEEDS)
        rows.append(r)
    pd.concat(rows).rename(columns={"mean": "recall_mean", "std": "recall_std"}).to_csv(
        RESULTS / "h2_detection_by_area.csv", index=False)
    return keep


# ---------------------------------------------------------------- §8 failure cases
def failure_cases(keep):
    """T2 seed 0, all classes, ranked by s / tau: the most inverted and the nearest-miss FN, likewise FP."""
    rows = []
    for obj, k in keep.items():
        s, tau, _ = k["maps"]["T2_padim"]
        for i in range(len(s)):
            rows.append({"object": obj, "i": i, "image": k["frame"].image[i], "label": int(k["y"][i]),
                         "stratum": k["frame"].stratum[i], "s": s[i], "tau": tau, "s_over_tau": s[i] / tau,
                         "defect_area_pct": 100 * k["area"][i]})
    df = pd.DataFrame(rows)
    fn = df[(df.label == 1) & (df.s < df.tau)].sort_values("s_over_tau")
    fp = df[(df.label == 0) & (df.s >= df.tau)].sort_values("s_over_tau")
    picks = [("FN, complete inversion", fn.iloc[0]), ("FN, near miss", fn.iloc[-1]),
             ("FP, complete inversion", fp.iloc[-1]), ("FP, near miss", fp.iloc[0])]
    cases = pd.DataFrame([{"case": n + 1, "type": t, **r.drop("i").to_dict()} for n, (t, r) in enumerate(picks)])
    cases["fn_count_T2_s0"], cases["fp_count_T2_s0"] = len(fn), len(fp)
    cases.to_csv(RESULTS / "failure_cases.csv", index=False)

    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 8))
    for j, (t, r) in enumerate(picks):
        k = keep[r.object]
        img, h, mask = k["images"][r.i], k["maps"]["T2_padim"][2][r.i], k["masks"][r.i]
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"#{j + 1} {t}\n{r.object}: {r.stratum}", fontsize=9)
        axes[1, j].imshow(img)
        im = axes[1, j].imshow(h, cmap="jet", alpha=0.45)
        if mask.any():
            for ax in axes[:, j]:
                ax.contour(mask, colors="white", linewidths=1)
        axes[1, j].set_title(f"s(x) = {r.s:.1f}, tau = {r.tau:.1f} (s/tau = {r.s_over_tau:.2f})", fontsize=9)
        fig.colorbar(im, ax=axes[1, j], fraction=0.046, label="Mahalanobis distance")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Tier 2 failure cases (test, seed 0); white contour = ground-truth defect mask")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_failure_cases.png", dpi=150)
    plt.close(fig)


def heatmap_gallery(keep):
    """One correctly flagged anomaly per structure group, T2 vs T3 heatmaps (interpretability, §8)."""
    picks = []
    for obj in ("pcb4", "chewinggum", "candle"):
        k = keep[obj]
        s, tau, _ = k["maps"]["T2_padim"]
        tp = np.flatnonzero((k["y"] == 1) & (s >= tau))
        picks.append((obj, tp[np.argmax(s[tp] / tau)]))
    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.4 * len(picks)))
    for r, (obj, i) in enumerate(picks):
        k = keep[obj]
        axes[r, 0].imshow(k["images"][i])
        axes[r, 0].contour(k["masks"][i], colors="white", linewidths=1)
        axes[r, 0].set_title(f"{obj}: {k['frame'].stratum[i]} (mask in white)", fontsize=9)
        for c, tier in enumerate(("T2_padim", "T3_patchcore"), 1):
            s, tau, h = k["maps"][tier]
            axes[r, c].imshow(k["images"][i])
            axes[r, c].imshow(h[i], cmap="jet", alpha=0.45)
            axes[r, c].contour(k["masks"][i], colors="white", linewidths=1)
            axes[r, c].set_title(f"{tier}: s/tau = {s[i] / tau:.2f}", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_heatmaps.png", dpi=150)
    plt.close(fig)
    pd.DataFrame([{"object": o, "image": keep[o]["frame"].image[i]} for o, i in picks]).to_csv(
        RESULTS / "heatmap_gallery.csv", index=False)


# ---------------------------------------------------------------- H1, practical impact, pipeline figure
def h1_groups():
    """T3 - T2 PR-AUC per class and by structure group, for 12 and 9 classes."""
    m = pd.read_csv(MAIN / "metrics.csv")
    pc = m.groupby(["object", "tier"]).pr_auc.mean().unstack()
    pc["T3_minus_T2"] = pc.T3_patchcore - pc.T2_padim
    pc["group"] = [GROUP[o] for o in pc.index]
    pc["affected"] = pc.index.isin(affected())
    pc.to_csv(RESULTS / "h1_per_class.csv")
    rows = []
    for subset, part in [("12 classes", pc), ("9 classes", pc[~pc.affected])]:
        for g, x in part.groupby("group"):
            rows.append({"subset": subset, "group": g, "n": len(x), "T2": x.T2_padim.mean(),
                         "T3": x.T3_patchcore.mean(), "T3_minus_T2": x.T3_minus_T2.mean()})
    pd.DataFrame(rows).to_csv(RESULTS / "h1_by_group.csv", index=False)


def practical_impact():
    """Per 10,000 inspected items at prevalence 1 %: missed defects = 100 (1 - TPR), false alarms = 9,900 FPR."""
    m = pd.read_csv(MAIN / "metrics.csv")
    m = m[m.tier.isin(["T1_template", "T2_padim", "T3_patchcore"])]
    rate = m.groupby(["tier", "seed"])[["recall", "fpr"]].mean().groupby("tier").mean()
    rate["tau_rule"] = "best-F2 on val"
    a = pd.read_csv(ROOT / "results" / "stage7_ablation" / "ablation_summary.csv")
    a = a[(a.subset == "12 classes") & (a.ablation == "A2_threshold")]
    for _, r in a.iloc[1:].iterrows():
        rate.loc[f"T2_padim ({r.setting})"] = [r.recall_mean, r.fpr_mean, r.setting]
    n = 10_000
    rate["missed_per_10k"] = n * PI * (1 - rate.recall)
    rate["false_alarms_per_10k"] = n * (1 - PI) * rate.fpr
    rate["flagged_per_10k"] = n * PI * rate.recall + rate.false_alarms_per_10k
    rate["true_defects_among_flagged_pct"] = 100 * n * PI * rate.recall / rate.flagged_per_10k
    rate.to_csv(RESULTS / "practical_impact.csv")


def pipeline_figure():
    steps = ["Input image\n(1274-1562 px)", "Resize 224x224\n+ ImageNet norm.", "WideResNet50\nlayer2 / 3 / 4",
             "Random 200 of\n3584 channels", "Gaussian per location\n(mu_ij, Sigma_ij + 0.01 I)",
             "Mahalanobis map\n28 x 28", "s(x) = max_ij s_ij", "s(x) >= tau ?\nalert / pass"]
    gap = 2.4
    fig, ax = plt.subplots(figsize=(18, 2.4))
    for i, s in enumerate(steps):
        ax.text(i * gap, 0, s, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#dbe9f6" if i < 7 else "#fde2c8", ec="0.3"))
        if i:
            ax.annotate("", xy=(i * gap - 1.05, 0), xytext=(i * gap - 1.35, 0), arrowprops=dict(arrowstyle="->"))
    ax.text(7 * gap, -0.75, "tau: best F2 on val (normal + 20 anomalies per class)", ha="center", fontsize=8)
    ax.set_xlim(-1.2, 7 * gap + 1.2)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    run.choose_configs()
    print("configs:", run.CONFIG)
    pipeline_figure()
    h1_groups()
    practical_impact()
    significance()
    keep = heatmap_analysis()
    failure_cases(keep)
    heatmap_gallery(keep)
    for f in sorted(RESULTS.glob("*.csv")):
        print(f"\n== {f.name}")
        print(pd.read_csv(f).round(3).to_string(index=False))
