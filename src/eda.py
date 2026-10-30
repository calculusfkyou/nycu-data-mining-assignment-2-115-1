"""§3 dataset tables and sample figure, §4 exploratory feature analysis, §5 center-crop check.

Never reads the test split's images, features or masks: feature statistics use train normal
(fit) and val normal / val anomaly (held-out contrast); mask statistics use val + contam anomalies.
Test appears only as label counts in the split/imbalance tables.

Usage:  python src/eda.py        # writes results/stage4_eda/eda_*.csv and fig_*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import CLASSES, ROOT, VISA_DIR, IMG_SIZE, load_split
from features import load_features

RESULTS = ROOT / "results" / "stage4_eda"
FEATURE_SPLITS = ["train", "val"]
MASK_SPLITS = ["val", "contam"]
# VisA paper (Zou et al., 2022) groups the 12 objects into three structure types
GROUP = {**dict.fromkeys(["candle", "capsules", "macaroni1", "macaroni2"], "multiple instances"),
         **dict.fromkeys(["pcb1", "pcb2", "pcb3", "pcb4"], "complex structure"),
         **dict.fromkeys(["cashew", "chewinggum", "fryum", "pipe_fryum"], "single instance")}
D_RAW = IMG_SIZE * IMG_SIZE * 3


# ---------------------------------------------------------------- §3 tables
def split_tables(split):
    counts = split.groupby(["object", "split", "label"]).size().unstack(["split", "label"]).fillna(0).astype(int)
    counts.columns = [f"{s}_{l}" for s, l in counts.columns]
    counts.loc["total"] = counts.sum()
    counts.to_csv(RESULTS / "eda_split_counts.csv")

    rows = []
    for name, part in [("full dataset", split), ("test", split[split.split == "test"]),
                       ("val", split[split.split == "val"])]:
        n_a, n_n = (part.label == "anomaly").sum(), (part.label == "normal").sum()
        rows.append({"subset": name, "N": len(part), "normal": n_n, "anomaly": n_a,
                     "anomaly_pct": 100 * n_a / len(part), "ratio_anomaly_to_normal": f"1 : {n_n / n_a:.2f}",
                     "all_normal_accuracy_pct": 100 * n_n / len(part)})
    for pi in (0.01, 0.05):  # deployment prevalence used for prec_pi
        rows.append({"subset": f"deployment pi={pi:.0%}", "anomaly_pct": 100 * pi,
                     "ratio_anomaly_to_normal": f"1 : {(1 - pi) / pi:.0f}",
                     "all_normal_accuracy_pct": 100 * (1 - pi)})
    pd.DataFrame(rows).to_csv(RESULTS / "eda_imbalance.csv", index=False)


def summary_table(split):
    sizes = np.array([Image.open(VISA_DIR / p).size for p in split.image])  # header read only
    m = split[split.split.isin(MASK_SPLITS) & (split.label == "anomaly")]
    areas = np.array([(np.asarray(Image.open(VISA_DIR / p).convert("L")) > 0).mean() for p in m["mask"]])
    rows = [
        ("N images", len(split)), ("normal", (split.label == "normal").sum()),
        ("anomaly", (split.label == "anomaly").sum()), ("objects", split.object.nunique()),
        ("defect types (first label)", split[split.label == "anomaly"].stratum.nunique()),
        ("raw width px (min / median / max)", f"{sizes[:, 0].min()} / {np.median(sizes[:, 0]):.0f} / {sizes[:, 0].max()}"),
        ("raw height px (min / median / max)", f"{sizes[:, 1].min()} / {np.median(sizes[:, 1]):.0f} / {sizes[:, 1].max()}"),
        ("input D after resize", f"{IMG_SIZE}x{IMG_SIZE}x3 = {D_RAW:,}"),
        ("Tier 2 feature dim per position", "1536 (layer2+3) -> 100 random"),
        ("defect area % of image, val+contam (median)", f"{100 * np.median(areas):.2f}"),
        ("defect area % of image, val+contam (p10 / p90)",
         f"{100 * np.percentile(areas, 10):.2f} / {100 * np.percentile(areas, 90):.2f}"),
    ]
    pd.DataFrame(rows, columns=["statistic", "value"]).to_csv(RESULTS / "eda_summary.csv", index=False)


def sample_figure(split):
    """One normal and one anomaly (non-test) per object for four objects, mask outlined in red."""
    objs = ["pcb1", "capsules", "cashew", "macaroni2"]
    fig, axes = plt.subplots(2, len(objs), figsize=(3.2 * len(objs), 7.2))
    for j, obj in enumerate(objs):
        cls = split[(split.object == obj) & split.split.isin(["train", "val"])]
        normal = cls[cls.label == "normal"].iloc[0]
        anom = cls[cls.label == "anomaly"].sort_values("stratum").iloc[0]
        for i, row in enumerate([normal, anom]):
            ax = axes[i, j]
            ax.imshow(Image.open(VISA_DIR / row.image).convert("RGB"))
            if row.label == "anomaly":
                ax.contour(np.asarray(Image.open(VISA_DIR / row["mask"]).convert("L")) > 0, colors="red", linewidths=1)
            ax.set_title(f"{obj} — {'normal' if i == 0 else row.stratum}", fontsize=9)
            ax.axis("off")
    fig.suptitle("VisA samples (train/val only); red contour = ground-truth defect mask")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_samples.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- §5 crop check
def crop_loss(split):
    """Share of defect pixels a resize-256 + center-crop-224 pipeline would cut away (val + contam masks)."""
    rows = []
    for r in split[split.split.isin(MASK_SPLITS) & (split.label == "anomaly")].itertuples():
        mask = np.asarray(Image.open(VISA_DIR / r.mask).convert("L")) > 0
        h, w = mask.shape
        s = 256 / min(h, w)  # torchvision Resize(256) scales the shorter side
        ch, cw = 224 / s, 224 / s  # crop window in original pixels
        top, left = (h - ch) / 2, (w - cw) / 2
        kept = mask[int(round(top)):int(round(top + ch)), int(round(left)):int(round(left + cw))].sum()
        rows.append({"object": r.object, "image": r.image, "lost_frac": 1 - kept / mask.sum()})
    df = pd.DataFrame(rows)
    per_obj = df.groupby("object").lost_frac.agg(
        mean_lost_pct=lambda x: 100 * x.mean(),
        imgs_lost_over_half_pct=lambda x: 100 * (x > 0.5).mean(),
        imgs_fully_lost_pct=lambda x: 100 * (x >= 0.999).mean())
    per_obj.loc["all"] = [100 * df.lost_frac.mean(), 100 * (df.lost_frac > 0.5).mean(),
                          100 * (df.lost_frac >= 0.999).mean()]
    per_obj.to_csv(RESULTS / "eda_crop_loss.csv")


# ---------------------------------------------------------------- §4 features
def global_embeddings(obj):
    """Spatially averaged layer2 and layer3 features for train and val rows (never test)."""
    frame, feats = load_features(obj, ("layer2", "layer3"), splits=FEATURE_SPLITS)
    assert set(frame.split) <= set(FEATURE_SPLITS)
    pooled = {l: np.asarray(f, dtype=np.float32).mean(axis=(2, 3)) for l, f in feats.items()}
    return frame, pooled


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))


def feature_analysis():
    sep, disp, red, proj = [], [], [], {}
    for obj in CLASSES:
        frame, pooled = global_embeddings(obj)
        emb = np.concatenate([pooled["layer2"], pooled["layer3"]], axis=1)
        tr = (frame.split == "train").to_numpy()
        vn = ((frame.split == "val") & (frame.label == "normal")).to_numpy()
        va = ((frame.split == "val") & (frame.label == "anomaly")).to_numpy()

        # separability: Mahalanobis distance to the train-normal Gaussian, val normal vs val anomaly
        lw = LedoitWolf().fit(emb[tr])
        d = np.sqrt(lw.mahalanobis(emb))
        scale = d[vn].std()
        pca = PCA(n_components=5).fit(emb[tr])
        z = pca.transform(emb)
        row = {"object": obj, "group": GROUP[obj], "n_val_normal": vn.sum(), "n_val_anomaly": va.sum(),
               "maha_KS": ks_2samp(d[vn], d[va]).statistic,
               "maha_KS_p": ks_2samp(d[vn], d[va]).pvalue,
               "maha_W1_std": wasserstein_distance(d[vn], d[va]) / scale,
               "maha_median_anom_over_normal": np.median(d[va]) / np.median(d[vn])}
        for k in range(3):
            row[f"PC{k + 1}_KS"] = ks_2samp(z[vn, k], z[va, k]).statistic
        sep.append(row)

        # dispersion of normal images: global spread, and spread at each spatial location (PaDiM's view)
        e = emb[tr] / np.linalg.norm(emb[tr], axis=1, keepdims=True)
        mu = e.mean(0) / np.linalg.norm(e.mean(0))
        _, feats = load_features(obj, ("layer3",), splits=["train"])
        l3 = np.asarray(feats["layer3"], dtype=np.float32)          # (N, C, 14, 14)
        per_loc_var = l3.var(axis=0).sum(axis=0).mean()             # variance across images at a fixed location
        pooled_var = l3.mean(axis=(2, 3)).var(axis=0).sum()          # variance of the image-level average
        disp.append({"object": obj, "group": GROUP[obj], "n_train": tr.sum(),
                     "mean_cosine_dist_to_mean": float((1 - e @ mu).mean()),
                     "layer3_per_location_var": float(per_loc_var),
                     "layer3_pooled_var": float(pooled_var),
                     "location_to_pooled_var_ratio": float(per_loc_var / pooled_var)})

        # redundancy: layer2 vs layer3 similarity and how compressible the concatenation is
        cum = np.cumsum(PCA().fit(emb[tr]).explained_variance_ratio_)
        red.append({"object": obj, "linear_CKA_layer2_layer3": linear_cka(pooled["layer2"][tr], pooled["layer3"][tr]),
                    "pcs_for_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
                    "pcs_for_95pct_var": int(np.searchsorted(cum, 0.95) + 1),
                    "dims": emb.shape[1]})

        keep = tr | va
        proj[obj] = {"pca": z[keep, :2], "is_anom": va[keep], "d_normal": d[vn], "d_anom": d[va], "emb": emb[keep]}

    pd.DataFrame(sep).to_csv(RESULTS / "eda_separability.csv", index=False)
    pd.DataFrame(disp).to_csv(RESULTS / "eda_dispersion.csv", index=False)
    pd.DataFrame(red).to_csv(RESULTS / "eda_redundancy.csv", index=False)
    return proj


AFFECTED_GAP_PP = 40  # plan, "EDA 發現的問題" 問題 1: fixed before any test result


def _modes(obj):
    """Two k-means modes of the train-normal embedding (2-D PCA space); returns frame, masks and in-minor flag."""
    frame, pooled = global_embeddings(obj)
    emb = np.concatenate([pooled["layer2"], pooled["layer3"]], axis=1)
    tr = (frame.split == "train").to_numpy()
    va = ((frame.split == "val") & (frame.label == "anomaly")).to_numpy()
    z = PCA(n_components=2).fit(emb[tr]).transform(emb)
    mode = KMeans(n_clusters=2, n_init=10, random_state=0).fit(z[tr]).predict(z)
    in_minor = mode == np.argmin(np.bincount(mode[tr]))
    return frame, tr, va, in_minor


def acquisition_modes():
    """Report how train normals and val anomalies fall into the smaller mode; a large gap flags a
    capture-condition shortcut. Classes past AFFECTED_GAP_PP are reported separately from §7 on."""
    rows = []
    for obj in CLASSES:
        _, tr, va, in_minor = _modes(obj)
        n_pct, a_pct = 100 * in_minor[tr].mean(), 100 * in_minor[va].mean()
        rows.append({"object": obj, "group": GROUP[obj],
                     "minor_mode_pct_of_train_normal": n_pct, "minor_mode_pct_of_val_anomaly": a_pct,
                     "gap_pp": a_pct - n_pct, "affected": a_pct - n_pct >= AFFECTED_GAP_PP})
    pd.DataFrame(rows).to_csv(RESULTS / "eda_acquisition_modes.csv", index=False)


def _gray_small(path):
    """Grayscale image downsampled to 64x48 via JPEG draft decoding (fast), as float32."""
    im = Image.open(VISA_DIR / path)
    im.draft("L", (im.width // 8, im.height // 8))
    return np.asarray(im.convert("L").resize((64, 48)), dtype=np.float32)


def acquisition_causes():
    """Explain the modes with two capture properties (train/val images only):
    brightness of the whole image and of a 5 % border (mostly background), and whether an image matches
    the train-normal mean template better after a 180-degree rotation."""
    rows = []
    for obj in CLASSES:
        frame, tr, va, in_minor = _modes(obj)
        imgs = np.stack([_gray_small(p) for p in frame.image])
        border = np.ones((48, 64), bool)
        border[3:-3, 3:-3] = False

        def zscore(x):
            x = x.reshape(len(x), -1)
            return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)

        template = zscore(imgs[tr].mean(0, keepdims=True))[0]
        flat, flat_rot = zscore(imgs), zscore(imgs[:, ::-1, ::-1])
        rotated = (flat_rot @ template) > (flat @ template)

        groups = {"normal_major": tr & ~in_minor, "normal_minor": tr & in_minor, "val_anomaly": va}
        row = {"object": obj, "group": GROUP[obj]}
        for g, m in groups.items():
            row[f"n_{g}"] = int(m.sum())
            row[f"gray_{g}"] = imgs[m].mean() if m.any() else np.nan
            row[f"border_gray_{g}"] = imgs[m][:, border].mean() if m.any() else np.nan
            row[f"rot180_pct_{g}"] = 100 * rotated[m].mean() if m.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "eda_acquisition_causes.csv", index=False)


def _grid(title, fname, draw):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, obj in zip(axes.flat, CLASSES):
        draw(ax, obj)
        ax.set_title(f"{obj} ({GROUP[obj]})", fontsize=9)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def projection_figures(proj):
    def scatter(ax, xy, is_anom, label):
        ax.scatter(*xy[~is_anom].T, s=4, c="0.6", label="train normal")
        ax.scatter(*xy[is_anom].T, s=10, c="tab:red", label="val anomaly")
        ax.set_xlabel(f"{label} 1")
        ax.set_ylabel(f"{label} 2")

    _grid("PCA of global WRN50 embedding (PCA fit on train normal)", "fig_pca.png",
          lambda ax, o: scatter(ax, proj[o]["pca"], proj[o]["is_anom"], "PC"))
    tsne = {o: TSNE(perplexity=30, init="pca", random_state=0).fit_transform(p["emb"]) for o, p in proj.items()}
    _grid("t-SNE of global WRN50 embedding (train normal vs val anomaly)", "fig_tsne.png",
          lambda ax, o: scatter(ax, tsne[o], proj[o]["is_anom"], "t-SNE"))

    def hist(ax, o):
        p = proj[o]
        bins = np.linspace(min(p["d_normal"].min(), p["d_anom"].min()), max(p["d_normal"].max(), p["d_anom"].max()), 30)
        ax.hist(p["d_normal"], bins=bins, alpha=0.6, density=True, color="0.5", label="val normal")
        ax.hist(p["d_anom"], bins=bins, alpha=0.6, density=True, color="tab:red", label="val anomaly")
        ax.set_xlabel("Mahalanobis distance")
        ax.set_ylabel("density")

    _grid("Global-embedding Mahalanobis distance to the train-normal Gaussian (unitless): "
          "val normal vs val anomaly", "fig_maha_hist.png", hist)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    split = load_split()
    split_tables(split)
    summary_table(split)
    sample_figure(split)
    crop_loss(split)
    projection_figures(feature_analysis())
    acquisition_modes()
    acquisition_causes()
    for f in sorted(RESULTS.glob("eda_*.csv")):
        print(f"\n== {f.name}")
        print(pd.read_csv(f).round(3).to_string(index=False))
