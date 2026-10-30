"""Fit the three tiers, pick tau on val (best F2), and evaluate once on test (§6, §7).

Per class: fit every (tier, seed) on train normals, choose tau on val, and only then load test and
score it with all fitted models. Writes <out>/metrics.csv and <out>/scores/<class>_<tier>_s<seed>.npz.

Hyperparameters are chosen on val only: --search fits every grid point (seed 0) and records val PR-AUC in
results/stage6_main/search_val.csv; the main run then uses, per tier, the setting with the best mean val
PR-AUC over the 12 classes (ties go to the earlier, smaller setting). Without that file the defaults are used.

Usage:  python src/run.py --quick                  # 1 class, seed 0 -> results/stage5_quick/
        python src/run.py --search                 # val-only grid search -> results/stage6_main/search_val.csv
        python src/run.py [--objs ...] [--seeds ...] # default: 12 classes x seeds 0 1 2 -> results/stage6_main/
        python src/run.py --latency pcb1           # per-image latency + environment.csv
        python src/run.py --ablation               # Tier 2 ablations -> results/stage7_ablation/
"""
import argparse
import ctypes
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from data import CLASSES, ROOT, VisADataset, load_split
from features import load_features
from methods import DEVICE, PaDiM, PatchCore, TemplateDiff

PIS = (0.01, 0.05)  # deployment prevalences for prec_pi
L23, L234 = ("layer2", "layer3"), ("layer2", "layer3", "layer4")
GRID = {"T2_padim": (PaDiM, [{"n_dims": d, "layers": l} for l in (L23, L234) for d in (50, 100, 200)]),
        "T3_patchcore": (PatchCore, [{"coreset": c, "layers": l} for l in (L23, L234) for c in (0.01, 0.10)])}
CONFIG = {"T2_padim": {"n_dims": 100, "layers": L23}, "T3_patchcore": {"coreset": 0.01, "layers": L23}}
SEARCH_PATH = ROOT / "results" / "stage6_main" / "search_val.csv"
TIERS = {"T1_template": lambda seed: TemplateDiff(),
         "T2_padim": lambda seed: PaDiM(eps=0.01, seed=seed, **CONFIG["T2_padim"]),
         "T3_patchcore": lambda seed: PatchCore(proj_dim=128, seed=seed, **CONFIG["T3_patchcore"])}


# ---------------------------------------------------------------- metrics
def best_f2_tau(y, s, beta=2):
    """Threshold on val maximising F_beta (default F2 = 5PR / (4P + R)); predictions are s >= tau."""
    p, r, thr = precision_recall_curve(y, s)
    f = (1 + beta**2) * p[:-1] * r[:-1] / np.maximum(beta**2 * p[:-1] + r[:-1], 1e-12)
    return float(thr[np.argmax(f)])


def metrics(y, s, tau):
    pred = s >= tau
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    prec = tp / max(tp + fp, 1)
    tpr, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    out = {"auroc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "tau": tau,
           "precision": prec, "recall": tpr, "fpr": fpr, "acc": (tp + tn) / len(y),
           "f1": 2 * prec * tpr / max(prec + tpr, 1e-12), "f2": 5 * prec * tpr / max(4 * prec + tpr, 1e-12)}
    for pi in PIS:
        den = pi * tpr + (1 - pi) * fpr
        out[f"prec_pi{pi:.0%}"] = pi * tpr / den if den > 0 else np.nan
    return out


def _check_metrics():
    """Hand-computed example: y = 0 0 1 1, s = .1 .4 .35 .8, tau = .35 -> TP 2, FP 1, FN 0, TN 1."""
    y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    m = metrics(y, s, 0.35)
    expected = {"auroc": 0.75, "pr_auc": 0.5 * 1 + 0.5 * 2 / 3, "precision": 2 / 3, "recall": 1.0, "fpr": 0.5,
                "acc": 0.75, "f1": 0.8, "f2": 10 / 11, "prec_pi1%": 0.01 / (0.01 + 0.99 * 0.5)}
    for k, v in expected.items():
        assert abs(m[k] - v) < 1e-9, (k, m[k], v)
    assert best_f2_tau(y, s) == 0.35  # tau = .35 catches both anomalies at 1 FP, the best F2 here


# ---------------------------------------------------------------- inputs
def load_images(frame):
    loader = DataLoader(VisADataset(frame, normalize=False), batch_size=64, num_workers=4)
    return torch.cat([x for x, _, _, _ in loader])


def inputs(obj, splits):
    """(frame, features dict, images) for some splits of one class; images only feed Tier 1."""
    frame, feats = load_features(obj, L234, splits=splits)
    return frame, feats, load_images(frame)


def peak_ram_gb():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
    c = Counters(cb=ctypes.sizeof(Counters))
    current_process = ctypes.c_ssize_t(-1)  # pseudo-handle; must be pointer-sized on 64-bit Windows
    ctypes.windll.psapi.GetProcessMemoryInfo(current_process, ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**30


# ---------------------------------------------------------------- experiment
def search(objs):
    """Val-only grid search (seed 0); never loads test."""
    rows = []
    for obj in objs:
        _, train = load_features(obj, L234, splits=["train"])
        frame, val = load_features(obj, L234, splits=["val"])
        y = (frame.label == "anomaly").to_numpy().astype(int)
        for tier, (cls, grid) in GRID.items():
            for i, cfg in enumerate(grid):
                t = time.time()
                s, _ = cls(seed=0, **cfg).fit(train).score(val)
                rows.append({"object": obj, "tier": tier, "config_id": i, "config": repr(cfg),
                             "val_pr_auc": average_precision_score(y, s), "val_auroc": roc_auc_score(y, s),
                             "fit_score_s": time.time() - t})
        print(f"{obj}: searched", flush=True)
        pd.DataFrame(rows).to_csv(SEARCH_PATH, index=False)  # rewritten per class so a crash keeps progress
    return pd.DataFrame(rows)


def choose_configs():
    """Best mean val PR-AUC per tier (ties -> lower config_id); updates CONFIG in place."""
    df = pd.read_csv(SEARCH_PATH)
    assert df.object.nunique() == len(CLASSES), "search must cover all 12 classes"
    mean = df.groupby(["tier", "config_id"]).val_pr_auc.mean().round(4).reset_index()
    for tier, g in mean.groupby("tier"):
        best = int(g.sort_values(["val_pr_auc", "config_id"], ascending=[False, True]).config_id.iloc[0])
        CONFIG[tier] = GRID[tier][1][best]
    return mean


def run_class(obj, seeds, out):
    train = inputs(obj, ["train"])
    assert (train[0].label == "normal").all()
    val = inputs(obj, ["val"])
    y_val = (val[0].label == "anomaly").to_numpy().astype(int)

    fitted = []
    for tier, make in TIERS.items():
        for seed in (seeds if tier != "T1_template" else seeds[:1]):  # Tier 1 is deterministic
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            model = make(seed).fit(train[2] if tier == "T1_template" else train[1])
            fit_s = time.time() - t
            s_val, _ = model.score(val[2] if tier == "T1_template" else val[1])
            fitted.append({"tier": tier, "seed": seed, "model": model, "fit_s": fit_s,
                           "tau": best_f2_tau(y_val, s_val),
                           "fit_vram_gb": torch.cuda.max_memory_allocated() / 2**30})
    del train

    # test is read here, once, after every model and tau is fixed
    test = inputs(obj, ["test"])
    y = (test[0].label == "anomaly").to_numpy().astype(int)
    rows = [{"object": obj, "tier": "T0_all_normal", "seed": 0, "acc": (y == 0).mean(), "recall": 0.0,
             "f1": 0.0, "f2": 0.0, "fpr": 0.0, "pr_auc": y.mean(), "auroc": 0.5}]
    (out / "scores").mkdir(parents=True, exist_ok=True)
    for f in fitted:
        t = time.time()
        s, _ = f["model"].score(test[2] if f["tier"] == "T1_template" else test[1])
        ms = 1000 * (time.time() - t) / len(y)
        rows.append({"object": obj, "tier": f["tier"], "seed": f["seed"], "config": repr(CONFIG.get(f["tier"], {})),
                     **metrics(y, s, f["tau"]),
                     "fit_s": f["fit_s"], "score_ms_per_image_cached": ms, "fit_vram_gb": f["fit_vram_gb"]})
        np.savez(out / "scores" / f"{obj}_{f['tier']}_s{f['seed']}.npz",
                 image=test[0].image.to_numpy(), label=y, score=s, tau=f["tau"])
    return rows


# ---------------------------------------------------------------- ablations (§7), Tier 2 only
# Each ablation changes one factor of the full Tier 2 method (CONFIG, best-F2 tau, clean train set).
# "fit" changes the fitted model, "score" only how its map is turned into s(x), "tau" only the threshold rule.
ABLATIONS = [
    ("A1_scoring", "Mahalanobis (full)", {}, {}, "best-F2"),
    ("A1_scoring", "Euclidean 1-NN", {"metric": "euclid"}, {}, "best-F2"),
    ("A1_scoring", "cosine 1-NN", {"metric": "cosine"}, {}, "best-F2"),
    ("A2_threshold", "best-F2 (full)", {}, {}, "best-F2"),
    ("A2_threshold", "best-F1", {}, {}, "best-F1"),
    ("A2_threshold", "val-normal 99th percentile", {}, {}, "p99"),
    ("A2_threshold", "review budget: top 10% of val", {}, {}, "budget10"),
    ("A3_contamination", "0% (full)", {}, {}, "best-F2"),
    ("A3_contamination", "1%", {"contam": 0.01}, {}, "best-F2"),
    ("A3_contamination", "2%", {"contam": 0.02}, {}, "best-F2"),
    ("A4_aggregation", "max (full)", {}, {}, "best-F2"),
    ("A4_aggregation", "top-1% mean", {}, {"agg": "top1pct"}, "best-F2"),
    ("A4_aggregation", "max of smoothed map (sigma=4)", {}, {"sigma": 4}, "best-F2"),
    ("A5_features", "patch-level (full)", {}, {}, "best-F2"),
    ("A5_features", "global pooled", {"pooled": True}, {}, "best-F2"),
]


def tau_rule(rule, y, s):
    if rule == "best-F2":
        return best_f2_tau(y, s)
    if rule == "best-F1":
        return best_f2_tau(y, s, beta=1)
    if rule == "p99":
        return float(np.percentile(s[y == 0], 99))
    return float(np.quantile(s, 0.90))  # budget10: inspect the 10 % highest-scoring val images


def ablation_class(obj, seeds):
    """Fit every distinct model once per seed on train (+ contam pool anomalies for A3), fix tau on val,
    then read test once."""
    _, train = load_features(obj, L234, splits=["train"])
    fv, val = load_features(obj, L234, splits=["val"])
    _, pool = load_features(obj, L234, splits=["contam"])
    y_val = (fv.label == "anomaly").to_numpy().astype(int)
    n_train = len(train["layer2"])

    fitted, plans = {}, []
    for seed in seeds:
        for abl, setting, fit_kw, score_kw, rule in ABLATIONS:
            key = (seed, repr(sorted(fit_kw.items())))
            if key not in fitted:
                kw = dict(fit_kw)
                rate = kw.pop("contam", 0.0)
                data = train
                if rate:  # the extra anomalies come from the contam pool only, never from val or test
                    k = round(rate * n_train)
                    pick = np.sort(np.random.default_rng(seed).choice(len(pool["layer2"]), k, replace=False))
                    data = {l: np.concatenate([train[l], pool[l][pick]]) for l in L234}
                fitted[key] = (PaDiM(eps=0.01, seed=seed, **CONFIG["T2_padim"], **kw).fit(data),
                               round(rate * n_train))
            model, n_contam = fitted[key]
            model.agg, model.sigma = score_kw.get("agg", "max"), score_kw.get("sigma")
            s_val, _ = model.score(val)
            plans.append({"ablation": abl, "setting": setting, "seed": seed, "key": key, "score_kw": score_kw,
                          "tau": tau_rule(rule, y_val, s_val), "n_contam": n_contam})

    test_frame, test = load_features(obj, L234, splits=["test"])  # read once, after every tau is fixed
    y = (test_frame.label == "anomaly").to_numpy().astype(int)
    rows, cache = [], {}
    for p in plans:
        ck = (p["key"], repr(p["score_kw"]))
        if ck not in cache:
            model = fitted[p["key"]][0]
            model.agg, model.sigma = p["score_kw"].get("agg", "max"), p["score_kw"].get("sigma")
            cache[ck] = model.score(test)[0]
        rows.append({"object": obj, "ablation": p["ablation"], "setting": p["setting"], "seed": p["seed"],
                     "n_contam": p["n_contam"], **metrics(y, cache[ck], p["tau"])})
    return rows


def summarize_ablations(df, out):
    """Class-average per seed, mean / std over seeds, and delta vs the full setting of the same ablation.
    `beyond_seed_std` marks |delta| larger than the seed std of both the full method and the setting."""
    modes = pd.read_csv(ROOT / "results" / "stage4_eda" / "eda_acquisition_modes.csv")
    affected = modes.loc[modes.affected, "object"].tolist()
    cols = ["pr_auc", "auroc", "f1", "f2", "fpr", "recall", "prec_pi1%"]
    rows = []
    for subset, part in [("12 classes", df), ("9 classes", df[~df.object.isin(affected)])]:
        per_seed = part.groupby(["ablation", "setting", "seed"], sort=False)[cols].mean()
        stats = per_seed.groupby(["ablation", "setting"], sort=False).agg(["mean", "std"])
        for (abl, setting), r in stats.iterrows():
            full = stats.loc[abl].iloc[0]  # the first setting of each ablation is the full method
            row = {"subset": subset, "ablation": abl, "setting": setting}
            for c in cols:
                row[f"{c}_mean"], row[f"{c}_std"] = r[(c, "mean")], r[(c, "std")]
                row[f"{c}_delta"] = r[(c, "mean")] - full[(c, "mean")]
                row[f"{c}_beyond_seed_std"] = abs(row[f"{c}_delta"]) > max(r[(c, "std")], full[(c, "std")])
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------- summary, latency, environment
METRIC_COLS = ["auroc", "pr_auc", "f1", "f2", "fpr", "prec_pi1%", "prec_pi5%"]


def summarize(df, out):
    """Per tier: class-average per seed, then mean / std over seeds; for all 12 classes and for the 9 classes
    not flagged by the EDA acquisition check (list read from stage 4, fixed before any test result)."""
    modes = pd.read_csv(ROOT / "results" / "stage4_eda" / "eda_acquisition_modes.csv")
    affected = modes.loc[modes.affected, "object"].tolist()
    rows = []
    for subset, part in [("12 classes", df), (f"9 classes (excl. {', '.join(affected)})",
                                              df[~df.object.isin(affected)])]:
        per_seed = part.groupby(["tier", "seed"])[METRIC_COLS].mean()
        for tier, g in per_seed.groupby("tier"):
            row = {"subset": subset, "tier": tier, "n_seeds": len(g)}
            for c in METRIC_COLS:
                row[f"{c}_mean"], row[f"{c}_std"] = g[c].mean(), (g[c].std(ddof=1) if len(g) > 1 else 0.0)
            fit = part[part.tier == tier].groupby("seed").fit_s.sum()  # all classes of one seed
            row["fit_s_total_mean"] = fit.mean()
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    for subset, g in summary.groupby("subset", sort=False):  # does the tier ranking change without the 3 classes?
        print(subset, "ranking by PR-AUC:", " > ".join(g.sort_values("pr_auc_mean", ascending=False).tier))
    return summary


def latency(obj, out, n=50, warmup=10):
    """Per-image latency on GPU, batch 1, after `warmup` images; features are extracted inside the timed region
    (images are decoded and resized beforehand). Uses val images, so test stays untouched."""
    from features import build_backbone, extract
    backbone = build_backbone(DEVICE)
    train = inputs(obj, ["train"])
    frame = load_split(obj, ["val"]).iloc[:warmup + n]
    norm = [VisADataset(frame)[i][0][None] for i in range(len(frame))]
    raw = [VisADataset(frame, normalize=False)[i][0][None] for i in range(len(frame))]
    rows = []
    for tier, make in TIERS.items():
        model = make(0).fit(train[2] if tier == "T1_template" else train[1])
        ms = []
        for i in range(len(frame)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if tier == "T1_template":
                model.score(raw[i])
            else:
                f = extract(backbone, norm[i].to(DEVICE))
                model.score({l: f[l].half().cpu().numpy() for l in L234})
            torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t))
        ms = np.array(ms[warmup:])
        rows.append({"object": obj, "tier": tier, "config": repr(CONFIG.get(tier, {})), "n": len(ms),
                     "ms_median": np.median(ms), "ms_mean": ms.mean(), "ms_p95": np.percentile(ms, 95)})
    pd.DataFrame(rows).to_csv(out / "latency.csv", index=False)
    return pd.DataFrame(rows)


def environment(out):
    import platform
    import torchvision
    rows = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
            "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "cpu": platform.processor(), "os": platform.platform()}
    pd.Series(rows).to_csv(out / "environment.csv", header=["value"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one class, seed 0, results/stage5_quick")
    ap.add_argument("--objs", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--padim-sigma", type=int, default=None,
                    help="smooth the Tier 2 map (upsample to 224 + Gaussian blur); results go to <out>_smooth")
    ap.add_argument("--search", action="store_true", help="val-only hyperparameter search, then exit")
    ap.add_argument("--latency", metavar="CLASS", help="per-image latency on one class's val images, then exit")
    ap.add_argument("--ablation", action="store_true", help="Tier 2 ablations -> results/stage7_ablation/, then exit")
    args = ap.parse_args()
    _check_metrics()
    if args.search:
        SEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        search(args.objs)
        print(choose_configs().to_string(index=False))
        print("chosen:", CONFIG)
        sys.exit()
    if not args.quick and SEARCH_PATH.exists():
        choose_configs()
    print("configs:", CONFIG, flush=True)
    if args.latency:
        environment(SEARCH_PATH.parent)
        print(latency(args.latency, SEARCH_PATH.parent).round(2).to_string(index=False))
        sys.exit()
    if args.ablation:
        out = ROOT / "results" / "stage7_ablation"
        out.mkdir(parents=True, exist_ok=True)
        rows, t0 = [], time.time()
        for obj in args.objs:
            rows += ablation_class(obj, args.seeds)
            pd.DataFrame(rows).to_csv(out / "ablation_metrics.csv", index=False)
            print(f"{obj}: done at {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB", flush=True)
        if len(args.objs) == len(CLASSES):
            s = summarize_ablations(pd.DataFrame(rows), out)
            show = ["subset", "ablation", "setting", "pr_auc_mean", "pr_auc_std", "pr_auc_delta",
                    "pr_auc_beyond_seed_std", "f2_mean", "fpr_mean", "prec_pi1%_mean"]
            print(s[show].round(3).to_string(index=False))
        sys.exit()
    objs, seeds = (["candle"], [0]) if args.quick else (args.objs, args.seeds)
    out = ROOT / "results" / ("stage5_quick" if args.quick else "stage6_main")
    if args.padim_sigma:
        TIERS["T2_padim"] = lambda seed: PaDiM(n_dims=100, eps=0.01, seed=seed, sigma=args.padim_sigma)
        out = out.with_name(out.name + "_smooth")
    out.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for obj in objs:
        rows += run_class(obj, seeds, out)
        print(f"{obj}: done at {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    cols = ["object", "tier", "seed", "auroc", "pr_auc", "f1", "f2", "fpr", "prec_pi1%", "fit_s",
            "score_ms_per_image_cached", "fit_vram_gb"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nwall {time.time() - t0:.0f}s, peak RAM {peak_ram_gb():.1f} GB, "
          f"peak VRAM (fit + val scoring) {df.fit_vram_gb.max():.2f} GB")
    if df.object.nunique() == len(CLASSES):
        summarize(df, out)
