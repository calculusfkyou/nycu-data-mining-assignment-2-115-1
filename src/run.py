"""Fit the three tiers, pick tau on val (best F2), and evaluate once on test (§6, §7).

Per class: fit every (tier, seed) on train normals, choose tau on val, and only then load test and
score it with all fitted models. Writes <out>/metrics.csv and <out>/scores/<class>_<tier>_s<seed>.npz.

Usage:  python src/run.py --quick                  # 1 class, seed 0 -> results/stage5_quick/
        python src/run.py [--objs ...] [--seeds ...] # default: 12 classes x seeds 0 1 2 -> results/stage6_main/
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
from methods import PaDiM, PatchCore, TemplateDiff

PIS = (0.01, 0.05)  # deployment prevalences for prec_pi
TIERS = {"T1_template": lambda seed: TemplateDiff(),
         "T2_padim": lambda seed: PaDiM(n_dims=100, eps=0.01, seed=seed),
         "T3_patchcore": lambda seed: PatchCore(coreset=0.01, proj_dim=128, seed=seed)}


# ---------------------------------------------------------------- metrics
def best_f2_tau(y, s):
    """Threshold on val maximising F2 = 5PR / (4P + R); predictions are s >= tau."""
    p, r, thr = precision_recall_curve(y, s)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return float(thr[np.argmax(f2)])


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
    frame, feats = load_features(obj, ("layer2", "layer3"), splits=splits)
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
        rows.append({"object": obj, "tier": f["tier"], "seed": f["seed"], **metrics(y, s, f["tau"]),
                     "fit_s": f["fit_s"], "score_ms_per_image_cached": ms, "fit_vram_gb": f["fit_vram_gb"]})
        np.savez(out / "scores" / f"{obj}_{f['tier']}_s{f['seed']}.npz",
                 image=test[0].image.to_numpy(), label=y, score=s, tau=f["tau"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one class, seed 0, results/stage5_quick")
    ap.add_argument("--objs", nargs="*", default=CLASSES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--padim-sigma", type=int, default=None,
                    help="smooth the Tier 2 map (upsample to 224 + Gaussian blur); results go to <out>_smooth")
    args = ap.parse_args()
    _check_metrics()
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
