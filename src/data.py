"""VisA download, the fixed train/val/contam/test split (§3), and a torch Dataset.

Usage:  python src/data.py          # download + extract (if missing), build split, print counts
"""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VISA_DIR = DATA_DIR / "VisA"
TAR_PATH = DATA_DIR / "VisA_20220922.tar"
SPLIT_PATH = ROOT / "splits" / "visa_split.csv"
URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"

CLASSES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
           "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
SEED = 0
VAL_NORMAL_FRAC = 0.11
N_VAL_ANOM, N_CONTAM_ANOM, N_TEST_ANOM = 20, 20, 60
MIN_STRATUM = 3  # defect types rarer than this are pooled into "other"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def download():
    if not TAR_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, TAR_PATH)
    if not (VISA_DIR / "split_csv" / "1cls.csv").exists():
        print(f"extracting to {VISA_DIR} ...")
        VISA_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TAR_PATH) as tar:
            tar.extractall(VISA_DIR)


def _defect_types():
    """image path -> defect stratum, taken from the first label in each class's image_anno.csv."""
    anno = pd.concat([pd.read_csv(VISA_DIR / c / "image_anno.csv") for c in CLASSES])
    anno = anno[anno.label != "normal"]
    return dict(zip(anno.image, anno.label.str.split(",").str[0].str.strip()))


def _split_anomalies(df, rng):
    """Stratified 20/20/60 split of one class's 100 anomalies.

    Rows are ordered by (stratum, random) and assigned with a repeating
    test,test,test,val,contam pattern, so every defect type keeps roughly
    the same 3:1:1 ratio while the totals are exactly 60/20/20.
    """
    counts = df.stratum.value_counts()
    df = df.assign(stratum=df.stratum.where(df.stratum.map(counts) >= MIN_STRATUM, "other"))
    df = df.assign(_r=rng.random(len(df))).sort_values(["stratum", "_r"]).drop(columns="_r")
    pattern = ["test"] * 3 + ["val", "contam"]
    df["split"] = [pattern[i % 5] for i in range(len(df))]
    return df


def build_split():
    rng = np.random.default_rng(SEED)
    official = pd.read_csv(VISA_DIR / "split_csv" / "1cls.csv")
    types = _defect_types()
    rows = []
    for c in CLASSES:
        cls = official[official.object == c]

        train_n = cls[(cls.split == "train") & (cls.label == "normal")].copy()
        val_idx = rng.choice(len(train_n), size=round(VAL_NORMAL_FRAC * len(train_n)), replace=False)
        train_n["split"] = "train"
        train_n.iloc[val_idx, train_n.columns.get_loc("split")] = "val"
        train_n["stratum"] = "normal"

        test_n = cls[(cls.split == "test") & (cls.label == "normal")].assign(split="test", stratum="normal")

        anom = cls[cls.label == "anomaly"].copy()
        anom["stratum"] = anom.image.map(types).fillna("other")
        anom = _split_anomalies(anom, rng)

        rows += [train_n, test_n, anom]

    split = pd.concat(rows)[["object", "split", "label", "stratum", "image", "mask"]]
    split = split.sort_values(["object", "split", "label", "image"]).reset_index(drop=True)
    SPLIT_PATH.parent.mkdir(exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)
    return split


def check_split(split):
    assert len(split) == 10821 and split.image.is_unique, "every image must appear exactly once"
    assert (split[split.split == "train"].label == "normal").all(), "train must be normal-only"
    for c, g in split.groupby("object"):
        n = g.groupby(["split", "label"]).size()
        assert n.get(("val", "anomaly")) == N_VAL_ANOM, c
        assert n.get(("contam", "anomaly")) == N_CONTAM_ANOM, c
        assert n.get(("test", "anomaly")) == N_TEST_ANOM, c
    ids = {s: set(g.image) for s, g in split.groupby("split")}
    for a in ids:
        for b in ids:
            assert a == b or not (ids[a] & ids[b]), f"{a} and {b} overlap"


def load_split(obj=None, splits=None):
    split = pd.read_csv(SPLIT_PATH)
    if obj is not None:
        split = split[split.object == obj]
    if splits is not None:
        split = split[split.split.isin(splits)]
    return split.reset_index(drop=True)


class VisADataset(Dataset):
    """Images resized straight to 224x224 (no center crop, see §5); mask returned at the same size."""

    def __init__(self, frame, normalize=True):
        self.frame = frame.reset_index(drop=True)
        tf = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
        if normalize:
            tf.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.tf = transforms.Compose(tf)
        self.mask_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = self.tf(Image.open(VISA_DIR / row.image).convert("RGB"))
        if isinstance(row["mask"], str):
            mask = (self.mask_tf(Image.open(VISA_DIR / row["mask"]).convert("L")) > 0).float()
        else:
            mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        return img, int(row.label == "anomaly"), mask, row.image


if __name__ == "__main__":
    download()
    split = build_split()
    check_split(split)
    table = split.groupby(["object", "split", "label"]).size().unstack(["split", "label"]).fillna(0).astype(int)
    print(table.to_string())
    print("\ntotal per split:")
    print(split.groupby(["split", "label"]).size().to_string())
    print(f"\nwrote {SPLIT_PATH.relative_to(ROOT)} ({len(split)} rows); all split checks passed")
