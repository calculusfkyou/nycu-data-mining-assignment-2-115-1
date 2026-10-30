"""WideResNet50 feature cache (§5): one float16 .npy per class and layer, at the layer's native resolution.

layer2 -> (N, 512, 28, 28), layer3 -> (N, 1024, 14, 14), layer4 -> (N, 2048, 7, 7).
Row order follows cache/<class>_index.csv, which is the class's rows of splits/visa_split.csv.
Upsampling and concatenation happen in methods.py, so the cache stays ~15 GB instead of ~52 GB.

Usage:  python src/features.py [class ...]      # default: all 12 classes, skipping cached ones
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from data import CLASSES, ROOT, VisADataset, load_split

CACHE_DIR = ROOT / "cache"
LAYERS = ("layer2", "layer3", "layer4")
CHANNELS = {"layer2": 512, "layer3": 1024, "layer4": 2048}
SIDE = {"layer2": 28, "layer3": 14, "layer4": 7}
BATCH = 32


def build_backbone(device):
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract(model, x):
    """Forward through the stem and layer1-4, returning the three cached layers."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer1(x)
    l2 = model.layer2(x)
    l3 = model.layer3(l2)
    l4 = model.layer4(l3)
    return {"layer2": l2, "layer3": l3, "layer4": l4}


def _path(obj, layer):
    return CACHE_DIR / f"{obj}_{layer}.npy"


def is_cached(obj):
    return (CACHE_DIR / f"{obj}_index.csv").exists() and all(_path(obj, l).exists() for l in LAYERS)


def cache_class(model, obj, device):
    frame = load_split(obj)
    loader = DataLoader(VisADataset(frame), batch_size=BATCH, num_workers=4, pin_memory=True)
    out = {l: np.lib.format.open_memmap(_path(obj, l), mode="w+", dtype=np.float16,
                                        shape=(len(frame), CHANNELS[l], SIDE[l], SIDE[l]))
           for l in LAYERS}
    i = 0
    for x, _, _, _ in loader:
        feats = extract(model, x.to(device, non_blocking=True))
        for l in LAYERS:
            out[l][i:i + len(x)] = feats[l].half().cpu().numpy()
        i += len(x)
    for m in out.values():
        m.flush()
    # written last, so an interrupted run is redone instead of being treated as cached
    frame.to_csv(CACHE_DIR / f"{obj}_index.csv", index=False)


def load_features(obj, layers=("layer2", "layer3"), splits=None):
    """Return (index frame, {layer: float16 memmap}) for one class, optionally filtered to some splits."""
    frame = pd.read_csv(CACHE_DIR / f"{obj}_index.csv")
    feats = {l: np.load(_path(obj, l), mmap_mode="r") for l in layers}
    if splits is not None:
        keep = frame.split.isin(splits).to_numpy()
        frame = frame[keep].reset_index(drop=True)
        feats = {l: f[keep] for l, f in feats.items()}
    return frame, feats


if __name__ == "__main__":
    targets = sys.argv[1:] or CLASSES
    CACHE_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone(device)
    for obj in targets:
        if is_cached(obj):
            print(f"{obj}: cached, skipping")
            continue
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        cache_class(model, obj, device)
        print(f"{obj}: {len(load_split(obj))} images in {time.time() - t:.0f}s, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
