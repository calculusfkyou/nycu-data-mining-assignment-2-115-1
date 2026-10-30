"""The three tiers (§5). Every method has fit(train) and score(x) -> (image_score (N,), heatmap (N, h, w)).

Tier 1  TemplateDiff  inputs: images (N, 3, 224, 224) in [0, 1]
Tier 2  PaDiM         inputs: {"layer2": (N, 512, 28, 28), "layer3": (N, 1024, 14, 14)} (float16 memmaps are fine)
Tier 3  PatchCore     same inputs as Tier 2

Features are moved to the GPU in chunks of CHUNK images, so a class never has to fit in VRAM at once.

Usage:  python src/methods.py        # synthetic self-check: all three tiers must reach AUROC > 0.9
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIDE = 28   # layer2 resolution; layer3 is upsampled to it
CHUNK = 64


def _chunks(n, size=CHUNK):
    return (slice(i, min(i + size, n)) for i in range(0, n, size))


def _to_gpu(a, sl, idx=None):
    a = a[sl] if idx is None else a[sl][:, idx]
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(DEVICE)


def _up(x):
    return F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False)


class TemplateDiff:
    """Tier 1: mean normal image as template; s(x) = mean |x - template|, heatmap = channel-mean |x - template|."""

    def fit(self, images):
        self.template = images.mean(0, keepdim=True)
        return self

    def score(self, images):
        diff = (images - self.template).abs().mean(1)
        return diff.mean((1, 2)).numpy(), diff.numpy()


class PaDiM:
    """Tier 2: per-location Gaussian on a random subset of layer2+layer3 channels.

    s_ij(x) = sqrt((f_ij - mu_ij)^T (Sigma_ij + eps I)^-1 (f_ij - mu_ij)),  s(x) = max_ij s_ij(x).
    With sigma set, the 28x28 map is first upsampled to 224x224 and Gaussian-blurred as in the PaDiM paper,
    and s(x) is the max of that smoothed map.
    """

    def __init__(self, n_dims=100, eps=0.01, seed=0, sigma=None):
        self.n_dims, self.eps, self.seed, self.sigma = n_dims, eps, seed, sigma

    def _embed(self, feats, sl):
        l2 = _to_gpu(feats["layer2"], sl, self.idx2)
        l3 = _up(_to_gpu(feats["layer3"], sl, self.idx3))
        return torch.cat([l2, l3], 1).flatten(2)                     # (B, d, 784)

    def fit(self, feats):
        c2, c3 = feats["layer2"].shape[1], feats["layer3"].shape[1]
        dims = np.sort(np.random.default_rng(self.seed).choice(c2 + c3, self.n_dims, replace=False))
        self.idx2, self.idx3 = dims[dims < c2], dims[dims >= c2] - c2
        n = len(feats["layer2"])
        x = torch.cat([self._embed(feats, sl) for sl in _chunks(n)])  # (N, d, 784)
        self.mu = x.mean(0)
        xc = (x - self.mu).permute(2, 0, 1)                             # (784, N, d)
        cov = xc.transpose(1, 2) @ xc / (n - 1)
        cov += self.eps * torch.eye(self.n_dims, device=DEVICE)
        self.prec = torch.linalg.inv(cov)                              # (784, d, d)
        return self

    def score(self, feats):
        maps = []
        for sl in _chunks(len(feats["layer2"])):
            d = (self._embed(feats, sl) - self.mu).permute(2, 0, 1)      # (784, B, d)
            m = ((d @ self.prec) * d).sum(-1).clamp_min(0).sqrt()         # (784, B)
            m = m.T.reshape(-1, 1, SIDE, SIDE)
            if self.sigma:
                m = F.interpolate(m, size=(224, 224), mode="bilinear", align_corners=False)
                m = gaussian_blur(m, kernel_size=8 * self.sigma + 1, sigma=self.sigma)  # 4-sigma truncation
            maps.append(m[:, 0].cpu())
        maps = torch.cat(maps)
        return maps.flatten(1).max(1).values.numpy(), maps.numpy()


class PatchCore:
    """Tier 3: locally aware patch features (3x3 average), greedy coreset memory bank, nearest-neighbour distance.

    The coreset is selected in a 128-d random projection; the bank keeps the full 1536-d features.
    """

    def __init__(self, coreset=0.01, proj_dim=128, seed=0):
        self.coreset, self.proj_dim, self.seed = coreset, proj_dim, seed

    def _embed(self, feats, sl):
        l2 = F.avg_pool2d(_to_gpu(feats["layer2"], sl), 3, 1, 1)
        l3 = _up(F.avg_pool2d(_to_gpu(feats["layer3"], sl), 3, 1, 1))
        x = torch.cat([l2, l3], 1)                                      # (B, 1536, 28, 28)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])            # (B*784, 1536)

    def fit(self, feats):
        g = torch.Generator(device=DEVICE).manual_seed(self.seed)
        n = len(feats["layer2"])
        bank, proj = [], None
        for sl in _chunks(n):
            x = self._embed(feats, sl)
            if proj is None:
                proj = torch.randn(x.shape[1], self.proj_dim, device=DEVICE, generator=g) / self.proj_dim ** 0.5
            bank.append((x.half().cpu(), x @ proj))
        full = torch.cat([b[0] for b in bank])                          # CPU float16
        z = torch.cat([b[1] for b in bank])                             # GPU float32, projected
        del bank
        k = max(1, int(self.coreset * len(z)))
        sel = [int(torch.randint(len(z), (1,), device=DEVICE, generator=g))]
        min_d = torch.cdist(z, z[sel[-1]][None]).squeeze(1)
        for _ in range(k - 1):
            sel.append(int(min_d.argmax()))
            min_d = torch.minimum(min_d, torch.cdist(z, z[sel[-1]][None]).squeeze(1))
        self.bank = full[torch.tensor(sel)].float().to(DEVICE)          # (k, 1536)
        return self

    def score(self, feats):
        maps = []
        for sl in _chunks(len(feats["layer2"]), 16):
            d = torch.cdist(self._embed(feats, sl), self.bank).min(1).values
            maps.append(d.reshape(-1, SIDE, SIDE).cpu())
        maps = torch.cat(maps)
        return maps.flatten(1).max(1).values.numpy(), maps.numpy()


def _self_check():
    """Normal = Gaussian noise; anomaly = the same plus a bright 4x4 (feature) / 32x32 (image) square."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)

    def feats(n, anomalous):
        f = {"layer2": rng.normal(size=(n, 64, 28, 28)).astype(np.float16),
             "layer3": rng.normal(size=(n, 64, 14, 14)).astype(np.float16)}
        if anomalous:
            for i in range(n):
                r, c = rng.integers(0, 12, 2)
                f["layer2"][i, :, 2 * r:2 * r + 4, 2 * c:2 * c + 4] += 3
                f["layer3"][i, :, r:r + 2, c:c + 2] += 3
        return f

    def images(n, anomalous):
        x = torch.from_numpy(rng.normal(0.5, 0.1, size=(n, 3, 224, 224)).astype(np.float32)).clamp(0, 1)
        if anomalous:
            for i in range(n):
                r, c = rng.integers(0, 192, 2)
                x[i, :, r:r + 32, c:c + 32] = 1.0
        return x

    y = np.r_[np.zeros(50), np.ones(50)]
    cat = lambda a, b: {k: np.concatenate([a[k], b[k]]) for k in a}
    test_f, test_i = cat(feats(50, False), feats(50, True)), torch.cat([images(50, False), images(50, True)])
    results = {
        "T1 TemplateDiff": TemplateDiff().fit(images(200, False)).score(test_i),
        "T2 PaDiM": PaDiM(n_dims=32).fit(feats(200, False)).score(test_f),
        "T2 PaDiM sigma=4": PaDiM(n_dims=32, sigma=4).fit(feats(200, False)).score(test_f),
        "T3 PatchCore": PatchCore().fit(feats(200, False)).score(test_f),
    }
    for name, (s, heat) in results.items():
        auroc = roc_auc_score(y, s)
        print(f"{name:16s} AUROC {auroc:.3f}  heatmap {heat.shape}")
        assert auroc > 0.9, name
        assert s.shape == (100,) and heat.shape[0] == 100, name
    print("self-check passed")


if __name__ == "__main__":
    _self_check()
