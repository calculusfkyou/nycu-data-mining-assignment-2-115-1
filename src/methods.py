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
    return x if x.shape[-1] == SIDE else F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False)


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

    Ablation switches (defaults = the method above):
      metric  "euclid" / "cosine": 1-NN distance to the train features at the same location instead of Mahalanobis
      agg     "top1pct": s(x) = mean of the top 1 % of map values instead of the max
      pooled  True: average the features over all locations first, i.e. one global Gaussian per image
    """

    def __init__(self, n_dims=100, eps=0.01, seed=0, sigma=None, layers=("layer2", "layer3"),
                 metric="mahalanobis", agg="max", pooled=False):
        self.n_dims, self.eps, self.seed, self.sigma, self.layers = n_dims, eps, seed, sigma, layers
        self.metric, self.agg, self.pooled = metric, agg, pooled

    def _embed(self, feats, sl):
        x = torch.cat([_up(_to_gpu(feats[l], sl, self.idx[l])) for l in self.layers], 1).flatten(2)  # (B, d, 784)
        return x.mean(2, keepdim=True) if self.pooled else x                                         # (B, d, P)

    def fit(self, feats):
        sizes = [feats[l].shape[1] for l in self.layers]
        dims = np.sort(np.random.default_rng(self.seed).choice(sum(sizes), self.n_dims, replace=False))
        starts = np.cumsum([0] + sizes)
        self.idx = {l: dims[(dims >= a) & (dims < b)] - a for l, a, b in zip(self.layers, starts, starts[1:])}
        n = len(feats["layer2"])
        x = torch.cat([self._embed(feats, sl) for sl in _chunks(n)])  # (N, d, P)
        if self.metric != "mahalanobis":
            x = F.normalize(x, dim=1) if self.metric == "cosine" else x
            self.train = x.permute(2, 0, 1).half().cpu()                 # (P, N, d), kept off the GPU
            return self
        self.mu = x.mean(0)
        xc = (x - self.mu).permute(2, 0, 1)                             # (P, N, d)
        cov = xc.transpose(1, 2) @ xc / (n - 1)
        cov += self.eps * torch.eye(self.n_dims, device=DEVICE)
        self.prec = torch.linalg.inv(cov)                              # (P, d, d)
        return self

    def _location_scores(self, e, train):
        if self.metric == "mahalanobis":
            d = (e - self.mu).permute(2, 0, 1)                          # (P, B, d)
            return ((d @ self.prec) * d).sum(-1).clamp_min(0).sqrt()     # (P, B)
        q = e.permute(2, 0, 1)
        if self.metric == "cosine":
            return (1 - F.normalize(q, dim=-1) @ train.transpose(1, 2)).min(-1).values
        return torch.cdist(q, train).min(-1).values

    def score(self, feats):
        train = self.train.float().to(DEVICE) if self.metric != "mahalanobis" else None
        maps = []
        for sl in _chunks(len(feats["layer2"])):
            m = self._location_scores(self._embed(feats, sl), train).T     # (B, P)
            side = 1 if self.pooled else SIDE
            m = m.reshape(-1, 1, side, side)
            if self.sigma and not self.pooled:
                m = F.interpolate(m, size=(224, 224), mode="bilinear", align_corners=False)
                m = gaussian_blur(m, kernel_size=8 * self.sigma + 1, sigma=self.sigma)  # 4-sigma truncation
            maps.append(m[:, 0].cpu())
        maps = torch.cat(maps)
        flat = maps.flatten(1)
        if self.agg == "top1pct":
            s = flat.topk(max(1, int(np.ceil(0.01 * flat.shape[1]))), dim=1).values.mean(1)
        else:
            s = flat.max(1).values
        return s.numpy(), maps.numpy()


class PatchCore:
    """Tier 3: locally aware patch features (3x3 average), greedy coreset memory bank, nearest-neighbour distance.

    The coreset is selected in a 128-d random projection; the bank keeps the full features (1536-d for layer2+3).
    """

    def __init__(self, coreset=0.01, proj_dim=128, seed=0, layers=("layer2", "layer3")):
        self.coreset, self.proj_dim, self.seed, self.layers = coreset, proj_dim, seed, layers

    def _embed(self, feats, sl):
        x = torch.cat([_up(F.avg_pool2d(_to_gpu(feats[l], sl), 3, 1, 1)) for l in self.layers], 1)  # (B, C, 28, 28)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])            # (B*784, C)

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
        # squared distances via ||z||^2 - 2 z.c + ||c||^2: one GEMV per step and no host sync (~5x faster than cdist)
        zz = (z * z).sum(1)
        sq_dist = lambda j: zz - 2 * (z @ z[j]) + zz[j]
        j = torch.randint(len(z), (), device=DEVICE, generator=g)
        sel, min_d = [j], sq_dist(j)
        for _ in range(k - 1):
            j = min_d.argmax()
            sel.append(j)
            min_d = torch.minimum(min_d, sq_dist(j))
        self.bank = full[torch.stack(sel).cpu()].float().to(DEVICE)    # (k, C)
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
        "T2 euclid 1-NN": PaDiM(n_dims=32, metric="euclid").fit(feats(200, False)).score(test_f),
        "T2 top-1% mean": PaDiM(n_dims=32, agg="top1pct").fit(feats(200, False)).score(test_f),
        "T3 PatchCore": PatchCore().fit(feats(200, False)).score(test_f),
    }
    for name, (s, heat) in results.items():
        auroc = roc_auc_score(y, s)
        print(f"{name:16s} AUROC {auroc:.3f}  heatmap {heat.shape}")
        assert auroc > 0.9, name
        assert s.shape == (100,) and heat.shape[0] == 100, name

    # cosine 1-NN: a uniform +3 shift barely changes the direction of zero-mean noise, so the AUROC test above
    # does not apply; check the distances against a brute-force computation instead
    tr, te = feats(30, False), feats(5, True)
    m = PaDiM(n_dims=16, metric="cosine").fit(tr)
    _, heat = m.score(te)
    emb = lambda f: torch.cat([_up(torch.from_numpy(np.asarray(f[l][:, m.idx[l]], np.float32))) for l in m.layers],
                              1).flatten(2).numpy()                     # (N, d, 784)
    a, b = emb(tr), emb(te)
    a, b = a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)
    brute = (1 - np.einsum("bdp,ndp->bnp", b, a)).min(1).reshape(5, SIDE, SIDE)
    assert np.abs(brute - heat).max() < 1e-2, np.abs(brute - heat).max()  # float16 storage of the train bank
    print("T2 cosine 1-NN   matches brute force")
    print("self-check passed")


if __name__ == "__main__":
    _self_check()
