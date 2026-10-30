# Draft: §3–§8 (English, for the ACL paper)

Every number below comes from `results/stage*/*.csv`. Citation keys match the BibTeX block at the end and
are written as `\citet{}` / `\citep{}` so the text can be pasted into the ACL template. Figure and table
numbers are provisional; §1, §2 and §9 are not part of this draft.

---

## 3 Task Definition & Data

**Data category and detection unit.** This is an *Image* anomaly detection task (category 3): industrial
inspection photographs whose challenge is spatial texture, background clutter and very small defects. The
unit of detection is one image. An image is *normal* if the object meets specification and *anomalous* if
it shows a surface defect (scratch, dent, colour spot, crack) or a structural defect (misplaced or missing
part). Figure 1 shows a normal and an anomalous sample for four objects; the median defect covers only
0.20 % of the image (10th / 90th percentile 0.03 % / 1.67 %).

**Dataset.** We use VisA \citep{zou2022spot}, released under CC BY 4.0
(https://github.com/amazon-science/spot-diff): N = 10,821 images (9,621 normal, 1,200 anomalous) of 12
objects in three structure groups (complex structure: pcb1–4; multiple instances: candle, capsules,
macaroni1–2; single instance: cashew, chewing gum, fryum, pipe fryum) with 34 defect types. Raw images are
1274–1562 × 960–1176 px; every tier receives a 224 × 224 × 3 input (D = 150,528). Labels are image-level
annotations with pixel masks provided by the dataset authors; subtle defects may carry label noise, which
we cannot quantify.

**Paradigm and split.** The setting is semi-supervised: models are fitted on normal images only, and a
small set of labelled anomalies is used only to choose thresholds and hyperparameters. Starting from the
official one-class split, we fix one split (seed 0) for all tiers (Table 1): 89 % of the official training
normals form *train*, 11 % plus 20 anomalies per class form *val*, 20 anomalies per class form a
*contamination pool* used only in Ablation A3, and the official test normals plus the remaining 60
anomalies per class form *test*. Anomalies are stratified by defect type, and no image appears in two
splits. For capsules every anomaly carries the same first label (bubble), so there the stratification
reduces to random sampling.

**Class imbalance.** The anomaly : normal ratio is 1 : 8.02 over the full dataset, 1 : 3.98 in val and
1 : 1.34 in test (42.8 % anomalies), far above a realistic production prevalence. A predict-all-normal rule
reaches 57.2 % test accuracy and would reach 99.0 % at a 1 % prevalence while catching nothing, so accuracy
is not reported as a metric. We use PR-AUC as the primary metric and translate every operating point to a
deployment prevalence π with prec_π = π·TPR / (π·TPR + (1 − π)·FPR), π ∈ {1 %, 5 %}.

*Table 1: Split sizes (normal / anomaly).* train 7,705 / 0; val 954 / 240; contamination pool 0 / 240;
test 962 / 720. (`results/stage4_eda/eda_split_counts.csv`)

## 4 Exploratory Feature Analysis

All statistics in this section use train normals and val images only; test images are never read.
Features are spatially averaged WideResNet-50 \citep{zagoruyko2016wide} layer2 + layer3 activations
(ImageNet weights \citep{deng2009imagenet}).

**Distribution contrast.** In the PCA and t-SNE projections (Figure 2), val anomalies overlap the
train-normal cloud for most objects and form no separate cluster, so a single global embedding is a weak
detector. The Mahalanobis distance of the global embedding to a Ledoit–Wolf Gaussian
\citep{ledoit2004well} fitted on train normals (Figure 3) separates val normal from val anomaly with a
two-sample KS statistic between 0.66 and 0.91 for ten objects, but only 0.45 for macaroni2 and 0.30 for
capsules (p = 0.12, not significant).

**Feature dispersion and redundancy.** The per-location variance of layer3 features across train normals
is 10.4 for capsules and 7.8 for macaroni2 versus 1.3–3.6 for the other objects, i.e. objects whose
instances move in the frame are much less regular at a fixed position. Layer2 and layer3 are partly
redundant (linear CKA \citep{kornblith2019similarity} 0.68–0.97), and 109–287 of the 1,536 pooled
dimensions explain 95 % of the variance, which motivates using a random channel subset.

**Acquisition shortcut.** For three objects the val anomalies concentrate in the smaller of two k-means
modes of the train normals (cashew 35.8 % of normals vs 95.0 % of anomalies; pcb1 20.1 % vs 95.0 %; pcb2
23.2 % vs 70.0 %). Image statistics explain two of them: anomalous cashew and pcb2 images are brighter
(mean grey 62.7 vs 53.1 / 55.8 and 97.7 vs 86.8 / 91.0), and 45 % of anomalous pcb1 images are rotated by
180° against about 7 % of normals. A detector can therefore score well on these objects without seeing
the defect, so we flag them before any test result (a gap of at least 40 percentage points) and report
every main result also on the remaining nine objects.

**Hypotheses.**
- **H1**: a location-wise Gaussian (Tier 2) loses most to a location-free nearest-neighbour model (Tier 3)
  on objects with high positional variance (capsules, macaroni2). Tested in §7 (per-object gap) and §8.
- **H2**: small defects are diluted in global features, so patch-level scoring is needed. Tested by
  Ablation A5 and by detection rate per defect size in §8.
- **H3**: aggregating the anomaly map by its maximum is more sensitive than by a top-1 % mean. Tested by
  Ablation A4.

*Figure 1: VisA samples from train/val; red contour = defect mask.* (`results/stage4_eda/fig_samples.png`)
*Figure 2: PCA (left) and t-SNE (right) of the global embedding, train normal vs val anomaly; axes are
unitless projections.* (`fig_pca.png`, `fig_tsne.png`)
*Figure 3: Mahalanobis distance of the global embedding (unitless), val normal vs val anomaly.*
(`fig_maha_hist.png`)

## 5 Method

**Pipeline (Figure 4).** An image is resized directly to 224 × 224 and normalised with ImageNet statistics.
We do not use the common resize-256 + centre-crop-224, because on val masks it would cut away 4.1 % of
defect pixels on average, 22.9 % for macaroni2 (12.5 % of its defects entirely) and 10.9 % for capsules.
WideResNet-50 layer2, layer3 and layer4 maps are upsampled to 28 × 28 and concatenated (3,584 channels),
and 200 channels are drawn at random (seed-dependent).

**Anomaly score.** At each location (i, j) a Gaussian is fitted on train normals \citep{defard2021padim}:

  s_ij(x) = sqrt( (f_ij − μ_ij)ᵀ (Σ_ij + εI)⁻¹ (f_ij − μ_ij) ),   ε = 0.01,
  s(x) = max_ij s_ij(x).

**Threshold.** τ is chosen per object as the threshold that maximises F2 on val, and an alert is raised
when s(x) ≥ τ. F2 weights recall four times as much as precision, matching the cost asymmetry below. We do
not fix "recall ≥ 95 %" because with 20 val anomalies per object that rule is decided by the single
hardest val anomaly and is therefore unstable.

**Error costs.** A false negative ships a defective product to a customer (returns, safety risk); a false
positive sends a good product to manual re-inspection (labour). The FN cost is higher, so τ is biased
towards recall; §8 converts the resulting false alarms into inspection workload.

**Baselines.** Tier 1 uses the mean train-normal image of each object as a template and scores the mean
absolute pixel difference on un-normalised RGB inputs; a predict-all-normal rule is reported as Tier 0.
Tier 3 is PatchCore \citep{roth2022towards}: layer2 + layer3 features with 3 × 3 local averaging (1,536
channels, no random subset), a 1 % greedy coreset memory bank selected in a 128-d random projection, and
the nearest-neighbour distance as patch score, aggregated by max. Tiers 2 and 3 share the backbone and the
224 × 224 input; Tier 1 shares only the input size.

*Figure 4: Tier 2 pipeline from input image to alert.* (`results/stage8_analysis/fig_pipeline.png`)

## 6 Experimental Setup

**Split and leakage.** All tiers use the split of §3. Models see train normals only; preprocessing uses
fixed ImageNet statistics, so nothing is fitted on val or test. For every object, all models are fitted
and all thresholds fixed before the test features are loaded, and test is evaluated once. Hyperparameter
search and latency measurement never read test. The only other test evaluation was a pipeline smoke test
on one object (candle, seed 0, default settings, with and without map smoothing) run before the search;
no setting or threshold was chosen from it.

**Hyperparameters (Table 2).** The search uses val only (seed 0) and selects, per tier, the setting with
the highest mean val PR-AUC over the 12 objects. The best Tier 2 setting lies at the edge of the grid,
which we keep as defined in advance.

| Tier | Hyperparameter | Search range | Selected | Mean val PR-AUC |
|---|---|---|---|---|
| T2 | random channels | 50, 100, 200 | 200 | 0.664 |
| T2 | backbone layers | {2, 3}, {2, 3, 4} | {2, 3, 4} | |
| T3 | coreset ratio | 1 %, 10 % | 1 % | 0.811 |
| T3 | backbone layers | {2, 3}, {2, 3, 4} | {2, 3} | |
| T2 | ε (covariance ridge) | fixed | 0.01 | – |
| T3 | neighbourhood / projection | fixed | 3 × 3 / 128-d | – |

*Table 2: Hyperparameter search on val (seed 0); grid points are the full cross product per tier.*
(`results/stage6_main/search_val.csv`)

**Metrics.** PR-AUC is primary because it is informative under imbalance \citep{davis2006relationship,
saito2015precision}; ROC-AUC is reported for comparison with the literature. At τ we report F1, F2 and the
false-positive rate, and prec_π at π = 1 % and 5 %, which states how many alerts would be real defects in
production. Results are the mean ± std over seeds 0, 1, 2 of the 12-object average (randomness: Tier 2
channel subset, Tier 3 coreset start and projection; Tier 1 is deterministic, std = 0).

**Hardware.** Python 3.12.8, torch 2.15.0 (CUDA 13.2), NVIDIA RTX 4060 Ti 8 GB, 32 GB RAM. Latency is
the median over 50 val images of pcb1 (val, so test stays untouched), measured on GPU with batch size 1
after 10 warm-up images and including feature extraction; images are decoded beforehand. Training time
is the model fitting only, excluding loading the cached features. Peak memory of the main run: 2.2 GB
VRAM, 6.6 GB RAM.

## 7 Results

**Main results (Table 3).**

| Method | PR-AUC | ROC-AUC | F1 | F2 | FPR | prec@1 % | ms / img |
|---|---|---|---|---|---|---|---|
| T0 all normal | 0.442 | 0.500 | 0 | 0 | 0 | – | – |
| T1 template | 0.702 | 0.762 | 0.693 | 0.767 | 0.465 | 0.023 | 0.12 (CPU) |
| T2 PaDiM | 0.801 ± 0.007 | 0.828 ± 0.008 | 0.733 ± 0.007 | 0.779 ± 0.006 | 0.332 ± 0.012 | 0.037 ± 0.005 | 7.38 |
| T3 PatchCore | **0.885 ± 0.002** | **0.902 ± 0.002** | **0.806 ± 0.009** | **0.808 ± 0.008** | **0.174 ± 0.021** | **0.148 ± 0.044** | 11.72 |

*Table 3: Test results averaged over 12 objects, mean ± std over 3 seeds; best per column in bold.
Without the three flagged objects (cashew, pcb1, pcb2) PR-AUC is 0.655 / 0.785 / 0.874 for T1 / T2 / T3,
the same ranking. Training time for all 12 objects: 0.1 s, 2.8 s, 114 s.*

Tier 3 outperforms Tier 2 on 11 of 12 objects: the PR-AUC gap is +0.084 with a paired bootstrap 95 % CI
of [0.065, 0.102] (test images resampled within object and label, 1,000 rounds), Wilcoxon p = 0.003 over
the 12 objects and paired t-test p = 0.003 over seeds; on nine objects the gap is +0.089 [0.067, 0.109],
p = 0.012. Tier 2 improves on Tier 1 by +0.099 PR-AUC, but only on 9 of 12 objects (Wilcoxon p = 0.077),
so we call this a trend rather than a significant gain. Our Tier 3 ROC-AUC (90.2) is 2.2 points below the
92.4 reported by \citet{zou2022spot} on the official split, a plausible gap for a re-implementation on a
different test set.

**Why Tier 2 loses.** Tier 2 assumes that the same part of the object appears at the same location in
every image. Tier 3 compares each patch with all stored normal patches regardless of position, which costs
41× the training time (114 s vs 2.8 s) and 4.3 ms more per image, both affordable on a production line. F2
hides the gap because test is 43 % anomalous: a loose τ still reaches high F2 (Tier 1: 0.767 at FPR
0.465). At 1 % prevalence only 3.7 % of Tier 2 alerts and 14.8 % of Tier 3 alerts would be real defects.

**Controlled ablations (Table 4).** Each ablation changes one factor of Tier 2; Δ is compared with the
seed std of both settings.

| Ablation | Setting | PR-AUC | Δ |
|---|---|---|---|
| A1 scoring | Mahalanobis (full) | 0.801 ± 0.007 | – |
| | Euclidean 1-NN | 0.734 ± 0.003 | −0.067* |
| | cosine 1-NN | 0.686 ± 0.021 | −0.115* |
| A3 contamination | 0 % (full) | 0.801 ± 0.007 | – |
| | 1 % | 0.797 ± 0.007 | −0.004 |
| | 2 % | 0.794 ± 0.008 | −0.007 |
| A4 aggregation | max (full) | 0.801 ± 0.007 | – |
| | top-1 % mean | 0.799 ± 0.006 | −0.002 |
| | max of smoothed map (σ = 4) | 0.796 ± 0.007 | −0.004 |
| A5 features | patch-level (full) | 0.801 ± 0.007 | – |
| | global pooled | 0.771 ± 0.006 | −0.029* |

*Table 4: Tier 2 ablations, 12 objects, 3 seeds; * = |Δ| larger than the seed std. On nine objects A5
gives −0.038 (0.785 → 0.747).*

| A2 threshold rule | Recall | FPR | F2 | prec@1 % |
|---|---|---|---|---|
| best F2 on val (full) | 0.820 | 0.332 | 0.779 | 0.037 |
| best F1 on val | 0.639 | 0.178 | 0.651* | 0.090* |
| 99th percentile of val normals | 0.294 | 0.019 | 0.329* | 0.513* |
| review budget: top 10 % of val | 0.345 | 0.039 | 0.389* | 0.340* |

*Table 5: Ablation A2. The model is unchanged, so PR-AUC stays 0.801; only the operating point moves.*

The location-wise covariance matters most (A1): replacing it with a nearest-neighbour distance at the same
location costs 0.07–0.12 PR-AUC. Pooling the map into one global vector (A5) costs 0.029, supporting H2.
The threshold rule (A2) leaves PR-AUC unchanged but moves the precision at 1 % prevalence from 3.7 % to
51.3 %, at the price of recall falling from 0.82 to 0.29. Two results are negative and we report them as
such. Contaminating train with up to 2 % anomalies (A3; 4–16 images per object, the pool allows at most
about 2.5 %, so 5 % and 10 % were not possible) changes PR-AUC by less than the seed std, presumably because
a median defect covers 0.20 % of the image, so the fraction of contaminated *locations* stays far below
2 %. The aggregation (A4) also has no effect beyond seed noise, so H3 is not supported; Gaussian smoothing
as in \citet{defard2021padim} does not help either.

## 8 Analysis

**Failure cases (Table 6, Figure 5).** We rank the Tier 2 test scores (seed 0; 137 FN among 720
anomalies, 308 FP among 962 normals) by s(x)/τ and take the most inverted and the nearest miss of each
error type.

| # | Type | Image | Defect (area) | s(x) | τ | s/τ |
|---|---|---|---|---|---|---|
| 1 | FN, inversion | pcb2 Anomaly/022 | melt (0.195 %) | 9.85 | 13.50 | 0.73 |
| 2 | FN, near miss | pipe_fryum Anomaly/055 | similar colour spot (0.038 %) | 14.24 | 14.25 | 0.999 |
| 3 | FP, inversion | pcb4 Normal/0871 | – | 26.42 | 12.67 | 2.09 |
| 4 | FP, near miss | pcb4 Normal/0457 | – | 12.67 | 12.67 | 1.000 |

*Table 6: Tier 2 failure cases on test.*

In case 1 the melt lies among dense components and the map stays low everywhere; its maximum is in the
background on the right, so the defect is simply not seen. In case 2 the score is high, but it comes from
the boundary between the object and the background, not from the tiny spot, so the near miss is
coincidental. In cases 3 and 4 the board sits slightly away from its usual position and the maximum lies
on the background or the board edge. The shared blind spot is position: a location-wise Gaussian treats a
shifted object as anomalous at its edges, and this edge response masks small defects. Across all test
anomalies, the Tier 2 map peaks inside the defect mask (dilated by 8 px) for 25.3 % of images (0 % for
capsules) against 70.3 % for Tier 3. The first fix we would try is image registration before scoring, or
a position-free patch score, which is what Tier 3 uses.

**H1 and H2.** H1 is only partly supported: capsules has the largest Tier 3 − Tier 2 gap (+0.196 PR-AUC),
but on macaroni2 Tier 2 is better (0.544 vs 0.492; all tiers are close to chance there), and the
multiple-instance group gap (+0.095) equals the complex-structure gap in the 12-object view. For H2,
patch-level scoring detects more anomalies than the global variant in every defect-size tercile (recall at
τ: small < 0.25 %: 0.770 vs 0.742; medium: 0.834 vs 0.800; large > 0.98 %: 0.983 vs 0.969), with the
largest gap for medium defects on nine objects (0.885 vs 0.793).

**Practical impact (Table 7).** At 1 % prevalence, 10,000 inspected items contain 100 defects.

| Method (τ rule) | Missed | False alarms | Flagged | Defects among flagged |
|---|---|---|---|---|
| T1 (best F2) | 16.7 | 4,604 | 4,687 | 1.8 % |
| T2 (best F2) | 18.0 | 3,288 | 3,370 | 2.4 % |
| T3 (best F2) | 18.5 | 1,722 | 1,804 | 4.5 % |
| T2 (99th percentile of val normals) | 70.6 | 186 | 215 | 13.6 % |

*Table 7: Expected outcome per 10,000 items at 1 % prevalence, from the class-averaged TPR and FPR (hence
different from the per-class prec@1 % in Table 3).*

With the recall-oriented threshold, Tiers 2 and 3 miss about 18 of 100 defects, but Tier 3 sends 1,566
fewer good items to re-inspection, roughly halving the inspectors' workload. If the line can only
re-inspect about 2 % of production, Tier 2 needs the 99th-percentile threshold and then misses about 71
of 100 defects; this trade-off between inspection labour and escaped defects is the decision the quality
manager has to make.

**Interpretability.** Figure 6 shows correctly flagged anomalies with Tier 2 and Tier 3 maps. For a
missing component on pcb4 both maps peak on the empty pad, so an engineer could act on the map directly.
For a chewing gum with a missing chunk, Tier 2 flags the image (s/τ = 5.19) but its map peaks on the
shadow above the object, not on the defect; Tier 3 peaks on the defect. Pixel ROC-AUC averages 0.925 for
Tier 2 and 0.973 for Tier 3. We would therefore show engineers the Tier 3 map as a localisation aid, and
treat the Tier 2 map, which points at the defect in only a quarter of cases, as unreliable. The
acquisition shortcut of §4 points the same way: on cashew, pcb1 and pcb2 a high score may reflect
brightness or orientation rather than the defect, so these objects are reported separately throughout.

*Figure 5: Tier 2 failure cases; white contour = defect mask.* (`results/stage8_analysis/fig_failure_cases.png`)
*Figure 6: Correctly flagged anomalies with Tier 2 and Tier 3 maps.* (`fig_heatmaps.png`)

---

## BibTeX

```bibtex
@inproceedings{zou2022spot,
  title     = {{SPot-the-Difference} Self-supervised Pre-training for Anomaly Detection and Segmentation},
  author    = {Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar},
  booktitle = {Computer Vision -- ECCV 2022},
  pages     = {392--408},
  year      = {2022},
  doi       = {10.1007/978-3-031-20056-4_23}
}
@inproceedings{defard2021padim,
  title     = {{PaDiM}: A Patch Distribution Modeling Framework for Anomaly Detection and Localization},
  author    = {Defard, Thomas and Setkov, Aleksandr and Loesch, Angelique and Audigier, Romaric},
  booktitle = {Pattern Recognition. ICPR International Workshops and Challenges},
  pages     = {475--489},
  year      = {2021},
  doi       = {10.1007/978-3-030-68799-1_35}
}
@inproceedings{roth2022towards,
  title     = {Towards Total Recall in Industrial Anomaly Detection},
  author    = {Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {14298--14308},
  year      = {2022},
  doi       = {10.1109/CVPR52688.2022.01392}
}
@inproceedings{zagoruyko2016wide,
  title     = {Wide Residual Networks},
  author    = {Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  pages     = {87.1--87.12},
  year      = {2016},
  doi       = {10.5244/C.30.87}
}
@inproceedings{deng2009imagenet,
  title     = {{ImageNet}: A Large-Scale Hierarchical Image Database},
  author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {248--255},
  year      = {2009},
  doi       = {10.1109/CVPR.2009.5206848}
}
@article{ledoit2004well,
  title     = {A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices},
  author    = {Ledoit, Olivier and Wolf, Michael},
  journal   = {Journal of Multivariate Analysis},
  volume    = {88},
  number    = {2},
  pages     = {365--411},
  year      = {2004},
  doi       = {10.1016/S0047-259X(03)00096-4}
}
@inproceedings{kornblith2019similarity,
  title     = {Similarity of Neural Network Representations Revisited},
  author    = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019},
  eprint    = {1905.00414},
  archivePrefix = {arXiv}
}
@inproceedings{davis2006relationship,
  title     = {The Relationship Between Precision-Recall and {ROC} Curves},
  author    = {Davis, Jesse and Goadrich, Mark},
  booktitle = {Proceedings of the 23rd International Conference on Machine Learning (ICML)},
  pages     = {233--240},
  year      = {2006},
  doi       = {10.1145/1143844.1143874}
}
@article{saito2015precision,
  title     = {The Precision-Recall Plot Is More Informative than the {ROC} Plot When Evaluating Binary Classifiers on Imbalanced Datasets},
  author    = {Saito, Takaya and Rehmsmeier, Marc},
  journal   = {PLOS ONE},
  volume    = {10},
  number    = {3},
  pages     = {e0118432},
  year      = {2015},
  doi       = {10.1371/journal.pone.0118432}
}
```
