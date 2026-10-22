# Training-only exploration

Run `python EDA/eda.py` from the repository root. It reads only `Dataset/train.csv` (82 rows: 32 successful and 50 unresolved runs). The autoencoder subsequently fits only on the 32 successful rows.

Outputs include per-feature KS, standardized Wasserstein and mutual information, a full Spearman redundancy matrix, distribution contrasts, and a PCA projection. The PCA/EDA scaler sees training rows only; model preprocessing is fitted separately on its permitted fitting rows. KS p-values are uncorrected descriptive diagnostics.

Received tokens have W=0.957, KS=0.505. Steps and command counts are exactly redundant; token and cost features also correlate. H1 asks whether workflow signals improve ranking over patch-only features; H2 tests contamination sensitivity; H3 tests F2 versus F1 thresholding. The current report resolves H1/H2 with qualified support and H3 with no observed difference.

Detailed repository figures are saved in `figures/`. `Paper/compact_figures.py` creates separate plots sized for readable single-column print. Test data are never loaded by EDA. Root README documents the revised split and earlier test reuse.
