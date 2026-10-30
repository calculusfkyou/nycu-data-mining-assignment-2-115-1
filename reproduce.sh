#!/usr/bin/env bash
# Rebuild every file under results/ from scratch, in stage order (see README for timings).
# Usage:  bash reproduce.sh            (uses `python`; override with PY=.venv/Scripts/python.exe bash reproduce.sh)
set -euo pipefail
PY=${PY:-python}

$PY src/data.py                       # stage 2: download + extract VisA, rebuild splits/visa_split.csv
$PY src/features.py                   # stage 3: WideResNet-50 feature cache (~15 GB in cache/)
$PY src/eda.py                        # stage 4: results/stage4_eda/
$PY src/methods.py                    # stage 5: synthetic self-check of the three tiers
$PY src/run.py --quick                # stage 5: results/stage5_quick/
$PY src/run.py --quick --padim-sigma 4  # stage 5: results/stage5_quick_smooth/ (Tier 2 smoothing check)
$PY src/run.py --search               # stage 6: val-only hyperparameter search
$PY src/run.py                        # stage 6: main run, 12 classes x 3 seeds
$PY src/run.py --latency pcb1         # stage 6: latency.csv, environment.csv
$PY src/run.py --ablation             # stage 7: results/stage7_ablation/
$PY src/analyze.py                    # stage 8: results/stage8_analysis/
