# 確認清單：VisA HW2 §3–§8

每個項目有兩個確認點：

- **實作**：由實作人員勾選，表示這一項已經做完。「證據」欄要寫出對應的檔案、數字或指令輸出。
- **Review**：由 reviewer 依「Review 方式」實際檢查後勾選，表示實作正確。不通過時標 ❌，並在證據欄寫明原因。

狀態標記：⬜ 未完成 ✅ 完成 🔶 部分完成或有疑問 ❌ Review 不通過

一個階段只有在所有項目的 **Review 都是 ✅** 時，才算完成並進入下一階段。

## 階段對照表

下方各區是依論文章節與主題分區；工作則依計畫的「執行順序」分成 10 個階段。完成一個階段時，依這張表勾選對應的項目。

| 階段 | 執行順序 | 內容 | 對應項目 |
|---|---|---|---|
| 1 | 步驟 1–2 | 檢查磁碟空間、建立分支、`.gitignore`、`.venv`、安裝套件 | 0.1–0.7 |
| 2 | 步驟 3 | `data.py`：下載、解壓、產生切分 | 0.8、1.1–1.11、11.1、11.2 |
| 3 | 步驟 4 | `features.py`：快取 12 類的特徵 | 4.13（快取部分）、10.1、10.2 |
| 4 | 步驟 5 | `eda.py`：§3 樣本圖與摘要表、§4 圖表與統計、center crop 的 mask 損失檢查、拍攝條件原因確認 | 2.3、2.5、2.6、2.8、2.9、3.1–3.5、4.3、10.1、11.3（EDA 部分）、13.1–13.3 |
| 5 | 步驟 6 | `methods.py`（Tier 1/2/3、合成資料自檢）+ `run.py --quick` | 4.4、4.8–4.11、4.13（memory bank 部分）、5.1、5.2、5.7、10.1、10.3、11.4 |
| 6 | 步驟 7 | `run.py` 主實驗，檢查 Tier 3 是否接近文獻數字 | 4.5、4.6、5.3–5.6、5.8、5.9、6.1、6.2、10.1、11.3（τ 部分）、11.5、13.4 |
| 7 | 步驟 8 | `run.py` ablations | 6.6–6.11、13.5 |
| 8 | 步驟 9 | `analyze.py`：顯著性、失敗案例、熱圖、實務影響、pipeline 圖 | 4.1、6.3、6.4、7.1、7.2、7.4、7.5、7.8、10.1、13.6 |
| 9 | 步驟 10–11 | 撰寫 `reports/sec3-8.md` 與 `reports/progress.md` | 2.1、2.2、2.4、2.7、2.10、2.11、3.6、3.7、4.2、4.7、4.12、6.5、6.12、7.3、7.6、7.7、7.9、8.1–8.10、9.1–9.3、11.6、13.7 |
| 10 | 步驟 12 | 更新 README、最後的 commit | 12.1–12.4 |

- 0.4（`main` 沒有新 commit；只在同意後 push；沒有 merge、沒有 PR）是持續項目，每個階段結束時都要重新確認。
- 10.1 會在每個新增 `src/` 檔案的階段更新，到第 8 階段才會全部完成。
- 各階段的結果放在 `results/` 底下自己的資料夾：第 4 階段 `stage4_eda/`、第 5 階段 `stage5_quick/`、第 6 階段 `stage6_main/`、第 7 階段 `stage7_ablation/`、第 8 階段 `stage8_analysis/`。

## 0. 環境與分支

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 0.1 | 所有檔案都在 `nycu-data-mining-assignment-2-115-1` repo 內，上層 `work2/` 沒有多出工作檔案 | ✅ | 🔶 | 列出 `work2/` 的內容，確認沒有程式或結果檔 | `work2/` 只有 PDF、計畫與舊版 checklist，沒有程式或結果檔；Review（2026-10-30，本機重現）：`work2/` 在 partner 電腦上，本機無法檢查；repo 內沒有多出的工作檔案 |
| 0.2 | D 槽剩餘空間 ≥ 25 GB；`data/`、`cache/`、`.venv/` 都在 repo 內並被 ignore | ✅ | ✅ | `git check-ignore -v data/x cache/x .venv` | D 槽剩 408 GB；`git check-ignore` 三條規則都有作用；Review（2026-10-30，本機重現）：本機為 C 槽，剩 317 GB；`git check-ignore -v` 對 data/、cache/、.venv、*.tar 都有作用 |
| 0.3 | 已從 `main` 建立 `feature/visa-experiments`，所有工作都在這個分支上 | ✅ | ✅ | `git log --oneline --graph --all` | Review（2026-10-30，本機重現）：`git log --graph --all` 為單一直線歷史，全部 commit 都在本分支 |
| 0.4 | `main` 上沒有新的 commit；只在同意後才 push 分支；沒有 merge、沒有開 PR | ✅ | ✅ | `git log main --oneline`、`git ls-remote --heads origin`、GitHub 的 Pull requests 頁面 | 2026-10-30：經同意將 `feature/visa-experiments` 推上 origin（adf096b）；`main` 仍只有 d6ca658；沒有 PR；Review（2026-10-30，本機重現）：`main` 只有 d6ca658；`ls-remote` 只有 main 與本分支；沒有 PR |
| 0.5 | `.venv` 使用 Python 3.9.1；CUDA 版 torch 可以使用 GPU | ✅ | 🔶 | `.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | torch 2.8.0+cu126，RTX 4070；Review（2026-10-30，本機重現）：本機 `.venv` 為 Python 3.12.8、torch 2.15.0+cu132，CUDA 可用；與 3.9.1 / 2.8.0 不同，但以下各項重現結果一致 |
| 0.6 | `requirements.txt` 版本已鎖定，並註明 CUDA 安裝指令 | ✅ | 🔶 | 開檔對照 `pip freeze` | Review（2026-10-30，本機重現）：版本已鎖定且附 CUDA 指令；本機未依此檔安裝，無法對照 partner 的 `pip freeze` |
| 0.7 | `.gitignore` 包含 `data/`、`cache/`、`.venv/`、`*.tar`，且在下載資料前就已建立 | ✅ | ✅ | `git log --diff-filter=A -- .gitignore` 的時間早於下載 | Review（2026-10-30，本機重現）：.gitignore 於 4547e0d（08:17）加入，早於資料管線 c185819（10:04） |
| 0.8 | `git status` 和 commit 歷史中都沒有資料集檔案 | ✅ | ✅ | `git log --all --stat \| grep -i -E "\.jpg\|\.png\|\.tar"` 沒有結果 | 歷史中只有 .gitignore、README.md、requirements.txt；grep 唯一命中是 commit 訊息文字；Review（2026-10-30，本機重現）：歷史中只有 4 張 `results/stage4_eda/fig_*.png` 結果圖，沒有 .jpg / .tar / .npy |

## 1. 資料與切分（§3）

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 1.1 | `src/data.py` 能下載並解壓 VisA 到 `data/`，已存在時會跳過 | ✅ | ✅ | 閱讀 `download()`；重跑一次確認不會重新下載 | Review（2026-10-30，本機重現）：本機重跑 `data.py` 沒有出現 downloading / extracting，直接產生切分 |
| 1.2 | 總數 N = 10,821（9,621 normal / 1,200 anomaly） | ✅ | ✅ | 讀 `splits/visa_split.csv` 計數 | 9,621 normal / 1,200 anomaly；影像集合與標籤都和官方 1cls.csv 完全一致；Review（2026-10-30，本機重現）：獨立計數 9,621 / 1,200；影像集合與標籤和官方 1cls.csv 完全一致 |
| 1.3 | 切分用 seed=0 產生，可以重現 | ✅ | ✅ | 重跑 `python src/data.py`，比對 md5 | 重跑後 md5 仍為 436809f3fdb8a8d0f2a40b093e51ab58；Review（2026-10-30，本機重現）：本機（numpy 2.5.4、pandas 3.1.0）重跑後 md5 仍為 436809f3…，跨版本可重現 |
| 1.4 | `splits/visa_split.csv` 已 commit | ✅ | ✅ | `git log -- splits/visa_split.csv` | Review（2026-10-30，本機重現）：c185819 加入 |
| 1.5 | train 為官方 train normal 的 ~89%，只含正常樣本 | ✅ | ✅ | 計數；確認 train 的 label 全是 normal | 7,705 張，label 只有 normal；Review（2026-10-30，本機重現）：7,705 張、label 只有 normal；train + val normal 與官方 train 集合完全相同 |
| 1.6 | val 為官方 train normal 的 ~11% + 每類 20 張 anomaly | ✅ | ✅ | 依類別計數 | 954 normal + 240 anomaly；每類 val normal 佔 11.0–11.1%；val 異常比例 17–29%；Review（2026-10-30，本機重現）：val normal 佔 10.95–11.11%；每類 val anomaly 剛好 20 |
| 1.7 | contam pool 為每類 20 張 anomaly | ✅ | ✅ | 依類別計數 | 240 張；Review（2026-10-30，本機重現）：每類 contam 剛好 20，共 240 |
| 1.8 | test 為官方 test normal 全部 + 每類 60 張 anomaly | ✅ | ✅ | 依類別計數；test normal 要和官方 1cls 完全相同 | 962 + 720 張；test normal 與官方 test normal 集合完全相同；Review（2026-10-30，本機重現）：每類 test anomaly 剛好 60；test normal 與官方 test normal 集合相同 |
| 1.9 | anomaly 依瑕疵類型分層抽樣；少於 3 張的類型併入 "other" | ✅ | ✅ | 閱讀 `_split_anomalies()`；用 crosstab 看各類型在三個 split 的分布 | 各類類型數 1–8 種；沒有類型需要併入 "other"；capsules 只有 `bubble` 一種；1,200 張 anomaly 都有 mask；Review（2026-10-30，本機重現）：64 個（類別, 瑕疵類型）層，最小 6 張；各層與 3:1:1 的最大偏差 test 1.2、val 0.8、contam 0.8 張；1,200 個 mask 檔都存在 |
| 1.10 | 各 split 之間的影像沒有重疊 | ✅ | ✅ | 閱讀 `check_split()` 的 assert，並獨立用集合交集再驗一次 | 獨立以集合交集驗證，六組 split 兩兩交集皆為 0；Review（2026-10-30，本機重現）：獨立以集合交集驗證，六組兩兩交集皆為 0 |
| 1.11 | `VisADataset`：直接 resize 224×224、可關閉 ImageNet 正規化、mask 用最近鄰縮放 | ✅ | ✅ | 取一張 anomaly 檢查 shape、標籤、mask 非空 | Review（2026-10-30，本機重現）：pcb1 val anomaly：影像 (3,224,224)、label 1、mask (1,224,224) 非空；normalize=False 時值域 [0.12, 1]；mask 用 NEAREST |

## 2. §3 Task Definition & Data

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 2.1 | 明確寫出資料類別為「Image / Video」，並討論這類資料的挑戰 | ⬜ | ⬜ | 閱讀草稿 §3 | |
| 2.2 | 偵測單位為 image-level；正常與異常的定義都有寫 | ⬜ | ⬜ | 閱讀草稿 §3 | |
| 2.3 | 正常與異常範例縮圖各一張 | ✅ | ✅ | 開圖確認標籤正確 | `results/stage4_eda/fig_samples.png`：pcb1、capsules、cashew、macaroni2 各一張正常與異常（只取 train/val），異常以紅線標出 mask；Review（2026-10-30，本機重現）：程式只從 train / val 取樣；異常以 mask 輪廓標出 |
| 2.4 | 引用 Zou et al., ECCV 2022、資料連結與 CC BY 4.0 | ⬜ | ⬜ | 到 ECCV 論文頁與 GitHub 查證 | |
| 2.5 | 報告 N、anomaly 數、D（150,528 與 Tier 2 的 100 維） | ✅ | ✅ | 數字與 `splits/visa_split.csv` 對照 | `results/stage4_eda/eda_summary.csv`：N 10,821、anomaly 1,200、D = 150,528、Tier 2 每位置 100 維；原圖寬 1274–1562、高 960–1176 px；Review（2026-10-30，本機重現）：數字與 `eda_summary.csv` 及切分計數一致 |
| 2.6 | 各類各 split 的數量表與摘要統計表 | ✅ | ✅ | 數字與 CSV 對照 | `results/stage4_eda/eda_split_counts.csv`（各類各 split）、`results/stage4_eda/eda_summary.csv`（摘要統計，含瑕疵面積中位數 0.20%）；Review（2026-10-30，本機重現）：本機重算 `eda_split_counts.csv` 與 commit 版本完全相同 |
| 2.7 | 標籤來源、可能的標籤雜訊、學習範式（半監督） | ⬜ | ⬜ | 閱讀草稿 §3 | |
| 2.8 | 不平衡比例一律寫成 anomaly : normal（全資料 1 : 8，test 約 1 : 1.34） | ✅ | ✅ | 自行計算並比對 | `results/stage4_eda/eda_imbalance.csv`：全資料 1 : 8.02、test 1 : 1.34、val 1 : 3.98；Review（2026-10-30，本機重現）：本機重算 `eda_imbalance.csv` 完全相同 |
| 2.9 | 用「全猜正常」示範 accuracy 會失效 | ✅ | ✅ | 檢查數字是否等於 normal 比例 | `results/stage4_eda/eda_imbalance.csv`：全猜正常的 accuracy 為全資料 88.9%、test 57.2%、π=1% 時 99.0%、π=5% 時 95.0%；Review（2026-10-30，本機重現）：accuracy 等於 normal 比例（57.2% = 962 / 1,682） |
| 2.10 | 對策：PR-AUC 為主要指標，另報 prec_π（π = 1%、5%） | ⬜ | ⬜ | 檢查公式 | |
| 2.11 | 提到 val 的異常比例遠高於部署時的盛行率 | ⬜ | ⬜ | 閱讀草稿 | |

## 3. §4 Exploratory Feature Analysis

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 3.1 | 只用 train normal + val anomaly，完全沒碰 test；文中寫明有使用 val anomaly | 🔶 | 🔶 | 在 `eda.py` 搜尋 `"test"`，確認沒有讀取 test | 程式部分完成：特徵只讀 train/val（`global_embeddings()` 有 assert），mask 只讀 val/contam，test 只出現在標籤計數表；另外使用 val normal 作為未參與擬合的正常對照。「文中寫明」待第 9 階段；Review（2026-10-30，本機重現）：程式部分 ✅：`eda.py` 中 test 只出現在第 42 行的計數表；mask 統計也用了 contam pool（非 test，可接受）。文字待第 9 階段 |
| 3.2 | 全域嵌入的 PCA / t-SNE 分布對比圖 | ✅ | ✅ | 開圖：有座標軸標籤、圖例 | `results/stage4_eda/fig_pca.png`、`results/stage4_eda/fig_tsne.png`（train normal vs val anomaly），另有 `results/stage4_eda/fig_maha_hist.png`；Review（2026-10-30，本機重現）：兩張圖都有座標軸標籤與圖例 |
| 3.3 | 以 Mahalanobis 距離與 PCA 分量計算 KS / Wasserstein 距離 | ✅ | ✅ | 閱讀程式與輸出表 | `results/stage4_eda/eda_separability.csv`：Mahalanobis KS 0.30（capsules，p=0.12）～0.91（pipe_fryum）；macaroni2 0.45；另附 W1/std 與 PC1–3 的 KS；Review（2026-10-30，本機重現）：本機重算：只有 capsules 改變（KS 0.300→0.317、p 0.121→0.088，仍 > 0.05，結論不變），其餘差異 < 0.003 |
| 3.4 | 每類的類內變異：對齊型 vs 多物體型 | ✅ | ✅ | 閱讀輸出表 | `results/stage4_eda/eda_dispersion.csv`：layer3 同一位置變異 capsules 10.4、macaroni2 7.8，其餘 1.3–3.6；依 VisA 三種結構類型分組；Review（2026-10-30，本機重現）：本機重算最大相對差 4e-5 |
| 3.5 | 特徵層冗餘分析（layer2 vs layer3） | ✅ | ✅ | 閱讀輸出 | `results/stage4_eda/eda_redundancy.csv`：layer2 vs layer3 linear CKA 0.68–0.97；解釋 95% 變異需 109–287 / 1536 維；Review（2026-10-30，本機重現）：本機重算最大相對差 9e-6 |
| 3.6 | 每張圖都有解讀文字 | ⬜ | ⬜ | 閱讀草稿 §4 | |
| 3.7 | H1–H3 都已寫出，並註明在 §7–§8 用什麼驗證 | ⬜ | ⬜ | 閱讀草稿 §4 | |

## 4. §5 Method

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 4.1 | `results/stage8_analysis/fig_pipeline.png` 包含 8 個步驟 | ⬜ | ⬜ | 開圖對照計畫 | |
| 4.2 | 前處理直接 resize 224×224（不做 center crop），並寫明理由 | ⬜ | ⬜ | 閱讀程式與草稿 | |
| 4.3 | `eda.py` 報告「若改用 center crop，mask 會損失多少比例」 | ✅ | ✅ | 檢查輸出數字 | `results/stage4_eda/eda_crop_loss.csv`：resize 256 + center crop 224 平均切掉 4.1% 瑕疵像素；macaroni2 22.9%（12.5% 影像完全切掉）、capsules 10.9%、pcb3 9.7%；Review（2026-10-30，本機重現）：本機重算完全相同；以 shorter side = 256 換算裁切框的邏輯正確 |
| 4.4 | s_ij(x)、s(x) = max、Σ + εI（ε = 0.01）的公式與程式一致 | ✅ | ⬜ | 公式與 `methods.py` 逐行比對 | `methods.PaDiM`：`s_ij` 以 `(Σ_ij + εI)^-1` 計算（ε = 0.01），`s(x)` = 28×28 分數圖的 max。主設定維持不平滑；另保留 `sigma` 選項（`run.py --padim-sigma 4`，上採樣 224 + Gaussian σ=4），結果在 `results/stage5_quick_smooth/`：12 類 × 3 seeds 的 val PR-AUC 0.578 → 0.581（Δ < seed std 0.013，Wilcoxon p = 0.57），可作為 ablation 的負面結果 |
| 4.5 | τ 每類各自決定；主設定為 val best-F2，並說明不用 recall ≥ 95% 的原因 | 🔶 | ⬜ | 閱讀 τ 的程式 | `run_class` 對每類、每個 (tier, seed) 各自以 `best_f2_tau(val)` 決定 τ；不用 recall ≥ 95% 的原因待草稿 |
| 4.6 | τ 的選擇完全沒用到 test | ✅ | ⬜ | 追蹤 τ 函式的輸入來源 | τ 只由 `best_f2_tau(y_val, s_val)` 產生；test 在所有 τ 固定後才讀取 |
| 4.7 | FP / FN 成本都有說明，並解釋 τ 為什麼傾向壓低 FN | ⬜ | ⬜ | 閱讀草稿 §5 | |
| 4.8 | Tier 1 模板差 + 「全猜正常」對照 | ✅ | ⬜ | 閱讀程式 | `methods.TemplateDiff`：每類 train normal 平均影像當模板，s(x) = 平均 |x − template|；`run.py` 另輸出 `T0_all_normal` 列（candle test acc 62.5%、PR-AUC 0.375） |
| 4.9 | Tier 2 PaDiM 式方法 | ✅ | ⬜ | 閱讀程式，並看合成資料自檢結果 | `methods.PaDiM`：layer2 + 上採樣的 layer3，隨機 100 維（seed 控制），每個位置一個高斯；合成資料自檢 AUROC 1.000 |
| 4.10 | Tier 3 PatchCore：3×3 鄰域平均、128 維投影、greedy coreset 1%、最近鄰距離 | ✅ | ⬜ | 閱讀程式，並看合成資料自檢結果 | `methods.PatchCore`：3×3 avg pool、128 維隨機投影上做 greedy coreset 1%、bank 保留 1536 維、最近鄰距離取 max；合成資料自檢 AUROC 1.000 |
| 4.11 | 三個方法共用介面 `fit(train)` / `score(x) → (image_score, heatmap)` | ✅ | ⬜ | 閱讀程式 | 三個 class 都是 `fit(train)` / `score(x) → (image_score (N,), heatmap (N, h, w))` |
| 4.12 | 各 tier 的前處理差異已寫明 | ⬜ | ⬜ | 閱讀草稿 | |
| 4.13 | 記憶體做法：每次只處理一類、分層 float16 快取、memory bank 放 CPU 分塊計算 | ✅ | 🔶 | 閱讀程式；看 `--quick` 記錄的 VRAM / RAM 峰值 | 特徵快取部分已完成：每類、每層分開存 float16，抽特徵時 VRAM 峰值 0.67 GB；memory bank 的分塊計算待 `methods.py` 實作；Review（2026-10-30，本機重現）：快取部分 ✅（每類每層分開、float16）；memory bank 部分待第 5 階段；PatchCore 全部 patch 以 float16 存在 CPU，只有 128 維投影與選出的 bank 上 GPU；打分每 16 張一塊。candle `--quick`：VRAM 峰值 1.44 GB、RAM 峰值 6.4 GB |

## 5. §6 Experimental Setup

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 5.1 | 所有方法共用同一份切分 | ✅ | ⬜ | 確認所有方法都透過 `load_split()` 讀資料 | `run.inputs()` 一律經 `load_features()` 讀取，其 index 與 `load_split()` 相同（第 3 階段已驗證） |
| 5.2 | 前處理只用 ImageNet 固定統計值；μ / Σ 只用 train 擬合 | ✅ | ⬜ | 閱讀 `fit` 的輸入 | `fit` 只收 train（`run_class` 有 assert train 全為 normal）；Tier 1 用 `normalize=False`，Tier 2/3 的特徵只用 ImageNet 固定統計值 |
| 5.3 | test 只在 `run.py` 最後讀取一次 | ✅ | ⬜ | 搜尋所有讀取 test 的地方 | `run_class` 在所有模型擬合、τ 決定後才呼叫 `inputs(obj, ["test"])`，每類一次；`--search`、`--latency` 都不讀 test |
| 5.4 | 超參數表：隨機維度、coreset 比例、特徵層，以及固定值（ε、3×3、128） | 🔶 | ⬜ | 與 `run.py` 的設定對照 | 搜尋範圍與結果在 `results/stage6_main/search_val.csv`；固定值 ε = 0.01、3×3、128 維投影寫在 `run.py`。選定：T2 200 維、layer 2+3+4；T3 coreset 1%、layer 2+3。論文表格待草稿 |
| 5.5 | 超參數只在 val 上搜尋，以 val PR-AUC 為標準 | ✅ | ⬜ | 閱讀搜尋程式 | `run.py --search` 只用 train / val、seed 0；`choose_configs()` 取 12 類平均 val PR-AUC 最高者（T2 0.664、T3 0.811）。T2 最佳值落在範圍上限 200 維 |
| 5.6 | 記錄 torch / torchvision 版本、CPU、RTX 4070 12 GB、RAM 32 GB | ✅ | ⬜ | 對照 `requirements.txt` | `results/stage6_main/environment.csv`：Python 3.12.8、torch 2.15.0+cu132、RTX 4060 Ti 8 GB、RAM 32 GB（本機，非 RTX 4070） |
| 5.7 | 指標 AUROC、PR-AUC、F1@τ、F2@τ、FPR@τ、prec@π，並寫明選用理由 | 🔶 | ⬜ | 用一個小例子手算驗證指標函式 | `run.metrics()` 輸出 AUROC、PR-AUC、F1、F2、FPR、prec@1%/5%；`_check_metrics()` 以手算例子 assert，每次執行都會跑。選用理由待草稿 |
| 5.8 | 延遲量測：GPU、batch 1、warm-up 10 張、包含特徵抽取；訓練時間為 12 類合計 | ✅ | ⬜ | 閱讀計時程式 | `results/stage6_main/latency.csv`（pcb1 val、GPU、batch 1、warm-up 10、計時 50 張、含特徵抽取）：T1 0.12、T2 7.38、T3 11.72 ms 中位數；訓練時間 12 類合計見 `summary.csv` 的 `fit_s_total_mean`（T3 114 s） |
| 5.9 | 3 個 seed（0 / 1 / 2）；Tier 1 的 std = 0 有註明 | ✅ | ⬜ | 檢查 `results/stage6_main/*.csv` 的 seed 欄位 | `metrics.csv` 有 seed 0 / 1 / 2；T1 為確定性方法只跑 seed 0，`summary.csv` 中 n_seeds = 1、std = 0 |

## 6. §7 Results

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 6.1 | 主結果表欄位齊全（AUROC、PR-AUC、F1、F2、FPR、prec@π、ms/image、訓練時間） | ✅ | ⬜ | 對照 CSV | `metrics.csv`（AUROC、PR-AUC、F1、F2、FPR、prec@1% / 5%、fit_s）+ `latency.csv`（ms / image）+ `summary.csv`（訓練時間合計） |
| 6.2 | 12 類平均 + 各類結果，mean ± std，每欄最佳值加粗 | 🔶 | ⬜ | 從 CSV 重算幾格 | `summary.csv` 為 12 類平均的 mean ± std，`metrics.csv` 為各類結果；加粗待報告。12 類 PR-AUC：T1 0.702、T2 0.801 ± 0.007、T3 0.885 ± 0.002 |
| 6.3 | T2 vs T3 的 paired bootstrap 95% CI（類內重抽、1,000 次） | ⬜ | ⬜ | 閱讀 bootstrap 程式 | |
| 6.4 | 12 類配對的 Wilcoxon 檢定（seed t-test 為補充） | ⬜ | ⬜ | 閱讀程式與輸出 | |
| 6.5 | 解釋 T2 贏或輸 T3 的原因，以及準確度與成本的取捨 | ⬜ | ⬜ | 閱讀草稿 §7 | |
| 6.6 | Ablation 1：Mahalanobis vs 歐氏 kNN vs cosine | ✅ | ⬜ | 確認只改了一個因子 | `results/stage7_ablation/ablation_summary.csv` A1：只換位置分數（同一份 200 維特徵、同 τ 規則）。12 類 PR-AUC：Mahalanobis 0.801 → 歐氏 1-NN 0.734（Δ −0.067）、cosine 1-NN 0.686（Δ −0.115），皆超過 seed std |
| 6.7 | Ablation 2：best-F2 vs best-F1 vs percentile vs 審核預算 | ✅ | ⬜ | 確認只改了一個因子 | A2：同一個模型只換 τ 規則；PR-AUC 不變，比較 F2 / FPR / prec@1%。best-F2：recall 0.82、FPR 0.33、prec@1% 0.037；best-F1：F2 −0.128、prec@1% 0.090；val-normal p99：FPR 0.019、prec@1% 0.513、recall 0.29；審核預算（val 前 10%）：prec@1% 0.340。皆超過 seed std |
| 6.8 | Ablation 3：污染 0 / 1 / 2%，並說明為什麼無法做到 5% / 10% | ✅ | ⬜ | 確認混入的是 contam pool，不是 val 或 test | A3：只從 contam pool 抽異常加入 train（seed 控制），val / test 不動。每類加入 4–8 張（1%）與 8–16 張（2%）；PR-AUC 0.801 → 0.797 → 0.794（Δ −0.007，未超過 seed std）。每類 pool 只有 20 張，大類別上限約 2.5%（20 / 805），所以做不到 5% / 10% |
| 6.9 | Ablation 4：max vs top-1% mean（H3） | ✅ | ⬜ | 確認只改了一個因子 | A4：同一個模型只換聚合方式。top-1% mean 0.799（Δ −0.002）、平滑後取 max（σ = 4）0.796（Δ −0.004），皆未超過 seed std → H3 不成立 |
| 6.10 | Ablation 5：patch 層級 vs 全域 pooled 特徵（H2） | ✅ | ⬜ | 確認只改了一個因子 | A5：同一組維度，先對全部位置平均再擬合單一高斯。PR-AUC 0.801 → 0.771（Δ −0.029，超過 seed std）；9 類 0.785 → 0.747（Δ −0.038）→ 支持 H2 |
| 6.11 | 每個 ablation 的 Δ 都有和 seed std 比較 | ✅ | ⬜ | 檢查表格 | `ablation_summary.csv` 每個設定都有 `*_delta` 與 `*_beyond_seed_std`（|Δ| 大於 full 與該設定兩者的 seed std）；full 設定與第 6 階段的 T2 結果逐位相同（candle seed 0 已核對） |
| 6.12 | 結果變差的設定也有報告並解釋 | 🔶 | ⬜ | 閱讀草稿 | 變差的設定已在 CSV（A1 兩種 kNN、A5 全域特徵、A3 污染）；解釋待草稿 |

## 7. §8 Analysis

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 7.1 | 至少 3 個失敗案例，其中至少 1 個 FP、1 個 FN | ⬜ | ⬜ | 對照 CSV 中的 label 與 s(x) | |
| 7.2 | 每個案例都有影像 ID、s(x)、τ、與 τ 的距離、熱圖 | ⬜ | ⬜ | 開圖與 CSV 對照 | |
| 7.3 | 共同盲點與第一個想嘗試的修正 | ⬜ | ⬜ | 閱讀草稿 | |
| 7.4 | H2 佐證：依 mask 面積分組的偵測率（T2 vs 全域特徵版本） | ⬜ | ⬜ | 閱讀程式與輸出 | |
| 7.5 | Tier 2 熱圖疊圖 + pixel AUROC | ⬜ | ⬜ | 開圖；閱讀 pixel AUROC 程式 | |
| 7.6 | 至少一張被標記樣本的具體解釋 | ⬜ | ⬜ | 閱讀草稿 | |
| 7.7 | Trust 討論：工程師是否會依熱圖採取行動 | ⬜ | ⬜ | 閱讀草稿 | |
| 7.8 | 以 1% 盛行率換算每 10,000 件的漏檢數與誤報數（T2 vs T3） | ⬜ | ⬜ | 用 TPR / FPR 自行計算並比對 | |
| 7.9 | 誤報數換算成人工複檢工作量，並連結到利害關係人 | ⬜ | ⬜ | 閱讀草稿 | |

## 8. 進度報告 `reports/progress.md`（繁中）

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 8.1 | Running Example 框（情境、研究問題、正常與異常樣本、模型與 s(x)、operational action） | ⬜ | ⬜ | 對照 PDF p.8 | |
| 8.2 | 相關研究比較表，註明「不同切分，僅供參考」 | ⬜ | ⬜ | 對照 PDF p.9；逐筆查證引用 | |
| 8.3 | 資料與不平衡 | ⬜ | ⬜ | 對照 PDF p.10 | |
| 8.4 | 主結果表 + CI + Reading | ⬜ | ⬜ | 對照 PDF p.14 | |
| 8.5 | Ablation 表 + Reading | ⬜ | ⬜ | 對照 PDF p.15 | |
| 8.6 | 失敗案例表 + τ + Shared pattern | ⬜ | ⬜ | 對照 PDF p.16 | |
| 8.7 | 實務影響 | ⬜ | ⬜ | 對照 PDF p.17 | |
| 8.8 | H1–H3 的驗證結果（各一句話附數字） | ⬜ | ⬜ | 數字與 CSV 對照 | |
| 8.9 | 是否值得繼續（證據強弱、缺口、換題風險、建議） | ⬜ | ⬜ | 閱讀 | |
| 8.10 | 所有數字都能在 `results/stage*/*.csv` 找到對應，沒有手動填寫的數字 | ⬜ | ⬜ | 隨機抽 5 個數字回查 CSV | |

## 9. 英文草稿 `reports/sec3-8.md`

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 9.1 | §3–§8 的章節順序與評分表一致 | ⬜ | ⬜ | 對照 PDF p.7 | |
| 9.2 | 圖表都有編號、標題，並在內文中引用；座標軸有標籤與單位 | ⬜ | ⬜ | 逐張檢查 | |
| 9.3 | 引用格式可轉成 BibTeX，使用 \citet / \citep | ⬜ | ⬜ | 檢查引用 | |

## 10. 程式與檔案結構

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 10.1 | `src/` 下六個檔案都存在，職責與計畫一致 | 🔶 | 🔶 | 列出 `src/` 並閱讀各檔開頭說明 | 目前有 `data.py`、`features.py`、`eda.py`；Review（2026-10-30，本機重現）：目前 3 / 6 個檔案，職責與計畫一致；目前 5 / 6 個檔案（新增 `methods.py`、`run.py`）；仍為 5 / 6 個檔案；`run.py` 新增 `--search`、`--latency`、`summarize()` |
| 10.2 | `features.py` 每類、每層分開快取（float16、原始解析度） | ✅ | ✅ | 檢查 `cache/` 的檔名、dtype、shape | 12 類 × 3 層（layer2/3/4）共 36 個 .npy + 12 個 index.csv，約 15 GB；shape、dtype、列順序與切分逐類 assert 通過；重新計算抽樣影像，差異只有 float16 捨入誤差；Review（2026-10-30，本機重現）：本機重建：12 類 × 3 層 shape、float16、index 與 `load_split` 一致，共 14.16 GB；capsules / pcb4 重新計算 3 張，相對誤差 ≤ 4.4e-3（float16 + TF32 捨入） |
| 10.3 | `run.py --quick`（1 類、1 seed）端到端跑通，並記錄 VRAM / RAM 峰值 | ✅ | ⬜ | 自行重跑一次 | `python src/run.py --quick`（candle、seed 0）144 s 跑完；VRAM 峰值 1.44 GB、RAM 峰值 6.4 GB；輸出 `results/stage5_quick/metrics.csv` 與 `scores/*.npz` |

## 11. 驗證

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 11.1 | assert train 裡沒有 anomaly | ✅ | ✅ | 閱讀 `check_split()` | Review（2026-10-30，本機重現）：`check_split()` 有 assert，本機重跑通過 |
| 11.2 | assert 各 split 的 ID 沒有交集 | ✅ | ✅ | 閱讀 `check_split()` | Review（2026-10-30，本機重現）：`check_split()` 兩兩交集 assert，本機重跑通過 |
| 11.3 | eda 與 τ 的選擇都沒有用到 test | ✅ | 🔶 | 搜尋程式中讀取 test 的地方 | EDA 部分完成（同 3.1）；τ 的部分待第 6 階段；Review（2026-10-30，本機重現）：EDA 部分 ✅（同 3.1）；τ 待第 6 階段；τ 部分：同 4.6 |
| 11.4 | `methods.py` 合成資料自檢：三個方法的 AUROC 都 > 0.9 | ✅ | ⬜ | 執行 `python src/methods.py` | `python src/methods.py`：T1 / T2 / T3 AUROC 皆為 1.000 |
| 11.5 | Tier 3 的平均 AUROC 與查證過的文獻數字（註明出處）相差 ≤ 5 個百分點 | ✅ | ⬜ | 查證文獻數字 | T3 平均 image AUROC 0.902；Zou et al. 2022 Table 6（PatchCore、WRN50、VisA 1-class）為 92.4，差 2.2 個百分點。注意對方使用官方切分 |
| 11.6 | 所有引用文獻都已查證為真實存在，沒有捏造 | ⬜ | ⬜ | 逐筆到出版社或會議頁面查證 | |

## 12. README 與收尾

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 12.1 | README：環境建立（含 CUDA torch）、下載資料、一鍵重跑、結果摘要表、CC BY 4.0 | ⬜ | ⬜ | 依 README 在乾淨環境重跑 | |
| 12.2 | README 說明如何用 `data.py` 和 `visa_split.csv` 重現資料，並註明「資料集處理方式待定」 | ⬜ | ⬜ | 閱讀 README | |
| 12.3 | 資料集仍被 ignore，沒有 commit 也沒有推上 GitHub | ⬜ | ⬜ | 同 0.8 | 最後怎麼處理由組員決定 |
| 12.4 | 每個 commit 訊息都先經過確認 | ⬜ | ⬜ | `git log` | |

## 13. EDA 發現的問題（計畫「EDA 發現的問題與處理方式」）

| # | 項目 | 實作 | Review | Review 方式 | 證據 / 備註 |
|---|---|---|---|---|---|
| 13.1 | 拍攝條件不均已量化：每類 train normal 與 val anomaly 落在小群的比例 | ✅ | ✅ | 閱讀 `acquisition_modes()`；確認只用 train/val | `results/stage4_eda/eda_acquisition_modes.csv`：cashew 35.8→95.0%、pcb1 20.1→95.0%、pcb2 23.2→70.0%；新增 `gap_pp`、`affected` 欄位，由 `AFFECTED_GAP_PP = 40` 判定；Review（2026-10-30，本機重現）：本機重算 `eda_acquisition_modes.csv` 完全相同；只用 train / val |
| 13.2 | 受影響類別的判定規則（差距 ≥ 40 個百分點）與名單在第 6 階段前寫入計畫，之後不再更改 | ✅ | ✅ | 用 `git log` 確認規則寫入時間早於第 6 階段的結果 | 名單：cashew、pcb1、pcb2；規則寫在計畫的「問題 1」；Review（2026-10-30，本機重現）：`AFFECTED_GAP_PP` 於 5d9d868 加入，尚無第 6 階段結果；本機重算名單相同（cashew、pcb1、pcb2）。計畫檔不在 repo，無法直接查證 |
| 13.3 | `acquisition_causes()` 確認原因（亮度、旋轉 180°），只用 train/val | ✅ | ✅ | 閱讀程式與 `eda_acquisition_causes.csv` | `results/stage4_eda/eda_acquisition_causes.csv`：cashew、pcb2 異常影像曝光較亮（灰階 62.7 vs 53.1/55.8；97.7 vs 86.8/91.0）；pcb1 有 45% 異常旋轉 180°（正常約 7%），其餘原因未明；只讀 train/val 影像；Review（2026-10-30，本機重現）：數字與 CSV 一致；pcb1 正常整體旋轉比例 = 162 × 36.4% / 805 ≈ 7.3% |
| 13.4 | 主結果表標註受影響類別，另報排除後的 9 類平均，並檢查 T1/T2/T3 排名是否改變 | 🔶 | ⬜ | 從 CSV 重算 9 類平均 | `summary.csv` 有 12 類與 9 類兩個版本；兩者排名皆為 T3 > T2 > T1 > T0，沒有改變（9 類 PR-AUC：T3 0.874、T2 0.785、T1 0.655）。表格標註待報告 |
| 13.5 | Ablation 5（H2）與 H1 的比較同時報告 12 類與 9 類版本 | 🔶 | ⬜ | 檢查兩個版本的表格 | A5 與 A1 的 12 類、9 類版本都在 `ablation_summary.csv`，結論方向一致；H1 的 12 / 9 類比較待第 8 階段 |
| 13.6 | 熱圖最高分位置落在 mask（擴 8 px）內的比例，依類別與 tier 報告；只在 test 評估之後做 | ⬜ | ⬜ | 閱讀程式；抽幾張圖目視比對 | |
| 13.7 | §4、§8 寫入 EDA 發現的問題；§9 的部分整理在 `progress.md` 的缺口與限制 | ⬜ | ⬜ | 閱讀草稿與 `progress.md` | 包含問題 1–5 |
