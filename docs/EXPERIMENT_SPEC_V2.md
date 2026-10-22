# HW2 v2：以本機 NLP 表徵偵測 coding-agent 失敗

日期：2026-10-23。狀態：新實驗規格，尚未實作或產生 v2 結果。
本文件取代舊實驗設計的方向；現有 PDF、投影片、CSV 與驗證紀錄仍是 v1，不能當作 v2 成果。

## 1. 要回答的問題與設計決策

研究問題：在相同 coding-agent 任務上，使用 action、observation 與 patch 的文字／程式碼表徵，能否比 24 個統計特徵更有效地辨識失敗，尤其減少對成功任務的誤報？

保留既有 SWE-agent 題目，先做有明確停止條件的探索實驗。主方法改為 frozen encoder + normal-only kNN distance，完整比較 TF-IDF、語意向量及程式碼向量。不得預設 embedding 一定較好，也不把 decoder LLM 或付費 API 加入流程。

三條可選路線的取捨：

| 路線 | 優點 | 問題 | 決定 |
|---|---|---|---|
| 24 統計量 + 換更大分類器 | 最省成本 | 原始語意仍遺失 | 僅保留為對照 |
| TF-IDF / MiniLM / CodeBERT + 小型偵測器 | 可利用真實文字，不需生成 API，可快取 | 小資料、領域轉移、向量未必懂正確性 | 本次主線 |
| 改用有稀有異常標籤的系統日誌 | 更符合 HW2 anomaly detection 前提 | 題目與資料改變 | 若主線不適合作業，提出 HDFS_v1 替代題目，不悄悄換資料 |

v1 的問題不只模型：284 筆只有 54 成功、230 失敗；舊測試集 150/162 是失敗，always-alert 已有 F2 0.984。seed 42 AE 的 TN/FP/FN/TP = 3/9/9/141；只少審 12 筆卻漏 9 筆失敗。高 AP 不能支撐可用性。

**資料適用性限制：失敗占 81%，不應稱為天然稀有異常資料。** 不把成功改名為「異常」來滿足比例，也不大量刪失敗做漂亮測試。若維持這個資料源，論文須明示這是 normal-reference failure screening 的探索研究，HW2 稀有事件情境仍有不足。NLP 改善不了這個資料定義問題。

## 2. 資料、標籤與可見時間

- 單位：一個已完成的 agent trajectory，分類為 Text/Logs。
- y=0：官方 resolved；y=1：官方 unresolved。標籤來自現有官方結果，不由文字猜測。
- 使用時間：agent 結束後、外部官方 correctness evaluation 結果未知時。所有完整軌跡模型只宣稱 task-end screening。
- 保留全部 284 筆，原始 URL、SHA-256、instance ID 與官方標籤稽核。
- 模型可讀：trajectory[].action、trajectory[].observation、agent 產出的 final patch。
- 預設不讀 thought/response（避免重複 action 和自我宣稱）；如未來研究此欄，另立預註冊 ablation。
- 禁止讀：resolved、官方測試結果、gold patch、gold tests、官方 grader 輸出、exit_status、instance_id、repo ID。agent 在自己的工具 observation 中看到的測試輸出允許使用，但須確認不是事後 evaluator 注入。
- 先人工稽核至少 10 筆正常與 10 筆失敗的 JSON 欄位及資料時間線。若不能確定 observation 的來源，停止該欄使用並記錄原因。
- 固定正規化：移除 ANSI escape；絕對工作目錄改為 <WORKDIR>；instance ID、repository 根目錄名改為 placeholder；保留程式符號、錯誤種類、相對檔案路徑與行順序。不刪所有數字或所有 error 詞。
- 缺少 patch 使用 empty-patch mask；向量設零，不把任意 placeholder 文字當成真實 patch。log 完全缺失則明示 missing mask，不靜默丟資料。

## 3. 表徵：文字確實進模型

### R0：原有 24 個統計量

保持原本定義，作為歷史基準。所有 scaler 僅在各方法允許的 fitting rows 擬合。

### R1：傳統 NLP（必要基準）

- action+observation 依步驟順序組成 log 文件；patch 另作文件。
- 每個 channel 使用 character TF-IDF，ngram 3–5，min_df=2，sublinear TF，最多 20,000 特徵，L2 normalization。相同次數的 token 以字典序決定截斷。
- 字元 ngram 適合保留程式符號、命令與 traceback 子字串；不宣稱其具備語意推理能力。
- 詞彙與 IDF 都只在 fitting rows 學習。T2 僅成功訓練資料；監督式參考可用完整 training partition。
- 全零 OOV 文件要計數並公開；不得用 test 補 vocabulary。kNN 使用 log/patch 等權 cosine distance，零向量依 R3 的缺失距離規則處理；監督式模型將兩個 L2-normalized channels 各乘 1/sqrt(2) 後拼接。

### R2：本機語意 encoder

- `sentence-transformers/all-MiniLM-L6-v2`，固定權重、不 fine-tune，輸出 384 維句向量。
- action、observation 分別切 chunk，保留 step ID。每 chunk 最多 224 個內容 tokens，overlap=32；加入標記後仍不得超過模型 256 token 預設上限。special tokens 納入檢查。
- 每步先平均該步各 chunk，L2 normalize；整條軌跡對各步等權平均，再 normalize，避免單筆巨長 observation 決定整條向量。
- 同時保留 step embeddings，供定位異常步驟與後續順序 ablation；預設 run-level 均值不聲稱學到順序。
- 不把整條長軌跡直接丟入模型造成尾部被靜默截斷。第一版不設全軌跡截斷；若資源不足，停止並回報耗時與容量，再明示修訂。

### R3：雙通道語意 + 程式碼（預先指定主表徵）

- Log：R2 的 MiniLM run embedding。
- Patch：`microsoft/codebert-base` frozen token encoder。依 diff hunk 分段，最多 480 個內容 tokens，overlap=64，加入 special tokens 後 <=512。
- 每 chunk 對非 padding、非 special tokens 的最後一層 hidden states 作 mean pooling；chunk 按新增內容 token 數加權平均，L2 normalize，產生 768 維 patch embedding。
- 這是本實驗的 pooling 設計，不宣稱 CodeBERT 原生就有經校準的句向量或能驗證 patch 正確性。
- 使用等權 channel distance：d=(d_log+d_patch)/2。不是直接拼接後讓 768 維自動主導 384 維。
- 兩個 patch 都缺失時 d_patch=0；只有一個缺失時 d_patch=1；都有 patch 時 d_patch=(1-cosine)/2。log 同理處理全空情況。相似度裁切至 [-1,1]。
- encoder revision、tokenizer revision、套件版本與每筆原文 hash 都寫進 cache manifest；模型下載一次後在本機計算，不呼叫生成 API。

## 4. 方法與真正「訓練」的部分

| 角色 | 表徵 + 方法 | 學習資料 | 用途 |
|---|---|---|---|
| T1 | 既有 error/repetition heuristic | 無模型擬合 | 與舊結果接軌 |
| T2 primary | R3 + kNN normal-distance | 只存成功 fitting embeddings | 以成功軌跡的近鄰距離定義異常 |
| T3 strong reference | R3 concatenation + L2 logistic regression | 正常與失敗 fitting rows | 測試有監督語意分類能否勝過正常參考法 |
| Necessary NLP control | R1 + L2 logistic regression | 正常與失敗 fitting rows | 確認 encoder 是否真優於傳統 NLP |
| Necessary tabular controls | R0-clean + Isolation Forest / inductive ECOD variant | 僅成功 fitting rows | 排除原 AE 選擇不佳造成的假 NLP 收益 |
| Legacy controls | R0 AE、R0 XGBoost | 依 v1 normal-only / mixed 規則 | 衡量新增文字資訊 |
| Context controls | always-alert、never-alert | 無 | 揭露不平衡與決策成本 |

T3 是競爭性 supervised reference，不標為已證實 SOTA。原 XGBoost 不移除，以免僅因更換對手就宣稱改善。

T2 score 為離測試向量最近 k 個正常 fitting runs 的平均 d，k 候選固定 {1,3,5}。不能把 validation/test 加入近鄰索引。T3 的 C 候選 {0.01,0.1,1,10}，class_weight 固定 balanced；log/patch 個別 normalize 後各乘 1/sqrt(2) 再拼接。R1 reference 使用相同 C grid。

Frozen encoder 不在 284 筆上重新訓練。實際學習包括 TF-IDF vocabulary/IDF、監督分類器權重、正常參考集合及 validation threshold；AE 對照仍有梯度訓練。不得用 encoder inference 冒充 fine-tuning。

## 5. 切分與模型選擇

舊 test 已經看過，不能重切後聲稱取得全新獨立 test。v2 定位為凍結方案後的 exploratory cross-validation；獨立確認必須是尚未用於設計的新來源／新 run，另立 acquisition manifest 後再取得標籤與評估。

主評估：固定 split seed=42，5-fold stratified outer CV，每筆恰有一個 outer-test prediction。每輪剩餘 80% 再按 label stratified，以 seed=100+fold 切 25% 作 validation，約為全資料 60/20/20。所有方法共用完全相同 manifest。相同 instance 的多個 chunks 絕不能跨 fold；若未來加入其他 agent runs，須把同一 instance 的所有版本綁在同組。

1. 所有 fit / vocabulary / PCA（若日後加入）只讀當輪 training；T2/AE 只能讀其中成功樣本。
2. 只用 validation 選 k 或 C 及 threshold。選完不再把 validation 加回 fitting，以免 threshold 對應不同模型。
3. 對每個候選 k/C 先依下一點的 validation policy 決定 threshold，再以該點的 validation balanced accuracy 選模型；同分取較小 k/C。選出的模型與 threshold 一起凍結。
4. 在已選模型上，部署討論用 threshold：枚舉 validation unique scores、及 +/-infinity，預測 anomaly 當 score>=tau；在 empirical FPR<=20% 的候選中最大化 failure recall，同分取較高 threshold。
5. 20% 是事先設定的探索目標，**不是小樣本保證**。每 fold 約 10 個正常 validation，無法可靠認證低 FPR；顯示分母及區間，不稱為 SLA。
6. optimization seeds={42,123,456}；無隨機性的 encoder/kNN 只算一次，不將複製的結果當三次獨立實驗。AE 等隨機方法每 fold 跑三個 seed。
7. 每個方法及 ablation 都完整跑全部 folds，一次輸出所有結果。不得因 outer-test 結果再更換設定或只報最好的 fold。

另做 repository-held-out 敏感度分析：各 repo 輪流測試，剩餘資料內切 validation；無法提供兩類或不足正常 fitting rows 的輪次標記不可估，不能默默刪除。它是 domain shift 診斷，不能與主 CV 混成更大的 n。

## 6. 評估主表：混淆矩陣是必要輸出

主決策指標是 fixed-validation-threshold 的 balanced accuracy、FPR、failure recall，以及「未被警報者中有多少真的成功」。排序指標 ROC-AUC、AP 與 AP 的 prevalence baseline 一起列，F1/F2 僅輔助。

每方法必須輸出：

- 每 fold 的 TN/FP/FN/TP 與 raw-count confusion matrix；rows=true success/failure，cols=predicted success/failure。
- 每個 optimization seed 的 pooled out-of-fold confusion matrix，總數必須是 284；不能把三個 seed 加成 852 人次。seed42 圖為預定主圖，其餘置附表。
- 原始 counts 與 row-normalized heatmap 並列，標示 success/failure，避免僅放百分比。
- balanced accuracy=(recall+specificity)/2；FPR=FP/(FP+TN)。
- unflagged success fraction=TN/(TN+FN)、unflagged coverage=(TN+FN)/N、review fraction=(TP+FP)/N。分母為零時報 NA。
- ROC/PR、FPR-recall tradeoff、risk-coverage 曲線。曲線使用 outer predictions，只作描述，不用來重新選 threshold。
- 各 fold mean/SD、class-stratified paired bootstrap 2,000 次的 pooled OOF difference CI，seed2026；重採樣單位為 instance，各方法使用同樣 indices。明示共享 fitting folds，CI 僅條件式描述；不把 folds/seeds 當獨立抽樣做顯著性誇大。
- 模型下載、文字處理、encoder 計算、classifier fitting 與 inference 分開計時；既報 batch throughput，也報 end-to-end per-run latency 與峰值記憶體。

事先設定的實用性門檻：OOF failure recall>=90%、FPR<=20%、unflagged success fraction>=80%、coverage>=5%，四者同時達標才討論放行價值；區間過寬仍不宣稱可靠部署。門檻未達則結論為「不能作為自動放行依據」。這是驗收假設，不是結果保證。

## 7. 控制實驗與假說

H1：同一 kNN、同一 folds 下，R3 比 R0 在上述 threshold policy 下提高 balanced accuracy / 降低 FPR，且 failure recall 沒有明顯犧牲。
H2：R3 比 log-only R2 更好，表示 patch 通道有額外資訊。
H3：normal reference 的污染會傷害辨識能力。

必要 ablation：

1. **表徵**：R0 / R1 / R2 / R3 共用 kNN 與 k grid；R0 在正常 fitting 資料 standardize，使用 Euclidean distance；R1 cosine；不同幾何是表徵設計的一部分，必須披露，不能聲稱只差 encoder。
2. **通道**：log-only / patch-only / equal-weight dual；保持 k、split 與 calibration policy 一致，各自用 validation 選 k，列出所選值。
3. **AD-specific 污染**：正常 fitting reference 固定大小，替換 floor(n*p) 個正常為 training failures，p={0,.05,.10}；失敗採樣 seed={42,123,456}、nested subsets、記錄實際比例。凍結 clean k 及 encoder；主分析保留 clean tau，另表列依同一 validation policy 重校準 tau，區分 detector 與 threshold 影響。

輔助診斷而非主線：last-step-only vs 全軌跡；固定遮掉 error/failed/traceback 字詞後重算 embeddings，檢查收益是否僅來自關鍵字。需在正式計算前凍結字詞清單，不按 test case 挑詞。

## 8. 錯誤分析與可解釋性

依 OOF predictions，以固定規則挑成功誤報分數最高、失敗漏報分數最低、最靠近 threshold 的錯誤，各一例；不足某類則明示，不捏造案例。

每例顯示 instance ID、官方標籤、prediction、score、tau、nearest normal reference ID，以及 log/patch distance。引用原始 action/observation/hunk，追蹤到 SHA-256。展示逐步與最近正常 step 的 distance，並對一個 step 或 hunk 做 leave-one-out 重算 score，作模型敏感度解釋；不得稱因果證據。若零錯誤，檢查 leakage，再用固定規則列最低 margin 正確案例。

embedding 看起來接近不等於 code 正確；真正的 patch 語意錯誤仍可能需要執行測試或形式驗證，這是方法的明確能力邊界。

## 9. 交付與停止條件

預定新增檔案（目前尚未存在，以下不是可執行指令）：

- `configs/experiment_v2.json`：一次凍結 models、revision、seed、grids、chunking、metrics。
- `Dataset/v2/splits.json`、`text_manifest.jsonl`：文字來源、可見時間稽核、原文 hash 與 fold IDs。
- `tools/build_text_embeddings.py`：本機 encoder + cache manifest；不得呼叫遠端 inference endpoint。
- `Model/experiment_v2.py`：全部方法 OOF prediction、scores、tau、confusions。
- `Model/ablations_v2.py`：三組必要 ablation。
- `Model/outputs/v2/`、`Model/tables/v2/`、`Model/figures/v2/`：與 v1 分開，含所有方法的混淆矩陣。

驗證包含：fold disjoint、chunk 不跨樣本、label 不進文字、normal-only fitting IDs、TF-IDF fit IDs、模型 cache revision、score->prediction->confusion->metrics 一致、284 OOF coverage、缺值與超長文字可追溯、一次小型資料 smoke run。

執行順序：資料欄位稽核 -> 凍結 config/splits -> 少量無標籤速度與記憶體 probe -> 完整表徵快取 -> 全部主模型 -> 必要 ablation -> OOF 誤差分析 -> 論文重寫。資源 probe 只決定 batch size，不看評分選 representation。

若 NLP 不優於 TF-IDF/統計基準，照實保留；若所有方法仍不達實用性門檻，不再透過 test 搜尋更多模型。要更貼合作業，可另外提出 HDFS_v1 block-level normal/anomaly 日誌題目：從官方原始 logs/labels 建立 block 群組，保留自然比例，防止共享原始 log 跨 split，重新凍結完整 spec。這是資料題目變更，不是現有數值的補丁，也不是本文件已授權的自動替換。

## 10. 核實的來源

- MiniLM model card（384 維、預設 256 word pieces）：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- CodeBERT 模型與 feature extraction：https://huggingface.co/microsoft/codebert-base
- CodeBERT 原論文，Findings of EMNLP 2020：https://aclanthology.org/2020.findings-emnlp.139/
- Loghub HDFS 官方來源與 block-level labels：https://github.com/logpai/loghub/blob/master/HDFS/README.md

以上引用支持模型／資料來源；不支持任何尚未執行的 v2 效能主張。

## 11. 納入使用者提供的 tabular / conditional 建議

本節是本次規格的一部分，與 NLP 主線共用第 5–6 節的 split、calibration、metrics，不另挑有利測試集。模型推薦星等與 N=200/1000 等分界不作為實驗證據；真正限制是每輪僅約 32 個正常 fitting rows，而非表面上的 N=284。

### 必做：合理的 tabular baseline

新增 R0-clean，並保留未改動的 R0 legacy 結果，避免前處理改善被誤認為 NLP 的貢獻。

- 固定移除 `num_tool_calls`（目前與 `num_steps` 相同），留下 23 欄。若重新取得資料發現不相同，先修訂 extraction contract，不默默改規則。
- 對以下非負長尾欄位取 log1p：num_steps、num_unique_tools、max_repeated_tool_streak、num_error_observations、consecutive_errors_max、stalled_turn_count、test_suite_run_count、total_cost_usd、tokens_sent、tokens_received、tokens_per_step、patch_char_length、patch_lines_added、patch_lines_removed、patch_num_files_changed、patch_hunk_count。
- 二元欄位保留 0/1；entropy、error rate、edit ratio 不 log。需要距離／重建時，非二元欄位以正常 fitting median/IQR scaling；IQR=0 改用 fitting SD，SD 也為 0 則 scale=1。
- 不根據全資料相關係數刪特徵。訓練期報告相關性；cost/token 多重計權仍是限制。constant dimensions 依 fitting 資料標記並公開。
- Isolation Forest：200 trees，max_samples=min(256,n_normal_fit)，max_features=1.0，bootstrap=false，seeds 42/123/456；以負 normality score 作 anomaly score，不使用套件內建 contamination threshold，套用共同 validation policy。
- ECOD：必須採 **inductive ECOD variant**，凍結正常 fitting ECDF 及 skewness。查閱的 PyOD 實作會把待評估 batch 與 training 串接重算 ECDF，故不能直接用它的 batch decision_function 當作本 spec 的固定參考分數。
- 具體 ECOD variant：每欄 pL=(1+count(train<=x))/(n+1)，pR=(1+count(train>=x))/(n+1)，UL=-log(pL)，UR=-log(pR)；依 fitting skewness 正負選 UR/UL，零 skewness 選 UL+UR；每欄取 max(UL,UR,selected)，最後各欄加總。此有限樣本 smoothing 與 frozen scoring 是明示修改，不冒稱完全重現原始 ECOD。保留每欄貢獻供解釋。
- 必驗證同一樣本獨自評分、放進不同 batch、改 batch 順序後分數不變；不得讓 test composition 改變參考分布。

### 有限、預先指定的延伸：DAE 與條件殘差

這兩項是 secondary experiments，不取代 NLP 主方法，不在看過 OOF 結果後才決定要不要報告。若執行階段採用，必須在第一次完整 OOF 前於 config 設 enabled=true，否則明示未執行，不事後挑選。

**Shallow DAE**：R0-clean 23→16→4→16→23，ReLU hidden，線性 numeric outputs + binary logits；訓練輸入獨立遮罩率 0.1，target 為未污染輸入，評分時不遮罩。固定 AdamW lr=0.001、weight_decay=0.0001、150 epochs、batch16、三 seeds；不以 validation 梯度更新或 early-stop 擬合。

每一 feature 的損失：numeric 使用 Huber delta=1，binary 使用 BCE-with-logits；先在 behavior 15 欄、patch 8 欄各自取平均，再等權平均兩組。這同一規則用於 training loss 與 anomaly score。binary 欄位確切為 reproduce_script_created、last_step_error、patch_is_empty、patch_touches_tests。必須承認 BCE/Huber 的數值尺度選擇是模型假設，不是天然已校準的失敗機率。

**Conditional baseline**：先用線性正則化模型驗證跨通道關係，避免約 32 筆正常資料立刻訓練 conditional MLP。輸入 R0-clean 的 15 個 behavior，預測 8 個 patch targets；numeric 用 ridge alpha=10，binary 用 L2 logistic C=0.1。若 fitting binary target 只有一類，改用 Laplace-smoothed 常數機率 (positives+1)/(n+2)。輸入與 numeric targets 依前述 fitting-only transform，score 為 8 targets 的平均 Huber/BCE，沿用共同 threshold policy。與直接 patch-only anomaly score 比較，檢查條件關係是否有價值。

不得把這個 residual 命名為 causal anomaly、reward hacking 或欺騙偵測。沒有任務難度／需求條件時，殘差可能只是工作量差異。issue description 可作將來的 context encoder，但本輪先不加：需另外核實來源、task-start 可見性、gold patch exclusion，並凍結新的對照設計。

**暫不做**：VAE、深層 AE、學習式 ensemble，以及一次加入所有 LOF/COPOD/Robust PCA 變體。它們會增加搜尋空間，尚無理由優先於必要 NLP、tabular 基準與控制實驗。

### 標籤與評估的採納界線

官方 resolved 標籤適用於「任務成功／失敗」，不需為此捏造人工標籤。若要改研究「異常行為／假裝完成」，必須另訂人工判準、盲標及一致性評估；不能把官方 failure 直接當作該新概念的 ground truth。

可補充 Precision@K（K=10、20、50，固定 OOF ranking；同分按 instance ID 排序），同時列失敗 prevalence baseline。因失敗本就占 81%，高 Precision@K 也可能沒有實用收益，混淆矩陣與未警報者風險仍是主輸出。

已核實參考：

- ECOD 原論文：https://arxiv.org/abs/2201.00382
- PyOD ECOD 原始碼（串接 batch 的行為，執行時需再鎖定版本）：https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py
- ADBench，30 方法／57 datasets：https://papers.neurips.cc/paper_files/paper/2022/file/cf93972b116ca5268827d575f2cc226b-Paper-Datasets_and_Benchmarks.pdf
- DRL，ICLR 2025：https://proceedings.iclr.cc/paper_files/paper/2025/hash/3da45dddfc85d9c6e962ce95cff0d127-Abstract-Conference.html
- Causal-aware Anomaly Detection for Tabular Data，ICML 2026：https://proceedings.mlr.press/v306/nguyen26j.html

以上論文確實存在，但不能拿它們的結果替本 repo 的模型排名、資料量門檻或因果能力背書。
