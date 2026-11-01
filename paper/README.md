# HW2 論文（ACL 格式）

ACL 格式的 LaTeX 原始檔。PDF 不放進 git（見 `.gitignore`），每個人在自己電腦上編譯。

## 檔案

| 檔案 | 用途 |
|---|---|
| `main.tex` | 具名版（`[preprint]`）：標題、成員、GitHub 連結，依序 `\input` 各章節 |
| `main_review.tex` | 匿名送審版（`[review]`，顯示 Team ID）；只設定 `\anonymous`，內容同 `main.tex`，不需另外維護 |
| `sections/*.tex` | 各章節，一章一檔 |
| `references.bib` | 參考文獻 |
| `acl.sty`, `acl_natbib.bst` | 官方 ACL 樣式，請勿修改 |

圖片從 `../results/` 讀取（`\graphicspath`），已在 repo 中：
`stage4_eda/fig_samples.png`、`stage4_eda/fig_eda_paper.png`、`stage8_analysis/fig_heatmaps.png`。

## 環境設定

1. 安裝 LaTeX：MiKTeX 或 TeX Live 皆可，確認 `pdflatex` 與 `bibtex` 在 PATH 上。
   MiKTeX 建議開啟「自動安裝缺少的套件」（Always install missing packages on-the-fly）。
2. 在 VS Code 安裝 **LaTeX Workshop** 擴充套件。
3. 用 VS Code 開啟**整個 repo 資料夾**（不是只開 `paper/`），才會讀到 `.vscode/settings.json` 的編譯設定
   （pdflatex → bibtex → pdflatex ×2，不需要 Perl / latexmk）。
4. 打開 `paper/main.tex` 存檔即自動編譯，PDF 在 VS Code 分頁中開啟。修改 `sections/` 中任一章節後存檔也會重新編譯。

全新 clone 第一次編譯時，LaTeX 可能提示 `Label(s) may have changed. Rerun`，再存檔（再編一次）即可。

不使用 VS Code 時，在 `paper/` 資料夾手動編譯：

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

匿名版把上面的 `main` 換成 `main_review`。

## 多人協作注意事項

- **一章一檔**：各自負責不同的 `sections/*.tex`，避免同時修改同一檔案，以減少 merge 衝突。
- **頁數限制**：正文（含 Contributions）最多 6 頁，References 不計。每次修改後確認正文沒有超過第 6 頁。
- **數字來源**：每個章節開頭的註解寫明數字出自 `results/` 的哪個 CSV；修改數字時請對照來源。
- **提交前**：確認編譯沒有錯誤或 Overfull 警告，再 commit。

## 繳交前待辦

- `main.tex`：標題、`\teamid`、成員姓名與學號。
- `sections/contributions.tex`：成員分工與 AI 使用聲明（約 5–8 行）。
- 繳交檔名：`HW2_<TeamID>_report.pdf`；GitHub 連結需在標題下方（`\repourl`）。
