# 基於多模態視覺語言模型（VLM）之電商與假客服釣魚防禦系統

Multi-Modal VLM-Powered Defense System Against E-Commerce & Customer-Service Phishing

> NTU 1142《Applying Large Language Models in Cybersecurity Systems》Term Project ｜ 第 62 組
> 鄧芸欣 (P14922003, 臺大) ／ 張緒柏 (D11307003, 臺科大)

傳統釣魚偵測依賴網域黑名單或純文字語意分析，對圖片化訊息、藏惡意 QR Code 的截圖、與
像素級仿冒官方 UI 的假登入頁存在盲區。本專案提出開源、可在地化部署的雙軌制多模態防禦
管道：以 VLM 辨識品牌仿冒與 UI 異常、以 LLM 剖析文字社交工程意圖，再由融合模組產出結構化
威脅報告。

## 架構

```
使用者輸入 URL
      │  fetch_page() 抓取 HTML / 可見文字 / 圖片 / 註冊網域
      ├──> Vision Track (VLM)  ──> vision risk_score
      │     └─ QR 解碼子分析 ──> 圖內 QR 網域 vs 頁面網域不符 = 強訊號
      ├──> Text   Track (LLM)  ──> text   risk_score
      ├──> URL    Track (LLM)  ──> url    risk_score（僅吃 URL 字串）
      ▼
   fuse()  ThreatScore = W_v·vision + W_t·text  (+ 否決規則)
      ▼
   結構化威脅報告（危險等級 / 緩解建議）
```

各分支統一輸出可被融合模組解析的 JSON：`risk_score, confidence, reasons, suspicious_phrases,
detected_language`。`risk_score ≥ 0.7` 判定為釣魚。

## Repo 結構

```
run_report_data.sh               # 一鍵產生填報告所需數據（HF in-process）
requirements.txt                 # Python 3.11 依賴（torch GPU 另從 cu124 index 裝）
datasets/
  build_url_only_dataset.py      # 產生 1:1 URL-only 資料集
  build_fusion_testset.py        # 建 143 頁文字與截圖同頁對齊集（text/url/visual）
  clean_resplit_text.py          # 文字資料去重 / 修錯標 / 網域分組重切（修洩漏）
  prepare_unsloth_studio_parquet.py
  text_train.jsonl / text_valid.jsonl           # 文字分支原始 (158 / 39)
  text_train_clean.jsonl / text_valid_clean.jsonl  # 清洗+重切後 (148 / 42)
  visual_train_studio.parquet / visual_valid_studio.parquet  # 視覺分支 (429 / 106)
  training_data/url_only_1to1/   # URL 分支 (train 540 / eval 60)
  fusion_test/                   # 對齊集（parquet 不進 git，用 builder 重生）
eval/
  run_eval.py / _v1 / _v2 / _v3  # 文字分支（prompt 迭代，Ollama）
  run_eval_url.py                # URL 分支（Ollama）
  run_eval_hf.py                 # 文字/URL 分支（HF in-process，無伺服器；支援 PROMPT_V2）
  run_eval_visual.py             # 視覺分支（OpenAI-compatible vision endpoint）
  run_eval_visual_hf.py          # 視覺分支（HF in-process，Qwen2-VL）
  run_eval_fusion.py             # 融合：2-way(text+visual) + 3-way(url+text+visual)
  RUN_SHEET.md                   # 給 8b/GPU 機器的完整執行清單
  REPORT_DATA_CHECKLIST.md       # 報告缺口 → 腳本 → 輸出檔 對照表
  results/                       # 評估結果 JSON
lora/
  train_lora_unsloth.py          # LoRA / QLoRA 訓練（實際以 Unsloth Studio 進行）
qr/
  qr_decode.py                   # QR 解碼 + 網域不符偵測（補 proposal 案例一盲區）
  testdata/                      # 釣魚 / 合法 QR 測試圖
playbook/
  playbook_generator.py          # 高風險 → Whois/165 檢舉信/使用者宣導 處置 playbook
report/
  final_report.tex               # Week 16 整合報告（ctexart，xelatex）
  Makefile                       # make → final_report.pdf
```

## 環境需求

- Python 3.10+（`requests`, `numpy`, `scikit-learn`, `tqdm`, `pyarrow`, `pillow`；QR 模組需 `opencv-python`，`pyzbar` 可選）
- 推論後端：Ollama（文字/URL 分支）＋ 一個 OpenAI-compatible 視覺端點（視覺分支）
- 報告：XeLaTeX + `ctex`（中文字型）
- **模型**：正式數字以 `llama3.1:8b` 為準（8b/GPU 機器）；`llama3.2:3b` 僅供 Mac smoke test。
  原因見 [eval/RUN_SHEET.md](eval/RUN_SHEET.md)。

## 快速開始

**一鍵（GPU 機器，HF in-process，不需伺服器）**：產生填報告所需的全部數據
```bash
pip install -r requirements.txt   # torch GPU 版另從 cu124 index 裝，見檔內說明
bash run_report_data.sh           # 文字/URL/視覺/融合一次跑完
```

**分步（Ollama 路線）**：模型透過環境變數切換 `EVAL_MODEL`、`OLLAMA_URL`、`VISION_MODEL`、`VISION_BASE_URL`。

```bash
# 1) URL 分支
EVAL_MODEL=llama3.1:8b python eval/run_eval_url.py

# 2) 文字分支（prompt v2 最佳）
EVAL_MODEL=llama3.1:8b python eval/run_eval_v2.py

# 3) 視覺分支（先 dry-run 驗證解碼）
python eval/run_eval_visual.py --dry-run --limit 3
VISION_MODEL=qwen2-vl python eval/run_eval_visual.py

# 4) 融合（143 頁對齊集），詳見 RUN_SHEET ④
python datasets/build_fusion_testset.py
# ... 在對齊集上跑 text + visual，再 run_eval_fusion.py

# 5) 編譯報告
cd report && make
```

完整逐步指令（含 Windows 寫法）見 [eval/RUN_SHEET.md](eval/RUN_SHEET.md)。

## 目前進度

| 分支 / 模組 | 狀態（8b 正式數字）|
|------|------|
| 文字（clean valid, 8b） | ✅ F1 0（標註與任務錯配：訊號不在可見文字）|
| URL（8b） | ✅ F1 0.84（3b 0.32 → 8b 0.84，證實模型容量瓶頸）|
| 視覺（Qwen2-VL 4-bit） | ✅ P 0.93 / R 0.035（受弱標註限制）|
| 融合（143 頁對齊集） | ✅ 2-way 0.275；3-way（URL+文字+視覺）0.692 ≥ 最佳單軌 |
| LoRA | 🟡 文字已試訓（觀察到 dataset shortcut）；URL-LoRA 列未來工作 |
| QR 解碼 | ✅ 完成（OpenCV，網域不符偵測，補案例一）|
| Playbook | ✅ 完成（Whois / 165 檢舉信 / 使用者宣導）|

已知限制（詳見報告 Limitations）：模型容量（3b 太弱）、視覺弱標註、文字資料不平衡、
標註與任務錯配、LoRA dataset shortcut、valid 規模小、推論延遲高。

## 分工

- **鄧芸欣 (P14922003)**：系統架構與評估方法論設計、Prompt 工程、資料品質工程（洩漏修正與 clean 重切）、融合實驗設計（2-way/3-way）、QR 與 Playbook 設計
- **張緒柏 (D11307003)**：核心 pipeline 實作、資料集抓取與前處理、模型評估與 LoRA 微調
