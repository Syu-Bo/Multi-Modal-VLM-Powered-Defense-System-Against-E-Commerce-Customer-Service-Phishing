# 評估執行清單（給 8b / GPU 機器跑）

Mac 端（3b）只做架構/腳本調通，**正式數字請在這台機器用 8b + 視覺模型跑**。
所有腳本走 `PROJECT_ROOT` 相對路徑，不需要改 `C:\project` 硬路徑。

## 為什麼一定要用 `llama3.1:8b`（不能用小模型交數字）

在 Mac 上用 `llama3.2:3b` 跑 URL 分支（60 筆 1:1）的結果：
**Precision 0.75 / Recall 0.20 / F1 0.316**。

看漏判就知道問題在模型容量，不是資料：
- 抓得到的只有尖叫式釣魚：`paypal.com.us.login.cgi-bin...`、`.../inst1.exe`
- 漏掉全是要推理的：`unitystydiying.top`（可疑 TLD+仿冒字）、`bit.ly/...`（短網址）、
  `221.194.44.194:9633`（裸 IP+怪 port）、被入侵正常站+重導參數

3b 讀不出 gold 標註裡編碼的 URL 啟發式特徵（可疑 TLD / 裸 IP / 短網址 / 品牌 token 不符），
只會 pattern match 教科書級樣本 → recall 崩。**8b 才有足夠容量做這類推理**，
跨機器比較與報告數字一律以 8b 為準；3b 僅供 Mac 端 smoke test。

> 延伸：`datasets/training_data/url_only_1to1/url_train.jsonl`（540 筆、1:1）gold 理由
> 正好在教這些 URL 特徵，**適合拿去 LoRA**，預期把 recall 大幅拉升 —— 這就是「為什麼要 fine-tune」的實證。

模型透過環境變數切換（預設值見各腳本）：

| 變數 | 用途 | 建議值 |
|------|------|--------|
| `EVAL_MODEL` | 文字 / URL 分支的 Ollama 模型 | `llama3.1:8b` |
| `OLLAMA_URL` | Ollama generate 端點 | `http://localhost:11434/api/generate` |
| `VISION_MODEL` | 視覺分支模型 | 你服務的 Qwen2-VL / LoRA 名稱 |
| `VISION_BASE_URL` | OpenAI 相容 `/v1` 端點 | `http://localhost:11434/v1` 或你的 vLLM |
| `VISION_API_KEY` | 視覺端點金鑰 | `ollama`（Ollama 隨意填）|

---

## 執行順序

### ① URL 分支（新增，驗證「文字看不到網域」的盲區）
```bat
set EVAL_MODEL=llama3.1:8b
python eval\run_eval_url.py
```
→ `eval/results/url_only_results.json`
資料：`datasets/training_data/url_only_1to1/url_eval.jsonl`（1:1 平衡，60 筆）。

### ② 文字分支（**正式數字請用 clean valid 重跑一次**）
舊 `text_valid.jsonl` 有網域洩漏 / 重複 / 錯標，數字偏高。final 一律以
`text_valid_clean.jsonl`（已修，無洩漏，valid 釣魚 10→18）為準：
```bat
set EVAL_MODEL=llama3.1:8b
set EVAL_DATA=datasets\text_valid_clean.jsonl
set EVAL_OUTPUT=eval\results\prompt_v2_clean_results.json
python eval\run_eval_v2.py
set EVAL_DATA=
set EVAL_OUTPUT=
```
→ `eval/results/prompt_v2_clean_results.json`
（不指定 `EVAL_DATA` 則跑舊的 `text_valid.jsonl`，僅供與洩漏前對照。）

### ③ 視覺分支（新增，需要你服務一個視覺模型）
先確認 `VISION_MODEL` / `VISION_BASE_URL` 指向可用的視覺端點，再跑：
```bat
python eval\run_eval_visual.py --dry-run --limit 3   ::先驗證解碼/payload，不呼叫模型
set VISION_MODEL=<你的 Qwen2-VL 名稱>
python eval\run_eval_visual.py
```
→ `eval/results/visual_results.json`
資料：`datasets/visual_valid_studio.parquet`（106 筆）。

### ④ 融合對照（新增，**不呼叫模型**，純讀 ②③ 結果 join）

**建議用對齊測試集（143 頁，文字＋截圖同頁），結論才穩。**
先重生對齊集（parquet 不進 git，需本機產生）：
```bat
python datasets\build_fusion_testset.py
```
→ `datasets/fusion_test/{text_fusion.jsonl, visual_fusion.parquet, pages.csv}`（143 頁、526 圖）

在這份子集上各跑一次文字與視覺分支（用 EVAL_DATA / EVAL_OUTPUT 指定）：
```bat
set EVAL_MODEL=llama3.1:8b
set EVAL_DATA=datasets\fusion_test\text_fusion.jsonl
set EVAL_OUTPUT=eval\results\text_fusion_results.json
python eval\run_eval_v2.py

set EVAL_DATA=datasets\fusion_test\visual_fusion.parquet
set EVAL_OUTPUT=eval\results\visual_fusion_results.json
python eval\run_eval_visual.py
set EVAL_DATA=
set EVAL_OUTPUT=
```
再融合：
```bat
python eval\run_eval_fusion.py ^
  --text-results eval\results\text_fusion_results.json ^
  --text-data datasets\fusion_test\text_fusion.jsonl ^
  --visual-results eval\results\visual_fusion_results.json
```
→ `eval/results/fusion_results.json` + 終端對照表：`text-only / visual-only / fuse(Wv 多組)`，挑出最佳 Wv。
text/visual 以 **Page URL** 對齊；一頁多張截圖時，頁面視覺分數取**該頁所有圖的 max**。

> 快速版：不想建對齊集，直接 `python eval\run_eval_fusion.py` 會用各自 valid 的交集（僅 ~16 頁），只適合 smoke test。

---

## 重要限制（寫進報告的 Limitations）

1. **視覺標註是弱標註**：圖片只要來自釣魚頁就標 phishing，不看圖片本身是否有視覺釣魚特徵 → 視覺分支指標天生偏低，屬資料限制而非模型失敗。
2. **valid 樣本偏少**：文字 39、視覺 106、融合對齊僅 ~16 → 指標雜訊大，結論需保守。
3. **URL-only gold 為啟發式自動標註**（URL 特徵 + CSV 原始標籤），非人工校驗。
4. **3b vs 8b**：Mac 端 3b 僅供 smoke test，跨機器比較一律以 8b 為準。

## 想擴大 URL 訓練/驗證量
```bat
python datasets\build_url_only_dataset.py --ratio 1.0 --total-rows 2000 --eval-ratio 0.1 --out datasets\training_data\url_only_1to1_big
```
（`--ratio 1.0` = 1:1 平衡；預設 3:1 會壓低 recall，不建議。）
