# 報告數據缺口對照表（填 final_report.tex 的 [待補]）

跑完依「輸出檔」欄的 JSON 數值，填進 report 對應表格。報告行號對應 `report/final_report.tex`。

| # | 報告位置（行） | 缺的數據 | 跑這支 | 指令 | 輸出檔 |
|---|---|---|---|---|---|
| 1 | L154-163 文字 Prompt v1/v2/v3 | clean valid 上的 v1/v2/v3 對照 | `run_eval_v1/v2/v3.py`（Ollama）或 `run_eval_hf.py`（單版） | ❶ | `eval/results/text_hf_clean_results.json` |
| 2 | L172-186 URL 分支 8b 列 | URL 8b 的 P/R/F1/FPR/P95 | `run_eval_hf.py` | ❷ | `eval/results/url_hf_results.json` |
| 3 | L181 URL 8b+LoRA 列 | LoRA 後 URL 數字 | LoRA 訓練 + 重評 | ❺ | `eval/results/url_lora_results.json` |
| 4 | L189-190 視覺分支 | 視覺 P/R/F1/FPR | `run_eval_visual.py` | ❸ | `eval/results/visual_results.json` |
| 5 | L192-206 融合對照 | text-only / visual-only / fused + Wv | `build_fusion_testset.py` + `run_eval_fusion.py` | ❹ | `eval/results/fusion_results.json` |
| 6 | L208-209 LoRA 前後 | base vs LoRA 之 F1 提升 | LoRA 訓練 + 重評 | ❺ | `eval/results/*_lora_results.json` |

---

## 指令明細

環境變數：`export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`

### ❶ 文字分支（clean valid, 8b）
```bash
EVAL_DATA=datasets/text_valid_clean.jsonl \
  EVAL_OUTPUT=eval/results/text_hf_clean_results.json \
  python eval/run_eval_hf.py
```
> 註：`run_eval_hf.py` 用資料內建的 system prompt（= 單一版本）。若要完整 v1/v2/v3
> 三版對照，改用 Ollama 跑 `run_eval_v1/v2/v3.py` 並加
> `EVAL_DATA=datasets/text_valid_clean.jsonl`。

### ❷ URL 分支（8b）
```bash
EVAL_DATA=datasets/training_data/url_only_1to1/url_eval.jsonl \
  EVAL_OUTPUT=eval/results/url_hf_results.json \
  python eval/run_eval_hf.py
```

### ❸ 視覺分支（需要一個視覺模型端點）
```bash
python eval/run_eval_visual.py --dry-run --limit 3        # 先驗證解碼
VISION_MODEL=<Qwen2-VL 名稱> VISION_BASE_URL=<端點> \
  python eval/run_eval_visual.py
```

### ❹ 融合（需先有 ❶ 文字 + ❸ 視覺結果）
```bash
python datasets/build_fusion_testset.py
# 在 143 頁對齊集上各跑一次文字與視覺，再融合（詳見 RUN_SHEET ④）
python eval/run_eval_fusion.py \
  --text-results eval/results/text_fusion_results.json \
  --text-data datasets/fusion_test/text_fusion.jsonl \
  --visual-results eval/results/visual_fusion_results.json
```

### ❺ LoRA（訓練後重評）
```bash
# 訓練見 lora/；產出 adapter 後，用同一支 eval 指向 LoRA 模型重跑：
HF_MODEL=<base+adapter 或 merged 模型路徑> \
  EVAL_DATA=datasets/training_data/url_only_1to1/url_eval.jsonl \
  EVAL_OUTPUT=eval/results/url_lora_results.json \
  python eval/run_eval_hf.py
```

---

## 最省力路線（3090 一鍵）
```bash
bash run_report_data.sh
```
自動跑 ❶❷，若設了 `VISION_MODEL` 再跑 ❸❹。結尾印出「結果檔 -> 報告對應位置」。
