#!/usr/bin/env bash
# 一鍵產生填報告所需的評估數據（3090 / HF in-process，不需伺服器）。
# 跑完在 eval/results/ 產生 JSON，並印出「結果檔 -> 報告對應表格」。
#
# 用法：
#   conda activate phish
#   bash run_report_data.sh
#
# 可選環境變數：
#   HF_MODEL         文字/URL 模型 (預設 meta-llama/Llama-3.1-8B-Instruct)
#   VISION_HF_MODEL  視覺模型     (預設 Qwen/Qwen2-VL-7B-Instruct)
#   SKIP_VISUAL=1    跳過視覺(最久的一步)，只做文字/URL 與單軌表

set -euo pipefail
cd "$(dirname "$0")"
export HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
R=eval/results
mkdir -p "$R"
OUTPUTS=()

echo "================================================================"
echo " 報告數據產生器  |  text/url=$HF_MODEL  visual=${VISION_HF_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
echo "================================================================"

# ---- 單軌表（valid 集）----
echo; echo "[1] 文字分支（clean valid）..."
EVAL_DATA=datasets/text_valid_clean.jsonl EVAL_OUTPUT="$R/text_hf_clean_results.json" \
  python eval/run_eval_hf.py
OUTPUTS+=("$R/text_hf_clean_results.json   -> §文字分支表")

echo; echo "[2] URL 分支（url_eval）..."
EVAL_DATA=datasets/training_data/url_only_1to1/url_eval.jsonl EVAL_OUTPUT="$R/url_hf_results.json" \
  python eval/run_eval_hf.py
OUTPUTS+=("$R/url_hf_results.json          -> §URL 分支 (8b 列)")

# ---- 融合對齊集（143 頁）----
echo; echo "[3] 重建 143 頁對齊集..."
python datasets/build_fusion_testset.py

echo; echo "[4] 對齊集-文字..."
EVAL_DATA=datasets/fusion_test/text_fusion.jsonl EVAL_OUTPUT="$R/text_fusion_results.json" \
  python eval/run_eval_hf.py

echo; echo "[5] 對齊集-URL..."
EVAL_DATA=datasets/fusion_test/url_fusion.jsonl EVAL_OUTPUT="$R/url_fusion_results.json" \
  python eval/run_eval_hf.py

if [[ "${SKIP_VISUAL:-}" != "1" ]]; then
  echo; echo "[6] 對齊集-視覺（526 張，最久；4-bit）..."
  LOAD_4BIT=1 EVAL_DATA=datasets/fusion_test/visual_fusion.parquet \
    EVAL_OUTPUT="$R/visual_fusion_results.json" \
    python eval/run_eval_visual_hf.py

  echo; echo "[7] 融合（A: text+visual  /  B: URL+text+visual）..."
  python eval/run_eval_fusion.py \
    --text-results "$R/text_fusion_results.json" \
    --text-data datasets/fusion_test/text_fusion.jsonl \
    --visual-results "$R/visual_fusion_results.json" \
    --url-results "$R/url_fusion_results.json"
  OUTPUTS+=("$R/fusion_results.json         -> §融合對照表 (2-way + 3-way)")
else
  echo; echo "（SKIP_VISUAL=1：跳過視覺與融合）"
fi

echo
echo "================================================================"
echo " 完成。結果檔 -> 報告對應位置："
echo "================================================================"
for o in "${OUTPUTS[@]}"; do echo "  $o"; done
echo
echo " 尚缺（需 LoRA 訓練）："
echo "  eval/results/url_lora_results.json    -> §URL 8b+LoRA 列 / §LoRA 前後"
