#!/usr/bin/env bash
# 一鍵產生填報告所需的評估數據（3090 / HF in-process 版）。
# 跑完會在 eval/results/ 產生對應 JSON，並印出「結果檔 -> 報告對應表格」。
#
# 用法：
#   conda activate phish
#   bash run_report_data.sh
#
# 可選環境變數：
#   HF_MODEL        文字/URL 模型 (預設 meta-llama/Llama-3.1-8B-Instruct)
#   VISION_MODEL    若設了就會跑視覺分支 + 融合
#   VISION_BASE_URL 視覺 OpenAI 相容端點 (預設 http://localhost:11434/v1)

set -euo pipefail
cd "$(dirname "$0")"

export HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
R=eval/results
mkdir -p "$R"
OUTPUTS=()

echo "================================================================"
echo " 報告數據產生器  |  HF_MODEL=$HF_MODEL"
echo "================================================================"

echo; echo "[1/2] 文字分支（clean valid）..."
EVAL_DATA=datasets/text_valid_clean.jsonl \
  EVAL_OUTPUT="$R/text_hf_clean_results.json" \
  python eval/run_eval_hf.py
OUTPUTS+=("$R/text_hf_clean_results.json    -> 報告 §文字分支表")

echo; echo "[2/2] URL 分支..."
EVAL_DATA=datasets/training_data/url_only_1to1/url_eval.jsonl \
  EVAL_OUTPUT="$R/url_hf_results.json" \
  python eval/run_eval_hf.py
OUTPUTS+=("$R/url_hf_results.json           -> 報告 §URL 分支 (8b 列)")

# 視覺 + 融合：只有設了 VISION_MODEL 才跑
if [[ -n "${VISION_MODEL:-}" ]]; then
  echo; echo "[+] 視覺分支（VISION_MODEL=$VISION_MODEL）..."
  python datasets/build_fusion_testset.py
  EVAL_DATA=datasets/fusion_test/visual_fusion.parquet \
    EVAL_OUTPUT="$R/visual_fusion_results.json" \
    python eval/run_eval_visual.py
  OUTPUTS+=("$R/visual_fusion_results.json   -> 報告 §視覺分支表")

  echo; echo "[+] 文字分支（fusion 對齊子集，供融合 join）..."
  EVAL_DATA=datasets/fusion_test/text_fusion.jsonl \
    EVAL_OUTPUT="$R/text_fusion_results.json" \
    python eval/run_eval_hf.py

  echo; echo "[+] 融合對照..."
  python eval/run_eval_fusion.py \
    --text-results "$R/text_fusion_results.json" \
    --text-data datasets/fusion_test/text_fusion.jsonl \
    --visual-results "$R/visual_fusion_results.json"
  OUTPUTS+=("$R/fusion_results.json          -> 報告 §融合對照表")
else
  echo; echo "（跳過視覺/融合：未設 VISION_MODEL。有視覺模型再設環境變數重跑本腳本。）"
fi

echo
echo "================================================================"
echo " 完成。結果檔 -> 報告對應位置："
echo "================================================================"
for o in "${OUTPUTS[@]}"; do echo "  $o"; done
echo
echo " 尚缺（需 LoRA 訓練 / 視覺模型）："
echo "  eval/results/url_lora_results.json    -> §URL 8b+LoRA 列 / §LoRA 前後"
[[ -z "${VISION_MODEL:-}" ]] && echo "  視覺 + 融合                            -> 設 VISION_MODEL 後重跑"
