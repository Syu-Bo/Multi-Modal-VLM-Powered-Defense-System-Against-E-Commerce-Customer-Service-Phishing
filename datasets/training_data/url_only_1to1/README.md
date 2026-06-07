# URL-only 3:1 Dataset

This dataset trains an LLM to classify phishing vs benign using only the URL string.
No page fetch, page text, image, or network-derived feature is used.

## Files

- `url_train.jsonl`: training split.
- `url_eval.jsonl`: evaluation split.
- `url_train_sources.csv`: URL + label provenance for train.
- `url_eval_sources.csv`: URL + label provenance for eval.
- `summary.json`: counts and split details.

## Labels

- `Label=good` from the CSV is converted to benign.
- `Label=bad` from the CSV is converted to phishing.

The selected data uses an exact 3:1 benign:phishing ratio, bounded by the number of clean benign URLs available.
