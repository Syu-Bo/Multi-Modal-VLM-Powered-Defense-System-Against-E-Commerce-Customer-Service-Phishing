#!/usr/bin/env python3
"""Create Parquet files for Unsloth Studio vision training.

JSONL cannot store PIL Image objects. This script creates Parquet files with:

  image     struct<bytes: binary, path: string>
  messages  JSON string in Unsloth/OpenAI conversation shape
  prompt    plain text prompt
  answer    plain text assistant target

Unsloth Studio/Data Recipes can use the image column as image bytes and the
prompt/answer or messages columns to build the final VLM conversation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_parts(row: dict[str, Any]) -> tuple[str, Path, str]:
    messages = row["messages"]
    user = next(message for message in messages if message["role"] == "user")
    assistant = next(message for message in messages if message["role"] == "assistant")

    prompt = ""
    image_path: Path | None = None
    for item in user["content"]:
        if item.get("type") == "text":
            prompt = item["text"]
        elif item.get("type") == "image":
            image_path = Path(item["image"])
        elif item.get("type") == "image_url":
            raise ValueError("Use path-based visual_train_unsloth_png.jsonl, not base64 JSONL")

    if image_path is None:
        raise ValueError("row has no image item")
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    content = assistant["content"]
    if isinstance(content, list):
        answer = next(item["text"] for item in content if item.get("type") == "text")
    else:
        answer = content
    return prompt, image_path, answer


def messages_without_embedded_image(prompt: str, answer: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def convert(src: Path, dest: Path) -> int:
    images: list[dict[str, Any]] = []
    prompts: list[str] = []
    answers: list[str] = []
    messages: list[str] = []
    source_image_paths: list[str] = []

    for row in read_jsonl(src):
        prompt, image_path, answer = extract_parts(row)
        images.append(
            {
                "bytes": image_path.read_bytes(),
                "path": image_path.name,
            }
        )
        prompts.append(prompt)
        answers.append(answer)
        messages.append(messages_without_embedded_image(prompt, answer))
        source_image_paths.append(str(image_path))

    image_type = pa.struct(
        [
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(images, type=image_type),
            pa.array(prompts, type=pa.string()),
            pa.array(answers, type=pa.string()),
            pa.array(messages, type=pa.string()),
            pa.array(source_image_paths, type=pa.string()),
        ],
        names=["image", "prompt", "answer", "messages", "source_image_path"],
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("training_data/mixed_unsloth_full"))
    parser.add_argument("--train", default="visual_train_unsloth_png.jsonl")
    parser.add_argument("--valid", default="visual_valid_unsloth_png.jsonl")
    parser.add_argument("--out-train", default="visual_train_studio.parquet")
    parser.add_argument("--out-valid", default="visual_valid_studio.parquet")
    args = parser.parse_args()

    for src_name, out_name in [(args.train, args.out_train), (args.valid, args.out_valid)]:
        count = convert(args.dataset_dir / src_name, args.dataset_dir / out_name)
        print(f"wrote {count} rows -> {args.dataset_dir / out_name}")


if __name__ == "__main__":
    main()
