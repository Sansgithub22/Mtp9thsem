#!/usr/bin/env python3

"""
Hindi -> Bhojpuri file translation using Meta's NLLB-200 model (offline).

- No Google Cloud, no external API.
- Model: facebook/nllb-200-distilled-600M
- Input : hindi-train.txt  (UTF-8)
- Output: hindi-train.bho.txt (UTF-8)
"""

from pathlib import Path
from typing import List

import torch
from transformers import pipeline

# ---------- CONFIG ----------

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# FLORES-200 language codes
SRC_LANG = "hin_Deva"   # Hindi (Devanagari)
TGT_LANG = "bho_Deva"   # Bhojpuri (Devanagari)

INPUT_FILE = Path("final/hindi-train.txt")
OUTPUT_FILE = Path("final/train_output.txt")

BATCH_SIZE = 16      # kitni lines ek saath translate karni
MAX_LENGTH = 256     # ek sentence ki max length (tokens)


# ---------- HELPERS ----------

def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def chunk_list(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------- MAIN LOGIC ----------

def load_translator():
    print(f"🔁 Loading model: {MODEL_NAME} (first time will download, thoda time lagega)...")
    device = 0 if torch.cuda.is_available() else -1

    translator = pipeline(
        task="translation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        max_length=MAX_LENGTH,
        device=device,
    )

    print("✅ Model loaded on", "GPU" if device == 0 else "CPU")
    return translator


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    print(f"📖 Reading Hindi file: {INPUT_FILE}")
    lines = read_lines(INPUT_FILE)
    total = len(lines)
    print(f"Total lines: {total}")

    translator = load_translator()

    output_lines: List[str] = []
    processed = 0

    for batch_idx, batch in enumerate(chunk_list(lines, BATCH_SIZE), start=1):
        # sab line blank ho to skip translation
        if all(not line.strip() for line in batch):
            output_lines.extend(["" for _ in batch])
            processed += len(batch)
            print(f"[Batch {batch_idx}] Empty lines, skipping ({processed}/{total})")
            continue

        print(f"[Batch {batch_idx}] Translating {len(batch)} lines... ({processed}/{total})")

        # HF pipeline directly list of strings leta hai
        results = translator(batch)

        # results: list of dicts with "translation_text"
        translated_batch = []
        for original, res in zip(batch, results):
            if not original.strip():
                translated_batch.append("")
            else:
                translated_batch.append(res["translation_text"])

        output_lines.extend(translated_batch)
        processed += len(batch)

    print(f"💾 Writing Bhojpuri file: {OUTPUT_FILE}")
    write_lines(OUTPUT_FILE, output_lines)
    print("🎉 DONE: hindi-train.txt → hindi-train.bho.txt")


if __name__ == "__main__":
    main()
