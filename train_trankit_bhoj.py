#!/usr/bin/env python3

"""
Train a Trankit POS+DEP model on Bhojpuri UD data and evaluate on test set.

Requirements (in env 'trankit310'):
- torch
- transformers==4.36.2
- trankit==1.1.0

Files expected:
- split/train.conllu
- split/dev.conllu
- splittest.conllu  (test)
"""

import os
from typing import List, Tuple

import trankit
from trankit import TPipeline, Pipeline

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# local XLM-R directory (already downloaded)
XLMR_DIR = r"C:\Users\HP\Downloads\Hey_mtp\xlmr_local"

TRAIN_CONLLU = os.path.join(BASE_DIR, "split", "train.conllu")
DEV_CONLLU   = os.path.join(BASE_DIR, "split", "dev.conllu")
TEST_CONLLU  = os.path.join(BASE_DIR, "splittest.conllu")

MODEL_DIR    = os.path.join(BASE_DIR, "bhoj_trankit_model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- Small CoNLL-U helpers ----------

def read_conllu_sentences(path: str):
    sentences = []
    cur = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sentences.append(cur)
                    cur = []
            else:
                cur.append(line)
        if cur:
            sentences.append(cur)
    return sentences


def extract_token_rows(lines):
    tokens = []
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split("\t")
        # skip multiword or empty nodes
        if "-" in parts[0] or "." in parts[0]:
            continue
        tokens.append(parts)
    return tokens


def build_raw_from_tokens(tokens):
    return " ".join(t[1] for t in tokens)


def write_conllu_sentences(sent_blocks, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sent_blocks) + "\n")


def evaluate_uas_las(gold_path: str, sys_path: str) -> Tuple[float, float]:
    gold_sents = read_conllu_sentences(gold_path)
    sys_sents  = read_conllu_sentences(sys_path)

    assert len(gold_sents) == len(sys_sents), "Mismatch in number of sentences"

    total_tokens = 0
    correct_heads = 0
    correct_head_and_label = 0

    for g_lines, s_lines in zip(gold_sents, sys_sents):
        g_tokens = extract_token_rows(g_lines)
        s_tokens = extract_token_rows(s_lines)

        n = min(len(g_tokens), len(s_tokens))
        for i in range(n):
            g = g_tokens[i]
            s = s_tokens[i]

            g_head = g[6]
            g_rel  = g[7]
            s_head = s[6]
            s_rel  = s[7]

            total_tokens += 1
            if g_head == s_head:
                correct_heads += 1
                if g_rel == s_rel:
                    correct_head_and_label += 1

    uas = 100.0 * correct_heads / total_tokens if total_tokens else 0.0
    las = 100.0 * correct_head_and_label / total_tokens if total_tokens else 0.0
    return uas, las


# ---------- Step 1: Train POS+DEP model ----------

def train_posdep():
    print("Training POS+DEP model with Trankit...")
    print("XLMR_DIR =", XLMR_DIR)

    training_config = {
        "category": "customized-mwt",
        "task": "posdep",
        "save_dir": MODEL_DIR,
        "train_conllu_fpath": TRAIN_CONLLU,
        "dev_conllu_fpath": DEV_CONLLU,
        "embedding_name": XLMR_DIR,   # USE LOCAL XLM-R
        "max_epochs": 5,
    }

    print("Training config =", training_config)

    trainer = TPipeline(training_config=training_config)
    trainer.train()
    print("Training complete. Model saved to:", MODEL_DIR)


# ---------- Step 2: Load model & predict on test ----------

def load_pipeline() -> Pipeline:
    print("Verifying and loading custom pipeline...")
    trankit.verify_customized_pipeline(
        category="customized-mwt",
        save_dir=MODEL_DIR,
        embedding_name=XLMR_DIR,
    )

    pipe = Pipeline(
        lang="customized-mwt",
        cache_dir=MODEL_DIR,
        gpu=False,
    )
    return pipe


def predict_on_test(pipe: Pipeline, out_path: str) -> None:
    print("Generating predictions on test set...")
    test_sents = read_conllu_sentences(TEST_CONLLU)

    out_blocks = []

    for sent_lines in test_sents:
        comments = [ln for ln in sent_lines if ln.startswith("#")]
        gold_tokens = extract_token_rows(sent_lines)
        if not gold_tokens:
            continue

        raw_text = build_raw_from_tokens(gold_tokens)
        doc = pipe(raw_text)
        pred_tokens = doc["sentences"][0]["tokens"]

        new_rows = []
        for gold_tok, pred_tok in zip(gold_tokens, pred_tokens):
            tid = gold_tok[0]
            form = gold_tok[1]
            lemma = pred_tok.get("lemma", "_")
            upos = pred_tok.get("upos", "_")
            xpos = pred_tok.get("xpos", "_")
            feats = "_"
            head = str(pred_tok.get("head", 0))
            deprel = pred_tok.get("deprel", "_")
            deps = "_"
            misc = "_"
            new_row = "\t".join(
                [tid, form, lemma, upos, xpos, feats, head, deprel, deps, misc]
            )
            new_rows.append(new_row)

        block = "\n".join(comments + new_rows)
        out_blocks.append(block)

    write_conllu_sentences(out_blocks, out_path)
    print(f"System predictions written to: {out_path}")


def main():
    train_posdep()
    pipe = load_pipeline()

    system_path = os.path.join(BASE_DIR, "system_pred.conllu")
    predict_on_test(pipe, system_path)

    uas, las = evaluate_uas_las(TEST_CONLLU, system_path)
    print("\n=== Evaluation on TEST set ===")
    print(f"UAS: {uas:.2f}%")
    print(f"LAS: {las:.2f}%")


if __name__ == "__main__":
    main()
