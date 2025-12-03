import requests
from conllu import parse_incr
from pathlib import Path

# ✅ Make sure this path is correct and file actually exists
INPUT_FILE = Path("Data/hindi_ud/hi_hdtb-ud-train.conllu")  # change if dev file
OUTPUT_FILE = Path("Data/server/bhojpuri_train_nllb.txt")
API_URL = "http://127.0.0.1:8000/translate"


def extract_sentences(path: Path):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            words = [tok["form"] for tok in sent if isinstance(tok["id"], int)]
            text = " ".join(words).strip()
            if text:
                sentences.append(text)
    return sentences


def translate_sentence(sentence: str) -> str:
    payload = {
        "text": sentence,
        "src_lang": "hin_Deva",
        "tgt_lang": "bho_Deva",
    }
    response = requests.post(API_URL, json=payload)

    # ✅ Safety: agar server error de, toh clean message mile
    if response.status_code != 200:
        print("HTTP error:", response.status_code, response.text)
        return ""

    data = response.json()
    return data.get("translation", "")


def main():
    if not INPUT_FILE.exists():
        print("❌ INPUT_FILE not found:", INPUT_FILE)
        return

    # ✅ Ensure output folder exists (important!)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    sentences = extract_sentences(INPUT_FILE)
    total = len(sentences)
    print(f"Total sentences: {total}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i, sent in enumerate(sentences, start=1):
            bho = translate_sentence(sent)
            fout.write(bho + "\n")

            if i % 10 == 0:
                print(f"Translated {i}/{total}")

    print("✅ Done! Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
