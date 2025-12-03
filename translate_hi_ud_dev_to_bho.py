from pathlib import Path
from conllu import parse_incr
from google import genai
import math
import time

# ============ CONFIG ============

# 👉 Yaha apna Gemini API key daalo:
API_KEY = "AIzaSyAsEJuorCOriPHK7pWXHc6QKgPQivEzuL0"

# Input: Hindi UD train file (same folder me rakho ya path change karo)
INPUT_FILE = Path("data/hindi_ud/hi_hdtb-ud-train.conllu")

# Output: sirf Bhojpuri sentences line-by-line
OUTPUT_FILE = Path("data/new/dev_bhojpuri_100_requests.txt")

# Hum chahte hain approx itne requests:
TARGET_REQUESTS = 100

# Rate limit handle karne ke liye:
SLEEP_BETWEEN_REQUESTS = 3     # har request ke baad kitna wait
RETRY_COOLDOWN_SECONDS = 20    # 429 aane pe kitna rukna
MAX_RETRIES = 5                # ek batch ke liye max retry

# ================================

client = genai.Client(api_key=API_KEY)


def extract_sentences_from_conllu(path: Path):
    """
    UD .conllu file se plain Hindi sentences ki list banata hai.
    """
    sentences = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            words = [tok["form"] for tok in sent if isinstance(tok["id"], int)]
            text = " ".join(words).strip()
            if text:
                sentences.append(text)
    return sentences


def translate_batch(sentences):
    """
    Ek batch (list of Hindi sentences) ko Gemini se Bhojpuri me translate karta hai.
    Quota error pe retry + cooldown bhi handle karta hai.
    """
    if not sentences:
        return []

    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    prompt = f"""
You are an expert translator for Indian languages.

Translate the following Hindi sentences into Bhojpuri.
Return ONLY the Bhojpuri translations, one per line, in the SAME order.
Do NOT return sentence numbers, Hindi text, or any explanation.

Hindi sentences:
{numbered}

Bhojpuri translations (one per line, same order):
"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            text = response.text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if len(lines) != len(sentences):
                print(
                    f"[WARN] Expected {len(sentences)} lines, got {len(lines)}. "
                    "Still writing whatever we got."
                )
            return lines

        except Exception as e:
            msg = str(e)
            print(f"[ERROR] Batch failed (attempt {attempt}/{MAX_RETRIES}): {msg}")

            # Agar quota / RESOURCE_EXHAUSTED hai to wait + retry
            if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                if attempt < MAX_RETRIES:
                    print(f"[INFO] Quota hit. Cooling down for {RETRY_COOLDOWN_SECONDS}s...")
                    time.sleep(RETRY_COOLDOWN_SECONDS)
                    continue
            # Agar koi aur error ya max retry cross, to empty outputs
            print("[ERROR] Giving up on this batch, filling empty outputs.")
            return [""] * len(sentences)


def main():
    if not INPUT_FILE.exists():
        print("❌ Input file not found:", INPUT_FILE)
        return

    print("📂 Reading Hindi UD train file:", INPUT_FILE)
    hindi_sents = extract_sentences_from_conllu(INPUT_FILE)
    total = len(hindi_sents)
    print(f"✅ Total sentences found: {total}")

    # Batch size aisa choose karo ki approx TARGET_REQUESTS ho
    batch_size = max(1, math.ceil(total / TARGET_REQUESTS))
    approx_requests = math.ceil(total / batch_size)
    print(f"🎯 Target requests: {TARGET_REQUESTS}")
    print(f"📏 Computed batch size: {batch_size} sentences/request")
    print(f"📡 Approx actual requests: {approx_requests}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = hindi_sents[start:end]
            print(f"\n➡ Translating sentences {start+1}–{end} (batch size={len(batch)})")

            bho_batch = translate_batch(batch)

            for bho in bho_batch:
                fout.write(bho + "\n")
                written += 1

            print(f"   ✅ Total sentences written so far: {written}")
            print(f"   ⏳ Sleeping {SLEEP_BETWEEN_REQUESTS}s before next request...")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("\n🎉 DONE!")
    print(f"Total sentences written: {written}")
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
