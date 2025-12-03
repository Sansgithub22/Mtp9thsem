from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"

print("Loading model... please wait ⏳")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

SRC_LANG = "hin_Deva"
TGT_LANG = "bho_Deva"


def translate(text):
    # ✅ CORRECT SAFE FORMAT
    formatted = f"{SRC_LANG} {TGT_LANG} {text}"

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=5
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    src = "मैं स्कूल जा रहा हूँ"
    print("SRC:", src)
    print("OUT:", translate(src))
