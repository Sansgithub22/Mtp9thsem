from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading model... (first time will download weights)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate(text: str, src_lang: str = "hin_Deva", tgt_lang: str = "bho_Deva") -> str:
    tokenizer.src_lang = src_lang
    enc = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **enc,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=256,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    src = "मैं स्कूल जा रहा हूँ"
    out = translate(src)
    print("Hindi   :", src)
    print("Bhojpuri:", out)
