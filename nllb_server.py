# nllb_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading NLLB model (this may take some time the first time)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

app = FastAPI(title="NLLB Hindi→Bhojpuri Server")

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "hin_Deva"
    tgt_lang: str = "bho_Deva"

class BatchRequest(BaseModel):
    texts: list[str]
    src_lang: str = "hin_Deva"
    tgt_lang: str = "bho_Deva"


def nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer.src_lang = src_lang
    enc = tokenizer(text, return_tensors="pt")
    generated = model.generate(
        **enc,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=256,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


@app.post("/translate")
def translate(req: TranslateRequest):
    out = nllb_translate(req.text, req.src_lang, req.tgt_lang)
    return {"translation": out}


@app.post("/translate_batch")
def translate_batch(req: BatchRequest):
    # Simple batch: loop (can be optimized later)
    outputs = []
    for t in req.texts:
        out = nllb_translate(t, req.src_lang, req.tgt_lang)
        outputs.append(out)
    return {"translations": outputs}


@app.get("/")
def root():
    return {"message": "NLLB Hindi→Bhojpuri server is running"}
