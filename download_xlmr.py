from transformers import XLMRobertaModel, XLMRobertaConfig
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "xlmr_local")

os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading xlm-roberta-base from HuggingFace...")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
model.save_pretrained(OUT_DIR)

print("Model saved to:", OUT_DIR)
