from indictrans2.inference.engine import ModelEngine

model = ModelEngine(
    model_name="ai4bharat/indictrans2-indic-indic-1B",
    device="cpu"  # change to cuda if GPU
)

text = "मैं स्कूल जा रहा हूँ"

result = model.translate(
    src=text,
    src_lang="hin_Deva",
    tgt_lang="bho_Deva"
)

print("SRC:", text)
print("OUT:", result)
