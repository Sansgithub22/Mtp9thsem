# convert_safetensors_to_bin.py
import os
import torch
from safetensors.torch import load_file

SRC = r"C:\Users\HP\.cache\huggingface\hub\models--xlm-roberta-base"
# if you copied to a different location use that path above.
safetensors_path = os.path.join(SRC, "model.safetensors")
out_path = os.path.join(SRC, "pytorch_model.bin")

print("SRC:", safetensors_path)
print("OUT:", out_path)

if not os.path.exists(safetensors_path):
    raise SystemExit("safetensors file not found: " + safetensors_path)

# load safetensors into a dict of tensors
sd = load_file(safetensors_path)

# convert keys to tensors (they already are) and save as pytorch_state_dict
# torch.save expects tensors to be CPU tensors (they likely already are)
for k, v in sd.items():
    if isinstance(v, torch.Tensor):
        sd[k] = v.cpu()
    else:
        sd[k] = torch.tensor(v).cpu()

torch.save(sd, out_path)
print("Saved pytorch_model.bin ->", out_path)
