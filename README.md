# Mtp9thsem

# Model Training & Evaluation
This repository contains a full pipeline for building a **Bhojpuri Universal Dependencies (UD) parser using Trankit**. Hindi UD annotations are projected to Bhojpuri using alignment-based transfer, after which the generated Bhojpuri UD data is used to fine-tune a **Trankit POS + Dependency Parsing model**. The training uses train.conllu and dev.conllu, and model quality is evaluated on test.conllu using standard metrics like UAS and LAS. The workflow outputs Bhojpuri POS tags, lemmas, dependency heads, and relations, enabling a functional Bhojpuri syntactic parser for further NLP research.

# 1. Environment Setup
# Create Virtual Environment
python -m venv mtp_env
mtp_env\Scripts\Activate.ps1

# 2. Install Requirements
pip install -r requirements.txt

If Trankit fails with new PyTorch versions:
pip install "trankit==1.1.0"
pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 3. Setup Offline XLM-R Base
Download xlm-roberta-base and place it here:
Mtp9thsem/xlmr_local/
    ├── config.json
    ├── model.safetensors   (1.1GB)
If Trankit requires pytorch_model.bin instead of safetensors:
Run:
python convert_safetensors_to_bin.py
This creates:
xlmr_local/pytorch_model.bin

# 4. Training the Bhojpuri Trankit Model
python -X utf8 train_trankit_bhoj.py


