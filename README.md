# Mtp9thsem

# Model Training & Evaluation
This repository contains a full pipeline for building a **Bhojpuri Universal Dependencies (UD) parser using Trankit**. Hindi UD annotations are projected to Bhojpuri using alignment-based transfer, after which the generated Bhojpuri UD data is used to fine-tune a **Trankit POS + Dependency Parsing model**. The training uses train.conllu and dev.conllu, and model quality is evaluated on test.conllu using standard metrics like UAS and LAS. The workflow outputs Bhojpuri POS tags, lemmas, dependency heads, and relations, enabling a functional Bhojpuri syntactic parser for further NLP research.

# 1. Environment Setup
