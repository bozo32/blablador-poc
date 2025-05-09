# smoke.py
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("faiss version:", faiss.__version__)
print("torch version:", torch.__version__)
tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base", use_fast=False)
print("tokenizer loaded")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base")
print("model loaded")