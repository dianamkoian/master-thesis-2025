"""
Standalone precompute of multilingual-e5-small embeddings for ozon_train.csv (197 198 rows).
Saves to cdsm/e5_small_embeddings.npy so notebook 08 can skip e5-encoding.

Runs as a fresh python process (no jupyter kernel, no fork issues with PyTorch+MPS).
Expected wall-clock on M4 Pro: ~10-12 min for full encode.
"""
from __future__ import annotations
import os, sys, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

ROOT = Path('/Users/diana/master-thesis-2025')
CSV  = ROOT / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv'
OUT  = ROOT / 'Диана_ВКР_финал' / 'notebooks' / 'cdsm' / 'e5_small_embeddings.npy'
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f'[{time.strftime("%H:%M:%S")}] start; MPS available: {torch.backends.mps.is_available()}')
t0 = time.time()

df = pd.read_csv(CSV, encoding='utf-8')
print(f'[{time.strftime("%H:%M:%S")}] CSV loaded: {df.shape} in {time.time()-t0:.1f}s')

t1 = time.time()
def build_text(row):
    parts = []
    for col in ['brand_name', 'description', 'name_rus']:
        v = row.get(col)
        if pd.notna(v) and str(v).strip() and str(v) != 'nan':
            parts.append(str(v))
    return 'passage: ' + (' '.join(parts) if parts else 'empty')

texts = df.apply(build_text, axis=1).values
print(f'[{time.strftime("%H:%M:%S")}] texts built: {len(texts)} in {time.time()-t1:.1f}s')

t2 = time.time()
device = 'cpu'  # MPS зависает после ~20 batches (memory leak в Apple Metal driver)
print(f'[{time.strftime("%H:%M:%S")}] loading e5 on device={device}')
model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)
print(f'[{time.strftime("%H:%M:%S")}] model loaded in {time.time()-t2:.1f}s')

t3 = time.time()
emb = model.encode(
    texts.tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f'[{time.strftime("%H:%M:%S")}] encoded {emb.shape} in {time.time()-t3:.1f}s')

np.save(OUT, emb)
print(f'[{time.strftime("%H:%M:%S")}] saved {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)')
print(f'[{time.strftime("%H:%M:%S")}] TOTAL: {time.time()-t0:.1f}s')
