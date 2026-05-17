"""
M2 RuBERT-variant: вариация мультимодальной модели Дианы с заменой TF-IDF SVD
на cointegrated/rubert-tiny2 (312-dim, специализированный русскоязычный BERT-tiny).
Обучается на едином командном split.

Выход: test_proba_diana_m2_rubert.npy (58410, float32).
Цель: новая строка в Таблице 5.1 как "real_estate M2-rubert (cointegrated/rubert-tiny2)".
"""
import gc, time, os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
SPLITS = ROOT / 'real_estate_approaches' / 'notebooks' / 'team_splits'
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm2_rubert_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2 rubert-variant (cointegrated/rubert-tiny2): старт')

# 1. Данные + индексы
log('Загрузка train.csv...')
df = pd.read_csv(DATA_CSV, encoding='utf-8')
train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')
log(f'  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

# 2. Тексты + CLIP
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
df['text'] = df['text'].str.slice(0, 512)

log('Загрузка CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
log(f'  CLIP merged: {len(clip_cols)} cols')
del clip_df, clip_matrix, clip_expanded; gc.collect()

# 3. RuBERT embeddings через transformers
log('Загрузка cointegrated/rubert-tiny2...')
from transformers import AutoTokenizer, AutoModel
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
log(f'  device: {DEVICE}')
tok = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
mdl = AutoModel.from_pretrained('cointegrated/rubert-tiny2').to(DEVICE)
mdl.eval()
log(f'  модель загружена, embedding dim = {mdl.config.hidden_size}')

texts = df['text'].tolist()
log(f'  всего текстов: {len(texts)}')

batch_size = 64
n = len(texts)
emb_dim = mdl.config.hidden_size
rubert = np.zeros((n, emb_dim), dtype=np.float32)
t_emb = time.time()

@torch.no_grad()
def encode_batch(batch_texts):
    enc = tok(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    out = mdl(**enc)
    # mean pooling over non-padded tokens
    mask = enc['attention_mask'].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    emb = (summed / counts).cpu().numpy()
    return emb

for i in range(0, n, batch_size):
    chunk = texts[i:i + batch_size]
    rubert[i:i + len(chunk)] = encode_batch(chunk)
    if i % (batch_size * 200) == 0 and i > 0:
        elapsed = time.time() - t_emb
        rate = i / elapsed
        eta = (n - i) / rate
        log(f'  rubert progress: {i}/{n} ({100*i/n:.1f}%), {rate:.1f} txt/s, ETA {eta/60:.1f} min')
log(f'  rubert готово за {(time.time()-t_emb)/60:.1f} мин')
del mdl, tok; gc.collect()
if DEVICE == 'mps':
    torch.mps.empty_cache()

# 4. PCA-25 на rubert
log('PCA-25 на rubert embeddings...')
from sklearn.decomposition import PCA
pca = PCA(n_components=25, random_state=42)
rubert_pca = pca.fit_transform(rubert)
log(f'  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}')
for i in range(25):
    df[f'rubert_{i}'] = rubert_pca[:, i]
del rubert, rubert_pca; gc.collect()

# 5. Tab признаки
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
cat_cols = ['CommercialTypeName4']
for c in cat_cols:
    df[c] = df[c].astype(str)

rubert_cols = [f'rubert_{i}' for i in range(25)]
feature_cols = tab_cols + clip_cols + rubert_cols
log(f'  итого признаков: {len(feature_cols)} (tab={len(tab_cols)} + clip={len(clip_cols)} + rubert={len(rubert_cols)})')

X = df[feature_cols].copy()
y = df['resolution'].values
X_train = X.iloc[train_idx].reset_index(drop=True); y_train = y[train_idx]
X_val = X.iloc[val_idx].reset_index(drop=True); y_val = y[val_idx]
X_test = X.iloc[test_idx].reset_index(drop=True); y_test = y[test_idx]
log(f'  X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}')

# 6. CatBoost
log('Обучение CatBoost...')
from catboost import CatBoostClassifier
cb = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05,
    eval_metric='AUC', random_seed=42, early_stopping_rounds=100,
    cat_features=cat_cols, thread_count=4, verbose=200,
    auto_class_weights='Balanced',
)
t_cb = time.time()
cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
log(f'  CatBoost обучен за {(time.time()-t_cb)/60:.1f} мин, best iter {cb.tree_count_}')

# 7. Predict + isotonic
from sklearn.isotonic import IsotonicRegression
p_val_raw = cb.predict_proba(X_val)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val_raw, y_val)
p_test_raw = cb.predict_proba(X_test)[:, 1].astype(np.float32)
p_test_cal = iso.predict(p_test_raw).astype(np.float32)

# 8. Метрики
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

log('=' * 50)
roc = roc_auc_score(y_test_true, p_test_raw)
pr = average_precision_score(y_test_true, p_test_raw)
r09 = rap(y_test_true, p_test_raw)
log(f'  M2-rubert RAW: ROC={roc:.4f}  PR={pr:.4f}  R@P>=0.9={r09:.4f}')
roc_c = roc_auc_score(y_test_true, p_test_cal)
pr_c = average_precision_score(y_test_true, p_test_cal)
log(f'  M2-rubert CAL: ROC={roc_c:.4f}  PR={pr_c:.4f}  R@P>=0.9={rap(y_test_true, p_test_cal):.4f}')

# Сравнения с другими моделями
p_m2 = np.load(OUT_DIR / 'test_proba_diana_team.npy')
p_e5 = np.load(OUT_DIR / 'test_proba_diana_m2_e5.npy')
p_no_te = np.load(OUT_DIR / 'test_proba_no_te.npy')
log(f'  Контекст: M2={average_precision_score(y_test_true, p_m2):.4f}  M2_e5={average_precision_score(y_test_true, p_e5):.4f}  M2_no_te={average_precision_score(y_test_true, p_no_te):.4f}')

# Простые ансамбли с rubert
log('  Ансамбли:')
for name, partner in [('M2', p_m2), ('M2_e5', p_e5), ('M2_no_te', p_no_te)]:
    mix = 0.5 * p_test_raw + 0.5 * partner
    log(f'    M2-rubert + {name}: PR={average_precision_score(y_test_true, mix):.4f}  R@P09={rap(y_test_true, mix):.4f}')

# Тройной
mix3 = (p_test_raw + p_m2 + p_e5) / 3
log(f'    M2-rubert + M2 + M2_e5: PR={average_precision_score(y_test_true, mix3):.4f}  R@P09={rap(y_test_true, mix3):.4f}')
mix3b = (p_test_raw + p_m2 + p_no_te) / 3
log(f'    M2-rubert + M2 + M2_no_te: PR={average_precision_score(y_test_true, mix3b):.4f}  R@P09={rap(y_test_true, mix3b):.4f}')

# 4-way
mix4 = (p_test_raw + p_m2 + p_e5 + p_no_te) / 4
log(f'    M2-rubert + M2 + M2_e5 + M2_no_te (4-way): PR={average_precision_score(y_test_true, mix4):.4f}  R@P09={rap(y_test_true, mix4):.4f}')

# 9. Save
np.save(OUT_DIR / 'test_proba_diana_m2_rubert.npy', p_test_raw)
np.save(OUT_DIR / 'test_proba_diana_m2_rubert_calibrated.npy', p_test_cal)
log(f'Saved → test_proba_diana_m2_rubert.npy')
log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
