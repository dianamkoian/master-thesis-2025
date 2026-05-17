"""
M2 e5-variant: вариация мультимодальной модели Дианы с заменой TF-IDF SVD
на multilingual-e5-base (768-dim). Обучается на едином командном split
(team_train_idx + team_test_idx от С. Красовской).

Выход: test_proba_diana_m2_e5.npy (58410, float32) — вероятности на team test.
Цель: новая строка в Таблице 5.1 как "real_estate M2-e5 (multilingual-e5-base)".
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
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm2_e5_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2 e5-variant: старт')

# 1. Данные + индексы Сони
log('Загрузка train.csv...')
df = pd.read_csv(DATA_CSV, encoding='utf-8')
log(f'  {df.shape}, positive {df["resolution"].mean():.4f}')
train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')
log(f'  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

# 2. Подготовка текстов и таблички
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
df['text'] = df['text'].str.slice(0, 512)

# 3. CLIP embeddings (по ItemID): развернуть list-эмбеддинги в столбцы img_0..img_511
log('Загрузка clip_embeddings.parquet...')
clip_df = pd.read_parquet(CLIP_PARQUET)
log(f'  CLIP parquet: {clip_df.shape}, columns={clip_df.columns.tolist()}')
# embedding — это list/np.array длины 512 в каждой ячейке
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
log(f'  CLIP matrix shape: {clip_matrix.shape}')
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
log(f'  CLIP столбцов после expand: {len(clip_cols)}')
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()

# 4. e5-base embeddings — на ВСЕ тексты сразу, чтобы не дублировать
log('Загрузка multilingual-e5-base...')
from sentence_transformers import SentenceTransformer
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
log(f'  device: {DEVICE}')
model = SentenceTransformer('intfloat/multilingual-e5-base', device=DEVICE)
log(f'  модель загружена; max_seq_length={model.max_seq_length}')

# Подготовить тексты с префиксом query как требует e5
texts = ('query: ' + df['text']).tolist()
log(f'  всего текстов: {len(texts)}')

# Inference батчами
batch_size = 32
n = len(texts)
emb_dim = 768
e5 = np.zeros((n, emb_dim), dtype=np.float32)
t_emb = time.time()
for i in range(0, n, batch_size):
    chunk = texts[i:i + batch_size]
    e5[i:i + batch_size] = model.encode(chunk, batch_size=batch_size,
                                         show_progress_bar=False,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)
    if i % (batch_size * 200) == 0 and i > 0:
        elapsed = time.time() - t_emb
        rate = i / elapsed
        eta = (n - i) / rate
        log(f'  e5 progress: {i}/{n} ({100*i/n:.1f}%), {rate:.1f} txt/s, ETA {eta/60:.1f} min')
log(f'  e5 готово за {(time.time()-t_emb)/60:.1f} мин')
del model; gc.collect()
import torch
if DEVICE == 'mps':
    torch.mps.empty_cache()

# 5. PCA-25 на e5 для совместимости с другими табличными признаками
log('PCA-25 на e5 embeddings...')
from sklearn.decomposition import PCA
pca = PCA(n_components=25, random_state=42)
e5_pca = pca.fit_transform(e5)
log(f'  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}')
for i in range(25):
    df[f'e5_{i}'] = e5_pca[:, i]
del e5, e5_pca; gc.collect()

# 6. Простой набор tab признаков из meta (как Соня описывает)
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
log(f'  tab колонок: {len(tab_cols)}')

# Импутация: median для числовых
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')

# Категориальные (CommercialTypeName4)
cat_cols = ['CommercialTypeName4']
for c in cat_cols:
    df[c] = df[c].astype(str)

# Соберём финальный X
e5_cols = [f'e5_{i}' for i in range(25)]
feature_cols = tab_cols + clip_cols + e5_cols
log(f'  итого признаков: {len(feature_cols)} (tab={len(tab_cols)} + clip={len(clip_cols)} + e5={len(e5_cols)})')

X = df[feature_cols].copy()
y = df['resolution'].values

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y[train_idx]
X_val = X.iloc[val_idx].reset_index(drop=True)
y_val = y[val_idx]
X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y[test_idx]
log(f'  X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}')

# 7. CatBoost
log('Обучение CatBoost...')
from catboost import CatBoostClassifier
cb = CatBoostClassifier(
    iterations=2500,
    depth=8,
    learning_rate=0.05,
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=100,
    cat_features=cat_cols,
    thread_count=4,
    verbose=200,
    auto_class_weights='Balanced',
)
t_cb = time.time()
cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
log(f'  CatBoost обучен за {(time.time()-t_cb)/60:.1f} мин, best iter {cb.tree_count_}')

# 8. Предсказание на val для isotonic + test
from sklearn.isotonic import IsotonicRegression
p_val_raw = cb.predict_proba(X_val)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val_raw, y_val)

p_test_raw = cb.predict_proba(X_test)[:, 1].astype(np.float32)
p_test_cal = iso.predict(p_test_raw).astype(np.float32)

# 9. Метрики
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

log('=' * 50)
log('Метрики на team test (истинные метки):')
roc = roc_auc_score(y_test_true, p_test_raw)
pr  = average_precision_score(y_test_true, p_test_raw)
r09 = rap(y_test_true, p_test_raw)
log(f'  RAW:        ROC={roc:.4f}  PR-AUC={pr:.4f}  R@P>=0.9={r09:.4f}')
roc_c = roc_auc_score(y_test_true, p_test_cal)
pr_c  = average_precision_score(y_test_true, p_test_cal)
r09_c = rap(y_test_true, p_test_cal)
log(f'  Calibrated: ROC={roc_c:.4f}  PR-AUC={pr_c:.4f}  R@P>=0.9={r09_c:.4f}')

# Сравнение с M2 (diana_team)
p_m2 = np.load(OUT_DIR / 'test_proba_diana_team.npy')
m2_roc = roc_auc_score(y_test_true, p_m2)
m2_pr  = average_precision_score(y_test_true, p_m2)
log(f'  M2 (current headline) для справки: ROC={m2_roc:.4f}  PR-AUC={m2_pr:.4f}')
log(f'  Прирост от e5: ΔPR-AUC = {pr - m2_pr:+.4f}, ΔROC = {roc - m2_roc:+.4f}')

# Ансамбль M2 + M2-e5
mix = 0.5 * p_m2 + 0.5 * p_test_raw
mix_pr = average_precision_score(y_test_true, mix)
mix_roc = roc_auc_score(y_test_true, mix)
mix_r09 = rap(y_test_true, mix)
log(f'  Ансамбль 0.5×M2 + 0.5×M2-e5: ROC={mix_roc:.4f}  PR={mix_pr:.4f}  R@P>=0.9={mix_r09:.4f}')

# 10. Сохранить
np.save(OUT_DIR / 'test_proba_diana_m2_e5.npy', p_test_raw)
np.save(OUT_DIR / 'test_proba_diana_m2_e5_calibrated.npy', p_test_cal)
log(f'Saved → {OUT_DIR}/test_proba_diana_m2_e5.npy')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
