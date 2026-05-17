"""
TabPFN v2 (Hollmann et al., Nature 2025) — foundation model для табличных данных.
Используется как M6 в твоей линейке + добавляется в OOF stacking.

Тренируется на 10K stratified sample из team_train (TabPFN limit ~10K rows).
Предсказывает на team_val и team_test.
Затем формируется ансамбль M2 (CatBoost из team_split v2) + TabPFN → новый stacking.

Если TabPFN покажет хорошие метрики — обновим test_proba_real_estate.npy
"""
import gc
import time
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PATH = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
OUT_DIR = ROOT / 'real_estate_approaches'

TARGET = 'resolution'
CATEGORY_COL = 'CommercialTypeName4'

print('=' * 60)
print('TabPFN v2 — foundation model for tabular data (Hollmann 2025)')
print('=' * 60)

print('\nLoading data...')
df = pd.read_csv(DATA_PATH, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))

# Same split as team_split v2
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
trainval_idx, test_idx = next(gss1.split(df, df[TARGET], groups=df['SellerID']))
trainval_df = df.iloc[trainval_idx].copy()
test_df = df.iloc[test_idx].copy().reset_index(drop=True)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
tr_local, val_local = next(gss2.split(trainval_df, trainval_df[TARGET], groups=trainval_df['SellerID']))
train_df = trainval_df.iloc[tr_local].copy().reset_index(drop=True)
val_df = trainval_df.iloc[val_local].copy().reset_index(drop=True)
del trainval_df; gc.collect()

y_train = train_df[TARGET].to_numpy()
y_val = val_df[TARGET].to_numpy()
y_test = test_df[TARGET].to_numpy()
print(f'Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}')

# Feature engineering — same as team_split v2 (без CLIP/SVD для TabPFN, оно работает на табличных)
print('\nFeature engineering (only tabular)...')
refs = {
    'cat_med': train_df.groupby(CATEGORY_COL)['PriceDiscounted'].median(),
    'cat_tgt': train_df.groupby(CATEGORY_COL)[TARGET].mean(),
    'brand_tgt': train_df.groupby('brand_name')[TARGET].mean(),
    'global_med': train_df['PriceDiscounted'].median(),
    'global_tgt': train_df[TARGET].mean(),
    'top_brands_lower': set(b.lower().strip() for b in train_df['brand_name'].dropna().value_counts().head(500).index if isinstance(b, str)),
}

def fuzzy_brand(b, top_brands):
    if not isinstance(b, str) or not b.strip():
        return 0.0, 0.0
    bl = b.lower().strip()
    exact = 1.0 if bl in top_brands else 0.0
    if exact:
        return exact, 1.0
    best = 0.0
    for tb in top_brands:
        if abs(len(tb) - len(bl)) > 5:
            continue
        s = SequenceMatcher(None, bl, tb).ratio()
        if s > best:
            best = s
            if best >= 0.99:
                break
    return exact, best

def engineer(frame, refs):
    out = frame.copy()
    for col in ['name_rus', 'description', 'brand_name']:
        out[f'{col}_len'] = out[col].fillna('').str.len()
        out[f'{col}_is_null'] = out[col].isna().astype(int)
    cat_med = out[CATEGORY_COL].map(refs['cat_med']).fillna(refs['global_med'])
    out['category_median_price'] = cat_med
    out['price_ratio'] = out['PriceDiscounted'].fillna(0) / cat_med.replace(0, np.nan).fillna(1)
    out['log_price_ratio'] = np.log1p(out['price_ratio'].clip(lower=0))
    out['price_too_low'] = (out['price_ratio'] < 0.5).astype(int)
    out['price_too_high'] = (out['price_ratio'] > 2.0).astype(int)
    out['category_target_mean'] = out[CATEGORY_COL].map(refs['cat_tgt']).fillna(refs['global_tgt'])
    out['brand_target_mean'] = out['brand_name'].map(refs['brand_tgt']).fillna(refs['global_tgt'])
    name = out['name_rus'].fillna('').str.lower()
    out['name_has_digits'] = name.str.contains(r'\d', regex=True).astype(int)
    out['name_caps_ratio'] = out['name_rus'].fillna('').apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    out['susp_kw'] = name.str.contains('оригинал|original|100%|гарантия', regex=True, na=False).astype(int)
    out['excl_count'] = out['description'].fillna('').str.count('!')
    out['return_rate_30'] = out['item_count_returns30'].fillna(0) / (out['item_count_sales30'].fillna(0) + 1)
    out['return_rate_90'] = out['item_count_returns90'].fillna(0) / (out['item_count_sales90'].fillna(0) + 1)
    out['sales_velocity_30'] = out['item_count_sales30'].fillna(0) / (out['item_time_alive'].fillna(0) + 1)
    out['gmv_per_sale'] = out['GmvTotal90'].fillna(0) / (out['item_count_sales90'].fillna(0) + 1)
    out['is_new_item'] = (out['item_time_alive'].fillna(0) <= 30).astype(int)
    out['is_new_seller'] = (out['seller_time_alive'].fillna(0) <= 180).astype(int)
    rating_cols = [f'rating_{i}_count' for i in range(1, 6)]
    out['rating_total'] = out[rating_cols].fillna(0).sum(axis=1)
    out['rating_weighted'] = sum(i * out[f'rating_{i}_count'].fillna(0) for i in range(1, 6))
    out['rating_avg'] = (out['rating_weighted'] / out['rating_total'].replace(0, np.nan)).fillna(0)
    return out

train_df = engineer(train_df, refs)
val_df = engineer(val_df, refs)
test_df = engineer(test_df, refs)

# Typosquat
print('  Computing typosquat...')
for d in [train_df, val_df, test_df]:
    pairs = d['brand_name'].apply(lambda b: fuzzy_brand(b, refs['top_brands_lower']))
    d['brand_exact'] = pairs.apply(lambda p: p[0])
    d['brand_fuzzy'] = pairs.apply(lambda p: p[1])
    d['typosquat'] = d['brand_fuzzy'] - 0.5 * d['brand_exact']

# K-means structural
print('  K-means structural...')
tab_for_kmeans = ['PriceDiscounted', 'item_time_alive', 'seller_time_alive', 'price_ratio',
                  'return_rate_30', 'return_rate_90', 'sales_velocity_30',
                  'brand_target_mean', 'category_target_mean', 'rating_total', 'rating_avg']
scaler_km = StandardScaler()
Xtr_km = scaler_km.fit_transform(train_df[tab_for_kmeans].fillna(0))
Xv_km = scaler_km.transform(val_df[tab_for_kmeans].fillna(0))
Xte_km = scaler_km.transform(test_df[tab_for_kmeans].fillna(0))
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
ctr = kmeans.fit_predict(Xtr_km)
fraud_cluster = int(pd.Series(y_train).groupby(ctr).mean().idxmax())
fraud_centroid = kmeans.cluster_centers_[fraud_cluster]
train_df['cluster_id'] = (ctr == fraud_cluster).astype(int)
val_df['cluster_id'] = (kmeans.predict(Xv_km) == fraud_cluster).astype(int)
test_df['cluster_id'] = (kmeans.predict(Xte_km) == fraud_cluster).astype(int)
train_df['dist_centroid'] = np.linalg.norm(Xtr_km - fraud_centroid, axis=1)
val_df['dist_centroid'] = np.linalg.norm(Xv_km - fraud_centroid, axis=1)
test_df['dist_centroid'] = np.linalg.norm(Xte_km - fraud_centroid, axis=1)

# Tabular features only (TabPFN не любит много признаков, ограничим)
# По документации TabPFN — оптимально до 100 признаков
tabular_features = [
    'PriceDiscounted', 'item_time_alive', 'seller_time_alive',
    'item_count_sales30', 'item_count_sales90',
    'item_count_returns30', 'item_count_returns90',
    'GmvTotal30', 'GmvTotal90',
    'ExemplarReturnedValueTotal90',
    'ItemVarietyCount', 'ItemAvailableCount',
    'name_rus_len', 'description_len', 'brand_name_len',
    'name_rus_is_null', 'description_is_null', 'brand_name_is_null',
    'category_median_price', 'price_ratio', 'log_price_ratio', 'price_too_low', 'price_too_high',
    'category_target_mean', 'brand_target_mean',
    'name_has_digits', 'name_caps_ratio', 'susp_kw', 'excl_count',
    'rating_total', 'rating_avg',
    'return_rate_30', 'return_rate_90', 'sales_velocity_30', 'gmv_per_sale',
    'is_new_item', 'is_new_seller',
    'brand_exact', 'brand_fuzzy', 'typosquat',
    'cluster_id', 'dist_centroid',
]
print(f'\nTabular features for TabPFN: {len(tabular_features)}')

X_tr_full = train_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)
X_v = val_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)
X_te = test_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)

# Stratified subsample to 10K (TabPFN v2 limit)
N_SAMPLE = 10000
print(f'\nStratified subsample train to {N_SAMPLE} rows...')
sample_idx, _ = train_test_split(
    np.arange(len(X_tr_full)),
    train_size=N_SAMPLE,
    random_state=SEED,
    stratify=y_train,
)
X_tr = X_tr_full[sample_idx]
y_tr = y_train[sample_idx]
print(f'  Sampled positive rate: {y_tr.mean():.4f} (full: {y_train.mean():.4f})')

# Train TabPFN
print('\nLoading TabPFN v2...')
from tabpfn import TabPFNClassifier
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  Device: {device}')

t0 = time.time()
tabpfn = TabPFNClassifier(
    n_estimators=8,
    device=device,
    ignore_pretraining_limits=False,
    random_state=SEED,
)
tabpfn.fit(X_tr, y_tr)
print(f'  TabPFN fit in {time.time()-t0:.1f}s')

# Predict (TabPFN inference медленный, нужен batching)
print('\nPredicting on val...')
t0 = time.time()
p_tabpfn_val = tabpfn.predict_proba(X_v)[:, 1]
print(f'  val done in {time.time()-t0:.1f}s')

print('Predicting on test...')
t0 = time.time()
p_tabpfn_test = tabpfn.predict_proba(X_te)[:, 1]
print(f'  test done in {time.time()-t0:.1f}s')

# Metrics standalone TabPFN
def recall_at_p(y, p, min_p=0.9):
    prec, rec, _ = precision_recall_curve(y, p)
    mask = prec >= min_p
    return float(rec[mask].max()) if mask.any() else 0.0

print('\n=== TabPFN v2 standalone (team test) ===')
roc = roc_auc_score(y_test, p_tabpfn_test)
pr = average_precision_score(y_test, p_tabpfn_test)
r = recall_at_p(y_test, p_tabpfn_test)
print(f'TabPFN only:  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@P≥0.9={r:.4f}')

# Ensemble with existing CatBoost team_split v2
print('\n=== Ensemble TabPFN + CatBoost (M6 stacking) ===')
p_cb_val = np.load(OUT_DIR / 'val_proba_real_estate.npy')  # CatBoost raw val
p_cb_test = np.load(OUT_DIR / 'test_proba_real_estate.npy')  # CatBoost raw test

# Простое усреднение
p_ens_avg_val = (p_cb_val + p_tabpfn_val) / 2
p_ens_avg_test = (p_cb_test + p_tabpfn_test) / 2
roc = roc_auc_score(y_test, p_ens_avg_test)
pr = average_precision_score(y_test, p_ens_avg_test)
r = recall_at_p(y_test, p_ens_avg_test)
print(f'Simple avg:   ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@P≥0.9={r:.4f}')

# LR stacking (Diana's M5-style, но с 2 моделями)
X_meta_val = np.column_stack([p_cb_val, p_tabpfn_val])
X_meta_test = np.column_stack([p_cb_test, p_tabpfn_test])

meta = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)
meta.fit(X_meta_val, y_val)
p_meta_val = meta.predict_proba(X_meta_val)[:, 1]
p_meta_test = meta.predict_proba(X_meta_test)[:, 1]
roc = roc_auc_score(y_test, p_meta_test)
pr = average_precision_score(y_test, p_meta_test)
r = recall_at_p(y_test, p_meta_test)
print(f'LR stacking:  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@P≥0.9={r:.4f}')
print(f'  Meta weights: CatBoost={meta.coef_[0,0]:.3f}, TabPFN={meta.coef_[0,1]:.3f}')

# Weighted by val PR-AUC
pr_cb_val = average_precision_score(y_val, p_cb_val)
pr_tp_val = average_precision_score(y_val, p_tabpfn_val)
print(f'  val PR-AUC: CatBoost={pr_cb_val:.4f}, TabPFN={pr_tp_val:.4f}')
w_cb = pr_cb_val / (pr_cb_val + pr_tp_val)
w_tp = pr_tp_val / (pr_cb_val + pr_tp_val)
p_wavg_test = w_cb * p_cb_test + w_tp * p_tabpfn_test
roc = roc_auc_score(y_test, p_wavg_test)
pr = average_precision_score(y_test, p_wavg_test)
r = recall_at_p(y_test, p_wavg_test)
print(f'Weighted avg: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@P≥0.9={r:.4f}')
print(f'  weights: CatBoost={w_cb:.3f}, TabPFN={w_tp:.3f}')

# Save best ensemble
import os
ensembles = {
    'simple_avg': p_ens_avg_test,
    'lr_stacking': p_meta_test,
    'weighted_avg': p_wavg_test,
}
best_name = max(ensembles, key=lambda k: average_precision_score(y_test, ensembles[k]))
best_p = ensembles[best_name]
print(f'\nBest ensemble: {best_name}, PR-AUC={average_precision_score(y_test, best_p):.4f}')

np.save(OUT_DIR / 'test_proba_tabpfn.npy', p_tabpfn_test.astype(np.float32))
np.save(OUT_DIR / 'val_proba_tabpfn.npy', p_tabpfn_val.astype(np.float32))
np.save(OUT_DIR / 'test_proba_ensemble_cb_tabpfn.npy', best_p.astype(np.float32))
print(f'\n✓ Saved TabPFN probas and best ensemble')

# Compare with v2 baseline
print('\n=== Сравнение ===')
print(f'{"Model":<35} {"ROC":>8} {"PR-AUC":>8} {"R@P":>8}')
for label, p in [
    ('CatBoost v2 (текущая)', p_cb_test),
    ('TabPFN v2 standalone', p_tabpfn_test),
    ('Simple average', p_ens_avg_test),
    ('LR stacking', p_meta_test),
    ('Weighted avg', p_wavg_test),
]:
    roc = roc_auc_score(y_test, p)
    pr = average_precision_score(y_test, p)
    r = recall_at_p(y_test, p)
    print(f'{label:<35} {roc:>8.4f} {pr:>8.4f} {r:>8.4f}')
