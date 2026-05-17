"""
Team split — версия v3 (m2_team_v3.py).

Отличия от mainline-M2 ноута 02 — три кросс-доменных компонента из домена
недвижимости автора:
  - cluster_id, dist_centroid — K-means structural из Mohd Amin 2024 [1];
  - clip_cat_sim — cosine-similarity CLIP-эмбеддинга карточки с центроидом
    её категории. Это естественное расширение K-means distance-to-centroid
    [1] на CLIP-пространство (поскольку CLIP-эмбеддинги L2-нормированы,
    cosine-distance монотонно связан с евклидовым).

Итого: 54 tab + 512 CLIP + 100 SVD = 666 признаков.
Сохраняет:
  - notebooks/test_proba_diana_team_v3.npy (RAW)
  - notebooks/test_proba_diana_team_v3_calibrated.npy
  - notebooks/val_proba_diana_team_v3.npy
"""
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT = Path('/Users/diana/master-thesis-2025')
DATA_PATH = ROOT / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PATH = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
OUT_DIR = ROOT / 'Диана_ВКР_финал' / 'notebooks'

TARGET = 'resolution'
CATEGORY_COL = 'CommercialTypeName4'

print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
print(f'Loaded: {df.shape}, positive {df[TARGET].mean():.4f}')

# Split exactly как Соня
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
assert (len(train_df), len(val_df), len(test_df)) == (69453, 69335, 58410)

# Feature engineering
print('\nFeature engineering...')
refs = {
    'cat_med': train_df.groupby(CATEGORY_COL)['PriceDiscounted'].median(),
    'cat_tgt': train_df.groupby(CATEGORY_COL)[TARGET].mean(),
    'brand_tgt': train_df.groupby('brand_name')[TARGET].mean(),
    'global_med': train_df['PriceDiscounted'].median(),
    'global_tgt': train_df[TARGET].mean(),
}


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

t0 = time.time()
train_df = engineer(train_df, refs)
val_df = engineer(val_df, refs)
test_df = engineer(test_df, refs)
print(f'  basic done in {time.time()-t0:.1f}s')

# CLIP embeddings
print('\nLoading CLIP embeddings...')
clip_df = pd.read_parquet(CLIP_PATH)

def build_clip_matrix(frame, lookup, dim=512):
    merged = frame[['ItemID']].merge(lookup, on='ItemID', how='left')
    emb = merged['embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(dim, dtype=np.float32))
    return np.vstack(emb.values).astype(np.float32)

X_clip_train = build_clip_matrix(train_df, clip_df)
X_clip_val = build_clip_matrix(val_df, clip_df)
X_clip_test = build_clip_matrix(test_df, clip_df)
del clip_df; gc.collect()
print(f'  CLIP shapes: {X_clip_train.shape}, {X_clip_val.shape}, {X_clip_test.shape}')

clip_scaler = StandardScaler()
X_clip_train_s = clip_scaler.fit_transform(X_clip_train).astype(np.float32)
X_clip_val_s = clip_scaler.transform(X_clip_val).astype(np.float32)
X_clip_test_s = clip_scaler.transform(X_clip_test).astype(np.float32)

# clip_cat_sim — расширение K-means distance-to-centroid из Mohd Amin 2024 [1]
# на CLIP-пространство: cosine-similarity между L2-нормированным CLIP-эмбеддингом
# карточки и центроидом её категории, вычисленным на train.
print('  Computing clip_cat_sim (K-means [1] extension to CLIP-space)...')

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)

X_clip_train_n = l2norm(X_clip_train)
X_clip_val_n = l2norm(X_clip_val)
X_clip_test_n = l2norm(X_clip_test)
del X_clip_train, X_clip_val, X_clip_test; gc.collect()

train_cats = train_df[CATEGORY_COL].fillna('unknown').values
category_centroids = {}
for cat in np.unique(train_cats):
    mask = train_cats == cat
    if mask.sum() > 0:
        c = X_clip_train_n[mask].mean(axis=0)
        category_centroids[cat] = c / (np.linalg.norm(c) + 1e-8)


def clip_cat_sim_compute(frame, X_n):
    cats = frame[CATEGORY_COL].fillna('unknown').values
    sims = np.zeros(len(cats), dtype=np.float32)
    for i, cat in enumerate(cats):
        if cat in category_centroids:
            sims[i] = float(X_n[i] @ category_centroids[cat])
    return sims


train_df['clip_cat_sim'] = clip_cat_sim_compute(train_df, X_clip_train_n)
val_df['clip_cat_sim'] = clip_cat_sim_compute(val_df, X_clip_val_n)
test_df['clip_cat_sim'] = clip_cat_sim_compute(test_df, X_clip_test_n)
del X_clip_train_n, X_clip_val_n, X_clip_test_n; gc.collect()

# K-means
print('  K-means clustering...')
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
del Xtr_km, Xv_km, Xte_km; gc.collect()

# TF-IDF SVD-100 (вместо e5)
print('\nTF-IDF SVD-100...')
t0 = time.time()
tfidf = TfidfVectorizer(max_features=80000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_text_train = tfidf.fit_transform(train_df['text'])
X_text_val = tfidf.transform(val_df['text'])
X_text_test = tfidf.transform(test_df['text'])
svd = TruncatedSVD(n_components=100, random_state=SEED)
X_svd_train = svd.fit_transform(X_text_train).astype(np.float32)
X_svd_val = svd.transform(X_text_val).astype(np.float32)
X_svd_test = svd.transform(X_text_test).astype(np.float32)
del X_text_train, X_text_val, X_text_test, tfidf, svd; gc.collect()
print(f'  TF-IDF SVD done in {time.time()-t0:.1f}s. shape {X_svd_train.shape}')

# Final assembly
tabular_features = [
    'PriceDiscounted', 'item_time_alive', 'seller_time_alive',
    'item_count_sales7', 'item_count_sales30', 'item_count_sales90',
    'item_count_returns7', 'item_count_returns30', 'item_count_returns90',
    'GmvTotal7', 'GmvTotal30', 'GmvTotal90',
    'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30', 'ExemplarAcceptedCountTotal90',
    'OrderAcceptedCountTotal7', 'OrderAcceptedCountTotal30', 'OrderAcceptedCountTotal90',
    'ExemplarReturnedCountTotal7', 'ExemplarReturnedCountTotal30', 'ExemplarReturnedCountTotal90',
    'ExemplarReturnedValueTotal7', 'ExemplarReturnedValueTotal30', 'ExemplarReturnedValueTotal90',
    'ItemVarietyCount', 'ItemAvailableCount',
    'name_rus_len', 'description_len', 'brand_name_len',
    'name_rus_is_null', 'description_is_null', 'brand_name_is_null',
    'category_median_price', 'price_ratio', 'log_price_ratio', 'price_too_low', 'price_too_high',
    'category_target_mean', 'brand_target_mean',
    'name_has_digits', 'name_caps_ratio', 'susp_kw', 'excl_count',
    'rating_total', 'rating_avg',
    'return_rate_30', 'return_rate_90', 'sales_velocity_30', 'gmv_per_sale',
    'is_new_item', 'is_new_seller',
    'cluster_id', 'dist_centroid', 'clip_cat_sim',
]
print(f'\nTabular features: {len(tabular_features)}')

dense_train = train_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)
dense_val = val_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)
dense_test = test_df[tabular_features].fillna(0).to_numpy(dtype=np.float32)
X_train = np.hstack([dense_train, X_clip_train_s, X_svd_train])
X_val = np.hstack([dense_val, X_clip_val_s, X_svd_val])
X_test = np.hstack([dense_test, X_clip_test_s, X_svd_test])
print(f'Final shape: {X_train.shape}, {X_val.shape}, {X_test.shape}')
del dense_train, dense_val, dense_test, X_clip_train_s, X_clip_val_s, X_clip_test_s, X_svd_train, X_svd_val, X_svd_test, train_df
gc.collect()

# Train
print('\nTraining CatBoost...')
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f'  scale_pos_weight ≈ {scale_pos:.2f}')
t0 = time.time()
model = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.04,
    eval_metric='AUC', scale_pos_weight=scale_pos, random_seed=SEED,
    early_stopping_rounds=150, verbose=200, bagging_temperature=0.5, border_count=128,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
print(f'\nTraining done in {time.time()-t0:.0f}s, best iter = {model.tree_count_}')

# Predict + calibrate
p_val_raw = model.predict_proba(X_val)[:, 1]
p_test_raw = model.predict_proba(X_test)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val_raw, y_val)
p_val = iso.transform(p_val_raw)
p_test = iso.transform(p_test_raw)

def recall_at_p(y, p, min_p=0.9):
    prec, rec, _ = precision_recall_curve(y, p)
    mask = prec >= min_p
    return float(rec[mask].max()) if mask.any() else 0.0

print('\n=== Метрики на team_test (58 410 объектов) ===')
for label, p in [('Raw', p_test_raw), ('Calibrated', p_test)]:
    print(f'{label:<12} ROC-AUC={roc_auc_score(y_test, p):.4f}  PR-AUC={average_precision_score(y_test, p):.4f}  R@P≥0.9={recall_at_p(y_test, p):.4f}')

# Bootstrap CI
print('\nBootstrap 95% CI:')
def bootstrap_ci(y, p, fn, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            continue
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
for label, fn in [('ROC-AUC ', roc_auc_score), ('PR-AUC  ', average_precision_score), ('R@P≥0.9 ', lambda y, p: recall_at_p(y, p))]:
    m, lo, hi = bootstrap_ci(y_test, p_test, fn)
    print(f'  {label} mean={m:.4f}  CI=[{lo:.4f}, {hi:.4f}]')

# Save — RAW + Calibrated в *_v3.npy
OUT_NB = ROOT / 'Диана_ВКР_финал' / 'notebooks'
out_proba = OUT_NB / 'test_proba_diana_team_v3.npy'
np.save(out_proba, p_test_raw.astype(np.float32))
np.save(OUT_NB / 'val_proba_diana_team_v3.npy', p_val_raw.astype(np.float32))
np.save(OUT_NB / 'test_proba_diana_team_v3_calibrated.npy', p_test.astype(np.float32))

feature_names = tabular_features + [f'clip_{i}' for i in range(512)] + [f'svd_{i}' for i in range(100)]
fi = model.get_feature_importance()
importance_df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=False)
importance_df.head(30).to_csv(OUT_DIR / 'feature_importance_team.csv', index=False)

print(f'\n✓ Saved test probas: {out_proba}')
print('\nTop-15 features:')
for _, row in importance_df.head(15).iterrows():
    print(f'  {row["feature"]:<40} {row["importance"]:.3f}')
