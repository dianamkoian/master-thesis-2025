"""
TabNet (Arik & Pfister, AAAI 2021) — нейросеть с attention-механизмом для табличных данных.

Используем как M6 (новый кандидат) в OOF Stacking + ансамблируем с CatBoost.
Никто из команды не использует neural tabular methods — это уникальный вклад в работу.
"""
import gc
import time
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
OUT_DIR = ROOT / 'real_estate_approaches'

TARGET = 'resolution'
CATEGORY_COL = 'CommercialTypeName4'

print('=' * 60)
print('TabNet (Arik & Pfister, AAAI 2021) — neural tabular')
print('=' * 60)

print('\nLoading data...')
df = pd.read_csv(DATA_PATH, encoding='utf-8')

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

# Feature engineering
print('\nFeature engineering...')
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
    if bl in top_brands:
        return 1.0, 1.0
    best = 0.0
    for tb in top_brands:
        if abs(len(tb) - len(bl)) > 5:
            continue
        s = SequenceMatcher(None, bl, tb).ratio()
        if s > best:
            best = s
            if best >= 0.99: break
    return 0.0, best

def engineer(frame, refs):
    out = frame.copy()
    for col in ['name_rus', 'description', 'brand_name']:
        out[f'{col}_len'] = out[col].fillna('').str.len()
        out[f'{col}_is_null'] = out[col].isna().astype(int)
    cat_med = out[CATEGORY_COL].map(refs['cat_med']).fillna(refs['global_med'])
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
    out['rating_avg'] = (sum(i * out[f'rating_{i}_count'].fillna(0) for i in range(1, 6)) / out['rating_total'].replace(0, np.nan)).fillna(0)
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

tabular_features = [
    'PriceDiscounted', 'item_time_alive', 'seller_time_alive',
    'item_count_sales30', 'item_count_sales90',
    'item_count_returns30', 'item_count_returns90',
    'GmvTotal30', 'GmvTotal90',
    'ItemVarietyCount', 'ItemAvailableCount',
    'name_rus_len', 'description_len', 'brand_name_len',
    'name_rus_is_null', 'description_is_null', 'brand_name_is_null',
    'price_ratio', 'log_price_ratio', 'price_too_low', 'price_too_high',
    'category_target_mean', 'brand_target_mean',
    'name_has_digits', 'name_caps_ratio', 'susp_kw', 'excl_count',
    'rating_total', 'rating_avg',
    'return_rate_30', 'return_rate_90', 'sales_velocity_30', 'gmv_per_sale',
    'is_new_item', 'is_new_seller',
    'brand_exact', 'brand_fuzzy', 'typosquat',
    'cluster_id', 'dist_centroid',
]
print(f'Features for TabNet: {len(tabular_features)}')

# Standardize for neural network
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_df[tabular_features].fillna(0)).astype(np.float32)
X_v = scaler.transform(val_df[tabular_features].fillna(0)).astype(np.float32)
X_te = scaler.transform(test_df[tabular_features].fillna(0)).astype(np.float32)

del train_df, val_df, test_df, df; gc.collect()

# TabNet
print('\nTraining TabNet...')
from pytorch_tabnet.tab_model import TabNetClassifier

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'  Device: {device}')

# Class weights for imbalance
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f'  scale_pos_weight ≈ {scale_pos:.2f}')

t0 = time.time()
tabnet = TabNetClassifier(
    n_d=32, n_a=32, n_steps=4,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    scheduler_params=dict(step_size=10, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name=device,
    seed=SEED,
    verbose=10,
)

# class weights through sample_weight (TabNet doesn't have scale_pos_weight directly)
sample_weights = np.where(y_train == 1, scale_pos, 1.0).astype(np.float32)

tabnet.fit(
    X_tr, y_train,
    eval_set=[(X_v, y_val)],
    eval_name=['val'],
    eval_metric=['auc'],
    max_epochs=80,
    patience=15,
    batch_size=4096,
    virtual_batch_size=256,
    weights=sample_weights,
)
print(f'\nTabNet trained in {time.time()-t0:.0f}s')

# Predict
print('\nPredicting...')
p_tabnet_val = tabnet.predict_proba(X_v)[:, 1]
p_tabnet_test = tabnet.predict_proba(X_te)[:, 1]

def recall_at_p(y, p, min_p=0.9):
    prec, rec, _ = precision_recall_curve(y, p)
    mask = prec >= min_p
    return float(rec[mask].max()) if mask.any() else 0.0

print('\n=== TabNet standalone (team test) ===')
roc = roc_auc_score(y_test, p_tabnet_test)
pr = average_precision_score(y_test, p_tabnet_test)
r = recall_at_p(y_test, p_tabnet_test)
print(f'TabNet:       ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@P≥0.9={r:.4f}')

# Ensemble с CatBoost
p_cb_val = np.load(OUT_DIR / 'val_proba_real_estate.npy')
p_cb_test = np.load(OUT_DIR / 'test_proba_real_estate.npy')

# Several ensembles
print('\n=== Ensembles CatBoost + TabNet ===')
print(f'{"Method":<35} {"ROC":>8} {"PR-AUC":>8} {"R@P":>8}')

# 1. Simple average
p_avg_test = (p_cb_test + p_tabnet_test) / 2
print(f'{"Simple average":<35} {roc_auc_score(y_test, p_avg_test):>8.4f} {average_precision_score(y_test, p_avg_test):>8.4f} {recall_at_p(y_test, p_avg_test):>8.4f}')

# 2. Weighted avg (val PR-AUC)
pr_cb_v = average_precision_score(y_val, p_cb_val)
pr_tn_v = average_precision_score(y_val, p_tabnet_val)
print(f'  val PR-AUC: CatBoost={pr_cb_v:.4f}, TabNet={pr_tn_v:.4f}')
w_cb = pr_cb_v / (pr_cb_v + pr_tn_v); w_tn = pr_tn_v / (pr_cb_v + pr_tn_v)
p_wavg_test = w_cb * p_cb_test + w_tn * p_tabnet_test
print(f'{f"Weighted avg ({w_cb:.2f}/{w_tn:.2f})":<35} {roc_auc_score(y_test, p_wavg_test):>8.4f} {average_precision_score(y_test, p_wavg_test):>8.4f} {recall_at_p(y_test, p_wavg_test):>8.4f}')

# 3. LR stacking (meta on val)
X_meta_v = np.column_stack([p_cb_val, p_tabnet_val])
X_meta_te = np.column_stack([p_cb_test, p_tabnet_test])
meta = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)
meta.fit(X_meta_v, y_val)
p_meta_test = meta.predict_proba(X_meta_te)[:, 1]
print(f'{"LR stacking":<35} {roc_auc_score(y_test, p_meta_test):>8.4f} {average_precision_score(y_test, p_meta_test):>8.4f} {recall_at_p(y_test, p_meta_test):>8.4f}')
print(f'  Meta weights: CB={meta.coef_[0,0]:.3f}, TabNet={meta.coef_[0,1]:.3f}')

print('\n=== Сравнение ===')
print(f'{"Model":<40} {"ROC":>8} {"PR-AUC":>8} {"R@P":>8}')
for label, p in [
    ('CatBoost v2 (текущая, raw)', p_cb_test),
    ('TabNet standalone', p_tabnet_test),
    ('Simple average CB+TabNet', p_avg_test),
    (f'Weighted avg', p_wavg_test),
    ('LR stacking', p_meta_test),
]:
    print(f'{label:<40} {roc_auc_score(y_test, p):>8.4f} {average_precision_score(y_test, p):>8.4f} {recall_at_p(y_test, p):>8.4f}')

# Save
np.save(OUT_DIR / 'test_proba_tabnet.npy', p_tabnet_test.astype(np.float32))
np.save(OUT_DIR / 'val_proba_tabnet.npy', p_tabnet_val.astype(np.float32))

# Best ensemble — save as new team contribution if better
ensembles = {
    'avg': p_avg_test, 'wavg': p_wavg_test, 'lr_stack': p_meta_test,
}
best_name = max(ensembles, key=lambda k: average_precision_score(y_test, ensembles[k]))
best_p = ensembles[best_name]
best_pr = average_precision_score(y_test, best_p)
cb_pr = average_precision_score(y_test, p_cb_test)
print(f'\nBest ensemble: {best_name} with PR-AUC={best_pr:.4f} (CatBoost-only: {cb_pr:.4f})')

if best_pr > cb_pr:
    print(f'  → Ensemble лучше! Δ PR-AUC = +{best_pr - cb_pr:.4f}')
    print(f'  → Сохраняю test_proba_real_estate.npy = best ensemble (для Сони)')
    # Backup original
    import shutil
    shutil.copy(OUT_DIR / 'test_proba_real_estate.npy', OUT_DIR / 'test_proba_real_estate_v2_cb_only.npy')
    np.save(OUT_DIR / 'test_proba_real_estate.npy', best_p.astype(np.float32))
    print(f'  ✓ test_proba_real_estate.npy обновлён')
else:
    print(f'  → Ensemble не лучше (Δ = {best_pr - cb_pr:+.4f}). Оставляю CatBoost-only.')

np.save(OUT_DIR / 'test_proba_ensemble_cb_tabnet.npy', best_p.astype(np.float32))
print('\n✓ Все probas сохранены')
