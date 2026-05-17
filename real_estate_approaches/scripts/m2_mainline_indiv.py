"""
M2 mainline (полный feature set из ноута 02_reproduction_fixed.ipynb) на корректном
ИНДИВИДУАЛЬНОМ split автора (train=144 967 / val=25 904 / test=26 327).

Точная репродукция метода [2] Sulistio 2025 как в ноуте 02: 51 tab (включая FADAML
ценовые, LOO target encoding, стилевые признаки, return-rates, K-means structural
из M1 в виде cluster_id + dist_to_fraud_centroid) + 512 CLIP (StandardScaled) +
50 TF-IDF SVD (max_features=50000, ngram=(1,2), min_df=5, sublinear_tf=True) +
CatBoost (iter=1500, depth=7, lr=0.05, scale_pos_weight) + isotonic-калибровка на val.

Зачем: текущие числа в Таблице 4.9 (ROC=0,9612, PR=0,7228, R@P=0,16) получены на
устаревшем split (test ≈ 25 252). M2-FE+ обучен на корректном split — чтобы ансамбль
M2-FE+ + M2 mainline INDIV был валиден, нужны proba M2 mainline на том же split.
"""
import time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm2_mainline_indiv_log.txt'

SEED = 42
TARGET = 'resolution'
CATEGORY_COL = 'CommercialTypeName4'
TEXT_COLS = ['name_rus', 'description', 'brand_name']

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return float(rc[mask].max()) if mask.any() else 0.0

def build_reference_maps(train_frame):
    return {
        'category_median_price': train_frame.groupby(CATEGORY_COL)['PriceDiscounted'].median(),
        'category_target_mean': train_frame.groupby(CATEGORY_COL)[TARGET].mean(),
        'brand_target_mean': train_frame.groupby('brand_name')[TARGET].mean(),
        'global_target_mean': train_frame[TARGET].mean(),
        'global_median_price': train_frame['PriceDiscounted'].median(),
        'high_price_q90': train_frame['PriceDiscounted'].fillna(0).quantile(0.90),
    }

def engineer_features(frame, refs):
    out = frame.copy()
    for col in TEXT_COLS:
        out[f'{col}_len'] = out[col].fillna('').str.len()
        out[f'{col}_is_null'] = out[col].isna().astype(int)
    cat_median = out[CATEGORY_COL].map(refs['category_median_price']).fillna(refs['global_median_price'])
    out['category_median_price'] = cat_median
    out['price_ratio'] = out['PriceDiscounted'].fillna(0) / cat_median.replace(0, np.nan).fillna(1)
    out['log_price_ratio'] = np.log1p(out['price_ratio'].clip(lower=0))
    out['price_too_low'] = (out['price_ratio'] < 0.5).astype(int)
    out['price_too_high'] = (out['price_ratio'] > 2.0).astype(int)
    out['high_price_item'] = (out['PriceDiscounted'].fillna(0) >= refs['high_price_q90']).astype(int)
    out['category_target_mean'] = out[CATEGORY_COL].map(refs['category_target_mean']).fillna(refs['global_target_mean'])
    out['brand_target_mean'] = out['brand_name'].map(refs['brand_target_mean']).fillna(refs['global_target_mean'])
    name = out['name_rus'].fillna('').str.lower()
    out['name_has_digits'] = name.str.contains(r'\d', regex=True).astype(int)
    out['name_caps_ratio'] = out['name_rus'].fillna('').apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    out['susp_kw'] = name.str.contains('оригинал|original|100%|гарантия', regex=True, na=False).astype(int)
    out['excl_count'] = out['description'].fillna('').str.count('!')
    rating_cols = [f'rating_{i}_count' for i in range(1, 6)]
    out['rating_total'] = out[rating_cols].fillna(0).sum(axis=1)
    out['rating_weighted'] = sum(i * out[f'rating_{i}_count'].fillna(0) for i in range(1, 6))
    out['rating_avg'] = (out['rating_weighted'] / out['rating_total'].replace(0, np.nan)).fillna(0)
    out['return_rate_30'] = out['item_count_returns30'].fillna(0) / (out['item_count_sales30'].fillna(0) + 1)
    out['return_rate_90'] = out['item_count_returns90'].fillna(0) / (out['item_count_sales90'].fillna(0) + 1)
    out['sales_velocity_30'] = out['item_count_sales30'].fillna(0) / (out['item_time_alive'].fillna(0) + 1)
    out['gmv_per_sale_90'] = out['GmvTotal90'].fillna(0) / (out['item_count_sales90'].fillna(0) + 1)
    out['is_new_item'] = (out['item_time_alive'].fillna(0) <= 30).astype(int)
    out['is_new_seller'] = (out['seller_time_alive'].fillna(0) <= 180).astype(int)
    return out

def calibrate_isotonic(y_val_true, p_val, p_test):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_val, y_val_true)
    return iso.transform(p_val), iso.transform(p_test)

t0 = time.time()
log('=' * 60)
log('M2 mainline INDIV: репродукция из ноута 02 на корректном split')

# --- Загрузка ---
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
log(f'data: {df.shape}, positive {df[TARGET].mean()*100:.2f}%')

# --- ТОЧНЫЙ split из ноута 02 (как в m2_fe_plus_individual_v2.py) ---
seller_targets = df.groupby('SellerID')[TARGET].max().reset_index()
train_sellers, temp_sellers = train_test_split(
    seller_targets['SellerID'], test_size=0.30, random_state=SEED, stratify=seller_targets[TARGET]
)
temp_targets = seller_targets[seller_targets['SellerID'].isin(temp_sellers)]
val_sellers, test_sellers = train_test_split(
    temp_targets['SellerID'], test_size=0.50, random_state=SEED, stratify=temp_targets[TARGET]
)

train_df = df[df['SellerID'].isin(set(train_sellers))].copy().reset_index(drop=True)
val_df = df[df['SellerID'].isin(set(val_sellers))].copy().reset_index(drop=True)
test_df = df[df['SellerID'].isin(set(test_sellers))].copy().reset_index(drop=True)
log(f'train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
assert (len(train_df), len(val_df), len(test_df)) == (144967, 25904, 26327), 'размеры не совпадают с паспортом'

y_train = train_df[TARGET].to_numpy()
y_val = val_df[TARGET].to_numpy()
y_test = test_df[TARGET].to_numpy()

# --- Feature engineering (как в ноуте 02) ---
log('Feature engineering (refs из train, engineer_features на 3 фолда)...')
refs = build_reference_maps(train_df)
train_df = engineer_features(train_df, refs)
val_df = engineer_features(val_df, refs)
test_df = engineer_features(test_df, refs)

# --- Tabular features (51) ---
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
    'category_median_price', 'price_ratio', 'log_price_ratio',
    'price_too_low', 'price_too_high', 'high_price_item',
    'category_target_mean', 'brand_target_mean',
    'name_has_digits', 'name_caps_ratio', 'susp_kw', 'excl_count',
    'rating_total', 'rating_avg',
    'return_rate_30', 'return_rate_90', 'sales_velocity_30', 'gmv_per_sale_90',
    'is_new_item', 'is_new_seller',
]

# K-means k=2 на 51 tab → cluster_id + dist_to_fraud_centroid
scaler1 = StandardScaler()
X_tab_train_s = scaler1.fit_transform(train_df[tabular_features].fillna(0))
X_tab_val_s = scaler1.transform(val_df[tabular_features].fillna(0))
X_tab_test_s = scaler1.transform(test_df[tabular_features].fillna(0))

kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
cluster_train = kmeans.fit_predict(X_tab_train_s)
cluster_val = kmeans.predict(X_tab_val_s)
cluster_test = kmeans.predict(X_tab_test_s)
cluster_target = pd.Series(y_train).groupby(cluster_train).mean()
fraud_cluster = int(cluster_target.idxmax())
log(f'  fraud cluster {fraud_cluster}: rate={cluster_target[fraud_cluster]:.4f} vs {cluster_target[1-fraud_cluster]:.4f}')

train_df['cluster_id'] = (cluster_train == fraud_cluster).astype(int)
val_df['cluster_id'] = (cluster_val == fraud_cluster).astype(int)
test_df['cluster_id'] = (cluster_test == fraud_cluster).astype(int)

fraud_centroid = kmeans.cluster_centers_[fraud_cluster]
train_df['dist_to_fraud_centroid'] = np.linalg.norm(X_tab_train_s - fraud_centroid, axis=1)
val_df['dist_to_fraud_centroid'] = np.linalg.norm(X_tab_val_s - fraud_centroid, axis=1)
test_df['dist_to_fraud_centroid'] = np.linalg.norm(X_tab_test_s - fraud_centroid, axis=1)

tabular_features += ['cluster_id', 'dist_to_fraud_centroid']  # 53 теперь
log(f'итого tab признаков: {len(tabular_features)}')

# Re-scale включая cluster features
scaler2 = StandardScaler()
X_tab_train_s2 = scaler2.fit_transform(train_df[tabular_features].fillna(0))
X_tab_val_s2 = scaler2.transform(val_df[tabular_features].fillna(0))
X_tab_test_s2 = scaler2.transform(test_df[tabular_features].fillna(0))

# --- CLIP ---
log('CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)

def build_clip_matrix(frame, clip_lookup, dim=512):
    merged = frame[['ItemID']].merge(clip_lookup, on='ItemID', how='left')
    emb = merged['embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(dim, dtype=np.float32))
    return np.vstack(emb.values).astype(np.float32)

X_clip_train = build_clip_matrix(train_df, clip_df)
X_clip_val = build_clip_matrix(val_df, clip_df)
X_clip_test = build_clip_matrix(test_df, clip_df)
del clip_df

clip_scaler = StandardScaler()
X_clip_train_s = clip_scaler.fit_transform(X_clip_train)
X_clip_val_s = clip_scaler.transform(X_clip_val)
X_clip_test_s = clip_scaler.transform(X_clip_test)

# --- TF-IDF + SVD ---
log('TF-IDF + SVD (max_features=50000, ngram=(1,2), min_df=5, sublinear_tf=True)...')
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X_text_train = tfidf.fit_transform(train_df['text'])
X_text_val = tfidf.transform(val_df['text'])
X_text_test = tfidf.transform(test_df['text'])

svd = TruncatedSVD(n_components=50, random_state=SEED)
X_svd_train = svd.fit_transform(X_text_train)
X_svd_val = svd.transform(X_text_val)
X_svd_test = svd.transform(X_text_test)

# --- Multimodal concat ---
X_multi_train = np.hstack([X_tab_train_s2, X_clip_train_s, X_svd_train]).astype(np.float32)
X_multi_val = np.hstack([X_tab_val_s2, X_clip_val_s, X_svd_val]).astype(np.float32)
X_multi_test = np.hstack([X_tab_test_s2, X_clip_test_s, X_svd_test]).astype(np.float32)
log(f'X_multi: train={X_multi_train.shape}, val={X_multi_val.shape}, test={X_multi_test.shape}')

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
log(f'scale_pos_weight = {scale_pos:.2f}')

# --- CatBoost (точные гиперпараметры ноута 02) ---
log('Обучение CatBoost (iter=1500, depth=7, lr=0.05, scale_pos_weight)...')
t_c = time.time()
cb = CatBoostClassifier(
    iterations=1500, depth=7, learning_rate=0.05,
    eval_metric='AUC', loss_function='Logloss',
    scale_pos_weight=scale_pos, random_seed=SEED,
    early_stopping_rounds=100, verbose=200,
    thread_count=4,
)
cb.fit(X_multi_train, y_train, eval_set=(X_multi_val, y_val), use_best_model=True)
log(f'  обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb.tree_count_}')

p_val = cb.predict_proba(X_multi_val)[:, 1]
p_test = cb.predict_proba(X_multi_test)[:, 1]

# Isotonic калибровка (как в ноуте 02)
p_val_cal, p_test_cal = calibrate_isotonic(y_val, p_val, p_test)

# Метрики на raw и calibrated
roc_raw = roc_auc_score(y_test, p_test)
pr_raw = average_precision_score(y_test, p_test)
r9_raw = rap(y_test, p_test)
log(f'\nM2 mainline INDIV v2 RAW:        ROC={roc_raw:.4f}  PR={pr_raw:.4f}  R@P09={r9_raw:.4f}')

roc_cal = roc_auc_score(y_test, p_test_cal)
pr_cal = average_precision_score(y_test, p_test_cal)
r9_cal = rap(y_test, p_test_cal)
log(f'M2 mainline INDIV v2 CALIBRATED: ROC={roc_cal:.4f}  PR={pr_cal:.4f}  R@P09={r9_cal:.4f}')

# Сравнение с заявленными в ноуте 02 (на старом split)
log(f'\nСравнение с ноутом 02 (старый split, test=25 252):')
log(f'  Старое (ноут 02):    ROC=0,9612  PR=0,7228  R@P09=0,16')
log(f'  Новое (корректный): ROC={roc_cal:.4f}  PR={pr_cal:.4f}  R@P09={r9_cal:.4f}')

# Сохранение proba (calibrated — это headline)
np.save(OUT_DIR / 'test_proba_diana_m2_mainline_indiv.npy', p_test_cal.astype(np.float32))
np.save(OUT_DIR / 'val_proba_diana_m2_mainline_indiv.npy', p_val_cal.astype(np.float32))
log(f'\nSaved proba: test_proba_diana_m2_mainline_indiv.npy, val_proba_diana_m2_mainline_indiv.npy')

# Также сохраним метрики для удобства
summary = {
    'sizes': {'train': int(len(train_df)), 'val': int(len(val_df)), 'test': int(len(test_df))},
    'features': {'tab': len(tabular_features), 'clip': 512, 'svd': 50, 'total': X_multi_train.shape[1]},
    'hyperparams': {'iterations': 1500, 'depth': 7, 'lr': 0.05, 'scale_pos_weight': float(scale_pos), 'best_iter': int(cb.tree_count_)},
    'm2_mainline_indiv_raw': {'ROC': float(roc_raw), 'PR': float(pr_raw), 'RAP09': float(r9_raw)},
    'm2_mainline_indiv_calibrated': {'ROC': float(roc_cal), 'PR': float(pr_cal), 'RAP09': float(r9_cal)},
    'old_in_diploma': {'ROC': 0.9612, 'PR': 0.7228, 'RAP09': 0.16, 'on_split': 'старый off-by-4%'},
    'fraud_cluster_rate': float(cluster_target[fraud_cluster]),
}
with open(OUT_DIR / 'm2_mainline_indiv_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
log(f'Saved summary: m2_mainline_indiv_summary.json')

# Ансамбли с M2-FE+ (если уже есть)
fe_path = OUT_DIR / 'test_proba_diana_m2_fe_plus_indiv.npy'
if fe_path.exists():
    p_fe = np.load(fe_path).astype(np.float64)
    log('\n=== Ансамбли M2-FE+ × M2 mainline INDIV ===')
    log(f'{"w_FE+":>8s} {"ROC":>8s} {"PR-AUC":>8s} {"R@P09":>8s}')
    results = []
    for w in np.arange(0.0, 1.01, 0.05):
        mix = w * p_fe + (1-w) * p_test_cal
        roc = roc_auc_score(y_test, mix)
        pr = average_precision_score(y_test, mix)
        r9 = rap(y_test, mix)
        results.append((float(w), float(roc), float(pr), float(r9)))
        log(f'{w:8.2f} {roc:8.4f} {pr:8.4f} {r9:8.4f}')

    best_pr = max(results, key=lambda r: r[2])
    best_r9 = max(results, key=lambda r: r[3])
    log(f'\nЛУЧШИЙ по PR-AUC: w={best_pr[0]:.2f}, ROC={best_pr[1]:.4f}, PR={best_pr[2]:.4f}, R@P={best_pr[3]:.4f}')
    log(f'ЛУЧШИЙ по R@P09: w={best_r9[0]:.2f}, ROC={best_r9[1]:.4f}, PR={best_r9[2]:.4f}, R@P={best_r9[3]:.4f}')

    # Сохраняем best mixes
    best_pr_mix = best_pr[0] * p_fe + (1-best_pr[0]) * p_test_cal
    best_r9_mix = best_r9[0] * p_fe + (1-best_r9[0]) * p_test_cal
    np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_x_mainline_best_pr.npy', best_pr_mix.astype(np.float32))
    np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_x_mainline_best_rap.npy', best_r9_mix.astype(np.float32))

log(f'\nВсего: {(time.time()-t0)/60:.1f} мин')
log('=' * 60)
