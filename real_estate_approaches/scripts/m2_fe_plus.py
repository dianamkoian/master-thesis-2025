"""
M2-FE+ : расширенный feature engineering для мультимодальной модели Дианы.
Добавляет 6 новых групп признаков сверх mainline M2 (tab + CLIP + TF-IDF SVD + K-means):

  №1 Text statistics: длины, доли цифр, спецсимволов, uppercase
     (методологическая основа: FADAML Nguyen 2025 [4] — domain-text block)
  №2 Brand-level aggregates: count_per_brand, brand_seller_concentration, etc.
  №3 Title-description coherence: cosine TF-IDF между name и description
  №4 Interaction features: price_ratio × age, returns_rate × age, etc.
  №6 CLIP-derived structural: distance to CLIP train centroid, KMeans на CLIP пространстве
  №7 Categorical interactions: CommercialTypeName4 × is_null_brand, etc.

Цель: новая строка в Таблице 5.1 "real_estate M2-FE+ (расширенный FE)".
"""
import gc, time, os, json, warnings, re
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path('/Users/diana/master-thesis-2025')
DATA_CSV = ROOT / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
SPLITS = ROOT / 'Диана_ВКР_финал' / 'notebooks' / 'team_splits'
OUT_DIR = ROOT / 'Диана_ВКР_финал' / 'notebooks'
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'm2_fe_plus_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2-FE+ : расширенный feature engineering')

# Данные
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['name_rus_str'] = df['name_rus'].fillna('').astype(str)
df['description_str'] = df['description'].fillna('').astype(str)
df['brand_str'] = df['brand_name'].fillna('').astype(str)
df['text'] = (df['name_rus_str'] + ' ' + df['description_str'] + ' ' + df['brand_str']).str.slice(0, 512)

train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')

# ---------- Группа №1 Text statistics (FADAML domain-text, Nguyen 2025 [4]) ----------
log('Группа №1: text statistics')
def text_stats(s):
    return pd.DataFrame({
        f'{s}_char_len': df[f'{s}_str'].str.len(),
        f'{s}_word_count': df[f'{s}_str'].str.split().str.len().fillna(0),
        f'{s}_digit_ratio': df[f'{s}_str'].apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)),
        f'{s}_upper_ratio': df[f'{s}_str'].apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)),
        f'{s}_special_count': df[f'{s}_str'].apply(lambda x: sum(c in '!?*#$%&@<>/\\' for c in x)),
        f'{s}_avg_word_len': df[f'{s}_str'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0),
    })
for s in ['name_rus', 'description', 'brand']:
    stats = text_stats(s)
    for col in stats.columns:
        df[col] = stats[col].astype(np.float32)
log('  text statistics готово (18 признаков)')

# ---------- Группа №2 Brand-level aggregates ----------
log('Группа №2: brand-level aggregates')
df['brand_key'] = df['brand_str'].str.lower().str.strip()
# Aggregate ТОЛЬКО на train для предотвращения leakage
df_train = df.iloc[train_idx]
brand_stats_train = df_train.groupby('brand_key').agg(
    brand_n_items=('ItemID', 'count'),
    brand_n_sellers=('SellerID', 'nunique'),
    brand_mean_price=('PriceDiscounted', 'mean'),
    brand_std_price=('PriceDiscounted', 'std'),
).fillna(0).reset_index()
brand_stats_train['brand_seller_concentration'] = brand_stats_train['brand_n_items'] / np.maximum(brand_stats_train['brand_n_sellers'], 1)
df = df.merge(brand_stats_train, on='brand_key', how='left')
for c in ['brand_n_items', 'brand_n_sellers', 'brand_mean_price', 'brand_std_price', 'brand_seller_concentration']:
    df[c] = df[c].fillna(-1).astype(np.float32)
log('  brand aggregates готово (5 признаков)')

# ---------- Группа №3 Title-description coherence ----------
log('Группа №3: title-description coherence')
df['nd_word_overlap'] = df.apply(lambda r: len(set(r['name_rus_str'].lower().split()) & set(r['description_str'].lower().split())) / max(len(set(r['name_rus_str'].lower().split())), 1), axis=1).astype(np.float32)
df['nd_len_ratio'] = (df['description_char_len'] / np.maximum(df['name_rus_char_len'], 1)).astype(np.float32)
df['nd_both_present'] = ((df['name_rus_char_len'] > 5) & (df['description_char_len'] > 10)).astype(np.float32)
log('  coherence готово (3 признака)')

# ---------- Группа №4 Interactions ----------
log('Группа №4: interactions')
# Используем медианы для импутации
median_vals = {}
for c in ['PriceDiscounted', 'item_time_alive', 'seller_time_alive', 'item_count_returns30', 'item_count_sales30']:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
        median_vals[c] = df[c].median()

df['price_per_age'] = (df['PriceDiscounted'] / np.maximum(df['item_time_alive'], 1)).astype(np.float32)
df['returns_per_sales'] = (df['item_count_returns30'] / np.maximum(df['item_count_sales30'], 1)).astype(np.float32)
df['item_seller_age_ratio'] = (df['item_time_alive'] / np.maximum(df['seller_time_alive'], 1)).astype(np.float32)
df['age_brand_concentration'] = (df['item_time_alive'] * df['brand_seller_concentration']).astype(np.float32)
log('  interactions готово (4 признака)')

# ---------- CLIP ----------
log('Загрузка CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()
log(f'  CLIP: {len(clip_cols)} cols')

# ---------- Группа №6 CLIP-derived structural ----------
log('Группа №6: CLIP-derived structural')
# Train CLIP centroid (среднее по позитивам и негативам отдельно)
y_train = df['resolution'].values[train_idx]
clip_train = df[clip_cols].values[train_idx]
centroid_pos = clip_train[y_train == 1].mean(axis=0)
centroid_neg = clip_train[y_train == 0].mean(axis=0)

clip_all = df[clip_cols].values.astype(np.float32)
df['clip_dist_to_pos_centroid'] = np.linalg.norm(clip_all - centroid_pos, axis=1).astype(np.float32)
df['clip_dist_to_neg_centroid'] = np.linalg.norm(clip_all - centroid_neg, axis=1).astype(np.float32)
df['clip_pos_neg_ratio'] = (df['clip_dist_to_pos_centroid'] / np.maximum(df['clip_dist_to_neg_centroid'], 1e-6)).astype(np.float32)
df['clip_diff_to_pos_neg'] = (df['clip_dist_to_pos_centroid'] - df['clip_dist_to_neg_centroid']).astype(np.float32)

# K-means на CLIP пространстве (k=8)
from sklearn.cluster import KMeans
log('  K-means на CLIP пространстве (k=8)...')
km_clip = KMeans(n_clusters=8, random_state=42, n_init=5, max_iter=100).fit(clip_train)
df['clip_cluster_id'] = km_clip.predict(clip_all).astype(np.int8)
df['clip_dist_to_own_cluster'] = km_clip.transform(clip_all).min(axis=1).astype(np.float32)
log('  CLIP-derived готово (6 признаков)')

# ---------- TF-IDF SVD ----------
log('TF-IDF + SVD-50...')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
tfv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
texts_train = df.iloc[train_idx]['text'].values
tfv.fit(texts_train)
X_text_all = tfv.transform(df['text'].values)
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(tfv.transform(texts_train))
svd_arr = svd.transform(X_text_all).astype(np.float32)
for i in range(50):
    df[f'svd_{i}'] = svd_arr[:, i]
del svd_arr; gc.collect()
svd_cols = [f'svd_{i}' for i in range(50)]

# ---------- Tab cols ----------
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)
cat_cols = ['CommercialTypeName4']

# ---------- Группа №7 Categorical interactions ----------
log('Группа №7: categorical interactions')
df['cat_x_null_brand'] = (df['CommercialTypeName4'] + '_' + (df['brand_str'] == '').astype(int).astype(str)).astype('category')
cat_cols_extra = ['cat_x_null_brand']
for c in cat_cols_extra:
    df[c] = df[c].astype(str)
log('  categorical interactions готово (1 признак)')

# ---------- Финальный feature set ----------
new_feat_groups = {
    'text_stats': [c for c in df.columns if any(s in c for s in ['_char_len', '_word_count', '_digit_ratio', '_upper_ratio', '_special_count', '_avg_word_len'])],
    'brand_agg': ['brand_n_items', 'brand_n_sellers', 'brand_mean_price', 'brand_std_price', 'brand_seller_concentration'],
    'coherence': ['nd_word_overlap', 'nd_len_ratio', 'nd_both_present'],
    'interactions': ['price_per_age', 'returns_per_sales', 'item_seller_age_ratio', 'age_brand_concentration'],
    'clip_structural': ['clip_dist_to_pos_centroid', 'clip_dist_to_neg_centroid', 'clip_pos_neg_ratio', 'clip_diff_to_pos_neg', 'clip_cluster_id', 'clip_dist_to_own_cluster'],
}
new_feats_all = sum(new_feat_groups.values(), [])
log(f'Всего новых признаков (вне tab+CLIP+SVD): {len(new_feats_all)}')

# Финальный набор: tab + CLIP + SVD + новые группы + categorical interactions
features = tab_cols + clip_cols + svd_cols + new_feats_all
all_cat = cat_cols + cat_cols_extra
log(f'Итого признаков: {len(features)} (tab={len(tab_cols)} + clip={len(clip_cols)} + svd={len(svd_cols)} + new={len(new_feats_all)} + cat_extra={len(cat_cols_extra)})')

# ---------- Обучение M2-FE+ ----------
X = df[features + cat_cols_extra].copy()
y = df['resolution'].values
X_tr = X.iloc[train_idx].reset_index(drop=True); y_tr = y[train_idx]
X_va = X.iloc[val_idx].reset_index(drop=True); y_va = y[val_idx]
X_te = X.iloc[test_idx].reset_index(drop=True)

from catboost import CatBoostClassifier
log('Обучение M2-FE+ ...')
cb = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=42, early_stopping_rounds=100, cat_features=all_cat,
    thread_count=4, verbose=200, auto_class_weights='Balanced',
)
t_cb = time.time()
cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
log(f'  CatBoost обучен за {(time.time()-t_cb)/60:.1f} мин, best iter {cb.tree_count_}')

# Predict + isotonic
from sklearn.isotonic import IsotonicRegression
p_val_raw = cb.predict_proba(X_va)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_val_raw, y_va)
p_test_raw = cb.predict_proba(X_te)[:, 1].astype(np.float32)
p_test_cal = iso.predict(p_test_raw).astype(np.float32)

# Метрики
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

log('=' * 50)
roc = roc_auc_score(y_test_true, p_test_raw)
pr = average_precision_score(y_test_true, p_test_raw)
r09 = rap(y_test_true, p_test_raw)
log(f'  M2-FE+ RAW:  ROC={roc:.4f}  PR={pr:.4f}  R@P>=0.9={r09:.4f}')
roc_c = roc_auc_score(y_test_true, p_test_cal)
pr_c = average_precision_score(y_test_true, p_test_cal)
log(f'  M2-FE+ CAL:  ROC={roc_c:.4f}  PR={pr_c:.4f}  R@P>=0.9={rap(y_test_true, p_test_cal):.4f}')

# Контекст
p_m2 = np.load(OUT_DIR / 'test_proba_diana_team.npy')
p_no_te = np.load(OUT_DIR / 'test_proba_no_te.npy')
p_e5 = np.load(OUT_DIR / 'test_proba_diana_m2_e5.npy')
log(f'  M2 baseline:    PR={average_precision_score(y_test_true, p_m2):.4f}  R@P09={rap(y_test_true, p_m2):.4f}')
log(f'  M2_no_te:       PR={average_precision_score(y_test_true, p_no_te):.4f}  R@P09={rap(y_test_true, p_no_te):.4f}')
log(f'  M2-e5:          PR={average_precision_score(y_test_true, p_e5):.4f}  R@P09={rap(y_test_true, p_e5):.4f}')
log(f'  Прирост FE+ vs M2: ΔPR = {pr - average_precision_score(y_test_true, p_m2):+.4f}')

# Ансамбли с FE+
log('Ансамбли с FE+:')
for name, partner in [('M2', p_m2), ('M2_no_te', p_no_te), ('M2_e5', p_e5)]:
    mix = 0.5 * p_test_raw + 0.5 * partner
    log(f'  M2-FE+ + {name:10s}: PR={average_precision_score(y_test_true, mix):.4f}  R@P09={rap(y_test_true, mix):.4f}')

# Тройные
mix3a = (p_test_raw + p_m2 + p_e5) / 3
mix3b = (p_test_raw + p_m2 + p_no_te) / 3
mix3c = (p_test_raw + p_no_te + p_e5) / 3
log(f'  M2-FE+ + M2 + M2_e5:    PR={average_precision_score(y_test_true, mix3a):.4f}  R@P09={rap(y_test_true, mix3a):.4f}')
log(f'  M2-FE+ + M2 + M2_no_te: PR={average_precision_score(y_test_true, mix3b):.4f}  R@P09={rap(y_test_true, mix3b):.4f}')
log(f'  M2-FE+ + M2_no_te + M2_e5: PR={average_precision_score(y_test_true, mix3c):.4f}  R@P09={rap(y_test_true, mix3c):.4f}')

# 4-way с FE+
mix4 = (p_test_raw + p_m2 + p_no_te + p_e5) / 4
log(f'  4-way M2-FE+ + M2 + M2_no_te + M2_e5: PR={average_precision_score(y_test_true, mix4):.4f}  R@P09={rap(y_test_true, mix4):.4f}')

# SHAP топ-15 нового feature set
log('SHAP топ-15 признаков M2-FE+:')
fi = cb.get_feature_importance()
fi_pairs = sorted(zip(features + cat_cols_extra, fi), key=lambda x: -x[1])[:15]
for i, (f, w) in enumerate(fi_pairs, 1):
    log(f'  {i:>2d}. {f:35s} {w:.3f}')

# Save
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus.npy', p_test_raw)
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_calibrated.npy', p_test_cal)
log(f'Saved → test_proba_diana_m2_fe_plus.npy')

# CDSM v3 production artifact: сохраняем обученную CatBoost-модель как RMM-канал
CDSM_ART = ROOT / 'Диана_ВКР_финал' / 'counterfeit_service' / 'artifacts' / 'cdsm_v3'
CDSM_ART.mkdir(parents=True, exist_ok=True)
cb.save_model(str(CDSM_ART / 'rmm_catboost.cbm'))
log(f'Saved → {CDSM_ART / "rmm_catboost.cbm"} (RMM-канал CDSM v3, см. § 5.6.2 ВКР)')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
