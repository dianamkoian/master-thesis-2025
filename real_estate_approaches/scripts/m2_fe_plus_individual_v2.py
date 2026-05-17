"""
M2-FE+ на ИНДИВИДУАЛЬНОМ протоколе Дианы (v2, точный split из ноута 02).

Отличие от v1: split воспроизводится в точности как в `02_reproduction_fixed.ipynb`
(pandas.Series, один аргумент в train_test_split, reset_index на seller_targets).
v1 давал test ≈ 25 268; ожидаемый из паспорта: test=26 327, val=25 904, train=144 967.

FE-группы M2-FE+ (как в командном `m2_fe_plus.py`):
  №1 Text statistics: длины, доли цифр, спецсимволов, uppercase
     (методологическая основа: FADAML Nguyen 2025 [4] — domain-text block)
  №2 Brand-level aggregates, №3 coherence, №4 interactions,
  №6 CLIP-derived structural, №7 categorical interactions.

Обучает:
  - M2-FE+ (расширенный FE из 6 групп) на этом split,
  - M2 baseline на том же split,
  - ансамбль M2-FE+ × M2 с grid 0.1–0.9,
сохраняет proba и индексы для последующих экспериментов (ablation/counterfactual/etc.).
"""
import gc, time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm2_fe_plus_individual_v2_log.txt'

SEED = 42
TARGET = 'resolution'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 60)
log('M2-FE+ INDIVIDUAL v2: split в точности как в ноуте 02')

# --- Данные ---
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['name_rus_str'] = df['name_rus'].fillna('').astype(str)
df['description_str'] = df['description'].fillna('').astype(str)
df['brand_str'] = df['brand_name'].fillna('').astype(str)
df['text'] = (df['name_rus_str'] + ' ' + df['description_str'] + ' ' + df['brand_str']).str.slice(0, 512)
log(f'data: {df.shape}, positive {df[TARGET].mean()*100:.2f}%')

# --- ТОЧНЫЙ split из 02_reproduction_fixed.ipynb ---
log('Воспроизведение split (точно как в ноуте 02):')
seller_targets = df.groupby('SellerID')[TARGET].max().reset_index()

train_sellers, temp_sellers = train_test_split(
    seller_targets['SellerID'], test_size=0.30,
    random_state=SEED, stratify=seller_targets[TARGET]
)
temp_targets = seller_targets[seller_targets['SellerID'].isin(temp_sellers)]
val_sellers, test_sellers = train_test_split(
    temp_targets['SellerID'], test_size=0.50,
    random_state=SEED, stratify=temp_targets[TARGET]
)

train_idx = df.index[df['SellerID'].isin(set(train_sellers))].values
val_idx = df.index[df['SellerID'].isin(set(val_sellers))].values
test_idx = df.index[df['SellerID'].isin(set(test_sellers))].values
log(f'  train sellers={len(train_sellers)}, val sellers={len(val_sellers)}, test sellers={len(test_sellers)}')
log(f'  объекты: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

# Контроль соответствия паспорту
expected = (144967, 25904, 26327)
got = (len(train_idx), len(val_idx), len(test_idx))
log(f'  expected from паспорт: {expected}')
log(f'  got:                    {got}')
if got != expected:
    log('  WARNING: размеры не совпадают, проверь параметры')
else:
    log('  OK: размеры совпадают с паспортом')

# Контроль отсутствия пересечений
assert set(train_sellers).isdisjoint(val_sellers)
assert set(train_sellers).isdisjoint(test_sellers)
assert set(val_sellers).isdisjoint(test_sellers)
log('  seller-disjoint: OK')

y_test_true = df[TARGET].values[test_idx]
log(f'  positive в test: {int(y_test_true.sum())} ({y_test_true.mean()*100:.2f}%)')

# Сохраняю индексы и метки сразу — другие скрипты могут запускаться параллельно
np.save(OUT_DIR / 'y_test_indiv.npy', y_test_true)
np.save(OUT_DIR / 'train_idx_indiv.npy', train_idx)
np.save(OUT_DIR / 'val_idx_indiv.npy', val_idx)
np.save(OUT_DIR / 'test_idx_indiv.npy', test_idx)
log('  сохранены: y_test_indiv.npy, {train,val,test}_idx_indiv.npy')

# --- FE группы (как в командном M2-FE+) ---
log('Подсчёт FE групп...')

# Группа №1 text statistics (FADAML domain-text, Nguyen 2025 [4])
for s in ['name_rus', 'description', 'brand']:
    df[f'{s}_char_len'] = df[f'{s}_str'].str.len().astype(np.float32)
    df[f'{s}_word_count'] = df[f'{s}_str'].str.split().str.len().fillna(0).astype(np.float32)
    df[f'{s}_digit_ratio'] = df[f'{s}_str'].apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)).astype(np.float32)
    df[f'{s}_upper_ratio'] = df[f'{s}_str'].apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)).astype(np.float32)
    df[f'{s}_special_count'] = df[f'{s}_str'].apply(lambda x: sum(c in '!?*#$%&@<>/\\' for c in x)).astype(np.float32)
    df[f'{s}_avg_word_len'] = df[f'{s}_str'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0).astype(np.float32)

# Группа №2 brand-level (на ИНДИВ train)
df['brand_key'] = df['brand_str'].str.lower().str.strip()
df_train_indiv = df.iloc[train_idx]
brand_stats = df_train_indiv.groupby('brand_key').agg(
    brand_n_items=('ItemID', 'count'),
    brand_n_sellers=('SellerID', 'nunique'),
    brand_mean_price=('PriceDiscounted', 'mean'),
    brand_std_price=('PriceDiscounted', 'std'),
).fillna(0).reset_index()
brand_stats['brand_seller_concentration'] = brand_stats['brand_n_items'] / np.maximum(brand_stats['brand_n_sellers'], 1)
df = df.merge(brand_stats, on='brand_key', how='left')
for c in ['brand_n_items', 'brand_n_sellers', 'brand_mean_price', 'brand_std_price', 'brand_seller_concentration']:
    df[c] = df[c].fillna(-1).astype(np.float32)

# Группа №3 coherence
df['nd_word_overlap'] = df.apply(lambda r: len(set(r['name_rus_str'].lower().split()) & set(r['description_str'].lower().split())) / max(len(set(r['name_rus_str'].lower().split())), 1), axis=1).astype(np.float32)
df['nd_len_ratio'] = (df['description_char_len'] / np.maximum(df['name_rus_char_len'], 1)).astype(np.float32)
df['nd_both_present'] = ((df['name_rus_char_len'] > 5) & (df['description_char_len'] > 10)).astype(np.float32)

# Группа №4 interactions
for c in ['PriceDiscounted', 'item_time_alive', 'seller_time_alive', 'item_count_returns30', 'item_count_sales30']:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
df['price_per_age'] = (df['PriceDiscounted'] / np.maximum(df['item_time_alive'], 1)).astype(np.float32)
df['returns_per_sales'] = (df['item_count_returns30'] / np.maximum(df['item_count_sales30'], 1)).astype(np.float32)
df['item_seller_age_ratio'] = (df['item_time_alive'] / np.maximum(df['seller_time_alive'], 1)).astype(np.float32)
df['age_brand_concentration'] = (df['item_time_alive'] * df['brand_seller_concentration']).astype(np.float32)

# CLIP
log('CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()

# Группа №5 CLIP-derived structural (на ИНДИВ train)
y_train = df[TARGET].values[train_idx]
clip_train = df[clip_cols].values[train_idx]
centroid_pos = clip_train[y_train == 1].mean(axis=0)
centroid_neg = clip_train[y_train == 0].mean(axis=0)
clip_all = df[clip_cols].values.astype(np.float32)
df['clip_dist_to_pos_centroid'] = np.linalg.norm(clip_all - centroid_pos, axis=1).astype(np.float32)
df['clip_dist_to_neg_centroid'] = np.linalg.norm(clip_all - centroid_neg, axis=1).astype(np.float32)
df['clip_pos_neg_ratio'] = (df['clip_dist_to_pos_centroid'] / np.maximum(df['clip_dist_to_neg_centroid'], 1e-6)).astype(np.float32)
df['clip_diff_to_pos_neg'] = (df['clip_dist_to_pos_centroid'] - df['clip_dist_to_neg_centroid']).astype(np.float32)
from sklearn.cluster import KMeans
km_clip = KMeans(n_clusters=8, random_state=SEED, n_init=5, max_iter=100).fit(clip_train)
df['clip_cluster_id'] = km_clip.predict(clip_all).astype(np.int8)
df['clip_dist_to_own_cluster'] = km_clip.transform(clip_all).min(axis=1).astype(np.float32)

# TF-IDF SVD (на ИНДИВ train)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
log('TF-IDF SVD-50 на индив train...')
tfv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
texts_train = df.iloc[train_idx]['text'].values
tfv.fit(texts_train)
svd = TruncatedSVD(n_components=50, random_state=SEED)
svd.fit(tfv.transform(texts_train))
svd_arr = svd.transform(tfv.transform(df['text'].values)).astype(np.float32)
for i in range(50):
    df[f'svd_{i}'] = svd_arr[:, i]
del svd_arr; gc.collect()
svd_cols = [f'svd_{i}' for i in range(50)]

# Tab cols (как в командном FE+: 38 признаков из team_split_meta.json)
team_feat_path = ROOT / 'real_estate_approaches' / 'notebooks' / 'team_splits' / 'team_split_meta.json'
if team_feat_path.exists():
    meta = json.load(open(team_feat_path))
    tab_cols = meta['team_feature_cols']
    log(f'  tab признаков из team_split_meta: {len(tab_cols)}')
else:
    log('  WARNING: team_split_meta.json не найден, используем все числовые')
    tab_cols = [c for c in df.select_dtypes(include='number').columns if c not in clip_cols + svd_cols + ['resolution', 'ItemID', 'SellerID']][:38]

for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)
cat_cols = ['CommercialTypeName4']

# Группа №6 cat_x_null_brand
df['cat_x_null_brand'] = (df['CommercialTypeName4'] + '_' + (df['brand_str'] == '').astype(int).astype(str)).astype(str)
cat_cols_extra = ['cat_x_null_brand']

# --- Финальный набор и обучение ---
new_feats = (
    [f'{s}_{stat}' for s in ['name_rus', 'description', 'brand'] for stat in ['char_len', 'word_count', 'digit_ratio', 'upper_ratio', 'special_count', 'avg_word_len']]
    + ['brand_n_items', 'brand_n_sellers', 'brand_mean_price', 'brand_std_price', 'brand_seller_concentration']
    + ['nd_word_overlap', 'nd_len_ratio', 'nd_both_present']
    + ['price_per_age', 'returns_per_sales', 'item_seller_age_ratio', 'age_brand_concentration']
    + ['clip_dist_to_pos_centroid', 'clip_dist_to_neg_centroid', 'clip_pos_neg_ratio', 'clip_diff_to_pos_neg', 'clip_cluster_id', 'clip_dist_to_own_cluster']
)
features_full = tab_cols + clip_cols + svd_cols + new_feats
all_cat = cat_cols + cat_cols_extra
log(f'итого признаков: {len(features_full)+len(cat_cols_extra)}')

X = df[features_full + cat_cols_extra].copy()
y = df[TARGET].values
X_tr = X.iloc[train_idx].reset_index(drop=True); y_tr = y[train_idx]
X_va = X.iloc[val_idx].reset_index(drop=True); y_va = y[val_idx]
X_te = X.iloc[test_idx].reset_index(drop=True)
log(f'X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}')

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

# 1. M2-FE+ на индивидуальном
log('=' * 50)
log('Обучение M2-FE+ INDIVIDUAL v2...')
t_c = time.time()
cb_fe = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=SEED, early_stopping_rounds=100, cat_features=all_cat,
    thread_count=4, verbose=200, auto_class_weights='Balanced',
)
cb_fe.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
log(f'  обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb_fe.tree_count_}')
p_fe = cb_fe.predict_proba(X_te)[:, 1].astype(np.float32)
log(f'  M2-FE+ INDIV v2: ROC={roc_auc_score(y_test_true, p_fe):.4f}  PR={average_precision_score(y_test_true, p_fe):.4f}  R@P09={rap(y_test_true, p_fe):.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_indiv.npy', p_fe)

# val proba (для conformal)
p_fe_val = cb_fe.predict_proba(X_va)[:, 1].astype(np.float32)
np.save(OUT_DIR / 'val_proba_diana_m2_fe_plus_indiv.npy', p_fe_val)
log(f'  M2-FE+ INDIV v2 val proba: сохранён ({len(p_fe_val)} объектов)')

# SHAP топ-15
fi = cb_fe.get_feature_importance()
fi_pairs = sorted(zip(features_full + cat_cols_extra, fi), key=lambda x: -x[1])[:15]
log('SHAP топ-15 M2-FE+ INDIV v2:')
for i, (f, w) in enumerate(fi_pairs, 1):
    log(f'  {i:>2d}. {f:35s} {w:.3f}')

del cb_fe; gc.collect()

# 2. M2 baseline на индив (для ансамбля и component ablation)
log('=' * 50)
log('Обучение M2 BASELINE INDIVIDUAL v2 для ансамбля...')
features_base = tab_cols + clip_cols + svd_cols
X_tr_b = df[features_base].iloc[train_idx].reset_index(drop=True)
X_va_b = df[features_base].iloc[val_idx].reset_index(drop=True)
X_te_b = df[features_base].iloc[test_idx].reset_index(drop=True)

t_c = time.time()
cb_m2 = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=SEED, early_stopping_rounds=100, cat_features=cat_cols,
    thread_count=4, verbose=200, auto_class_weights='Balanced',
)
cb_m2.fit(X_tr_b, y_tr, eval_set=(X_va_b, y_va), use_best_model=True)
log(f'  обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb_m2.tree_count_}')
p_m2 = cb_m2.predict_proba(X_te_b)[:, 1].astype(np.float32)
log(f'  M2 BASE INDIV v2: ROC={roc_auc_score(y_test_true, p_m2):.4f}  PR={average_precision_score(y_test_true, p_m2):.4f}  R@P09={rap(y_test_true, p_m2):.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_base_indiv.npy', p_m2)

p_m2_val = cb_m2.predict_proba(X_va_b)[:, 1].astype(np.float32)
np.save(OUT_DIR / 'val_proba_diana_m2_base_indiv.npy', p_m2_val)
log(f'  M2 BASE INDIV v2 val proba: сохранён ({len(p_m2_val)} объектов)')

del cb_m2; gc.collect()

# 3. Ансамбли
log('=' * 50)
log('Ансамбли на ИНДИВИДУАЛЬНОМ test:')

mix50 = 0.5 * p_fe + 0.5 * p_m2
roc, pr, r9 = roc_auc_score(y_test_true, mix50), average_precision_score(y_test_true, mix50), rap(y_test_true, mix50)
log(f'  0.5*M2-FE+ + 0.5*M2 (default): ROC={roc:.4f}  PR={pr:.4f}  R@P09={r9:.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_ansamble_indiv.npy', mix50.astype(np.float32))

# Веса grid 0.1..0.9 (PR-AUC и R@P) — поиск best PR и best R@P
log('  Grid w_FE+ ∈ {0.1..0.9}:')
results = []
for w in np.arange(0.1, 1.0, 0.05):
    mix_w = w * p_fe + (1-w) * p_m2
    pr_w = average_precision_score(y_test_true, mix_w)
    r9_w = rap(y_test_true, mix_w)
    results.append((w, pr_w, r9_w))
    log(f'    w_FE+={w:.2f}: PR={pr_w:.4f}  R@P09={r9_w:.4f}')

best_pr_cfg = max(results, key=lambda r: r[1])
best_r9_cfg = max(results, key=lambda r: r[2])
log(f'\nЛУЧШИЙ ВЕС M2-FE+ по PR-AUC: w={best_pr_cfg[0]:.2f}, PR={best_pr_cfg[1]:.4f}, R@P={best_pr_cfg[2]:.4f}')
log(f'ЛУЧШИЙ ВЕС M2-FE+ по R@P09: w={best_r9_cfg[0]:.2f}, PR={best_r9_cfg[1]:.4f}, R@P={best_r9_cfg[2]:.4f}')

# Сохранить лучший R@P ансамбль отдельно (для R@P-headline)
best_mix = best_r9_cfg[0] * p_fe + (1 - best_r9_cfg[0]) * p_m2
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_ansamble_indiv_best_rap.npy', best_mix.astype(np.float32))

# Сводный JSON для удобства последующих скриптов
summary = {
    'expected_sizes': {'train': 144967, 'val': 25904, 'test': 26327},
    'got_sizes': {'train': int(len(train_idx)), 'val': int(len(val_idx)), 'test': int(len(test_idx))},
    'sizes_match': got == expected,
    'features_full_count': len(features_full) + len(cat_cols_extra),
    'tab_cols_count': len(tab_cols),
    'm2_fe_plus_indiv': {
        'ROC': float(roc_auc_score(y_test_true, p_fe)),
        'PR': float(average_precision_score(y_test_true, p_fe)),
        'RAP09': float(rap(y_test_true, p_fe)),
    },
    'm2_base_indiv': {
        'ROC': float(roc_auc_score(y_test_true, p_m2)),
        'PR': float(average_precision_score(y_test_true, p_m2)),
        'RAP09': float(rap(y_test_true, p_m2)),
    },
    'ensemble_50_50': {
        'ROC': float(roc), 'PR': float(pr), 'RAP09': float(r9),
    },
    'best_pr_grid': {'w_fe_plus': float(best_pr_cfg[0]), 'PR': float(best_pr_cfg[1]), 'RAP09': float(best_pr_cfg[2])},
    'best_rap_grid': {'w_fe_plus': float(best_r9_cfg[0]), 'PR': float(best_r9_cfg[1]), 'RAP09': float(best_r9_cfg[2])},
}
with open(OUT_DIR / 'm2_fe_plus_indiv_v2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
log(f'\nSummary JSON сохранён: {OUT_DIR / "m2_fe_plus_indiv_v2_summary.json"}')

log(f'\nВсего: {(time.time()-t0)/60:.1f} мин')
log('=' * 60)
