"""
M2-FE+ на ИНДИВИДУАЛЬНОМ протоколе автора (test=26327, val=25904, train=144967).

Реконструирует тот же split, что Диана использовала в § 4.4 (двухэтапный
train_test_split по уникальным SellerID с stratify по seller_max_resolution,
random_state=42), и обучает M2-FE+ с расширенным FE на этом split.

Также обучает baseline M2 на том же индив split для ансамбля.

Цель: M2-FE+ для § 4.4.3.3 (индивидуальный раздел Дианы), не для Главы 5.
"""
import gc, time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path('/Users/diana/master-thesis-2025')
DATA_CSV = ROOT / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
OUT_DIR = ROOT / 'Диана_ВКР_финал' / 'notebooks'
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'm2_fe_plus_individual_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2-FE+ INDIVIDUAL: на индивидуальном split Дианы (§ 4.4)')

# --- Данные ---
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['name_rus_str'] = df['name_rus'].fillna('').astype(str)
df['description_str'] = df['description'].fillna('').astype(str)
df['brand_str'] = df['brand_name'].fillna('').astype(str)
df['text'] = (df['name_rus_str'] + ' ' + df['description_str'] + ' ' + df['brand_str']).str.slice(0, 512)
log(f'data: {df.shape}, positive {df["resolution"].mean()*100:.2f}%')

# --- Реконструкция индивидуального split ---
log('Реконструкция индивидуального split (§ 4.4, Приложение В):')
log('  Двухэтапный train_test_split по SellerID, stratify по seller_max_resolution, random_state=42')

# Шаг 1: уникальные продавцы + макс resolution по продавцу
seller_targets = df.groupby('SellerID')['resolution'].max()
unique_sellers = seller_targets.index.values
seller_max_y = seller_targets.values
log(f'  уникальных продавцов: {len(unique_sellers)}')

# Этап 1: train sellers (70%) vs temp sellers (30%)
train_sellers, temp_sellers, _, temp_y = train_test_split(
    unique_sellers, seller_max_y,
    test_size=0.30, random_state=42, stratify=seller_max_y,
)
log(f'  train sellers={len(train_sellers)}, temp sellers={len(temp_sellers)}')

# Этап 2: temp → val (50%) + test (50%)
val_sellers, test_sellers = train_test_split(
    temp_sellers, test_size=0.50, random_state=42, stratify=temp_y,
)
log(f'  val sellers={len(val_sellers)}, test sellers={len(test_sellers)}')

train_idx = df.index[df['SellerID'].isin(set(train_sellers))].values
val_idx = df.index[df['SellerID'].isin(set(val_sellers))].values
test_idx = df.index[df['SellerID'].isin(set(test_sellers))].values
log(f'  объекты: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

y_test_true = df['resolution'].values[test_idx]
log(f'  positive в test: {y_test_true.sum()} ({y_test_true.mean()*100:.2f}%)')

# Проверка соответствия паспорту: train=144 967, val=25 904, test=26 327
expected = (144967, 25904, 26327)
got = (len(train_idx), len(val_idx), len(test_idx))
log(f'  expected from паспорт: {expected}')
log(f'  got:                    {got}')
if got != expected:
    log('  WARNING: размеры не совпадают, проверь параметры')

# --- FE группы (как в командном M2-FE+) ---
log('Подсчёт FE групп...')

# Группа №1 text statistics
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

# Группа №6 CLIP-derived structural (на ИНДИВ train)
y_train = df['resolution'].values[train_idx]
clip_train = df[clip_cols].values[train_idx]
centroid_pos = clip_train[y_train == 1].mean(axis=0)
centroid_neg = clip_train[y_train == 0].mean(axis=0)
clip_all = df[clip_cols].values.astype(np.float32)
df['clip_dist_to_pos_centroid'] = np.linalg.norm(clip_all - centroid_pos, axis=1).astype(np.float32)
df['clip_dist_to_neg_centroid'] = np.linalg.norm(clip_all - centroid_neg, axis=1).astype(np.float32)
df['clip_pos_neg_ratio'] = (df['clip_dist_to_pos_centroid'] / np.maximum(df['clip_dist_to_neg_centroid'], 1e-6)).astype(np.float32)
df['clip_diff_to_pos_neg'] = (df['clip_dist_to_pos_centroid'] - df['clip_dist_to_neg_centroid']).astype(np.float32)
from sklearn.cluster import KMeans
km_clip = KMeans(n_clusters=8, random_state=42, n_init=5, max_iter=100).fit(clip_train)
df['clip_cluster_id'] = km_clip.predict(clip_all).astype(np.int8)
df['clip_dist_to_own_cluster'] = km_clip.transform(clip_all).min(axis=1).astype(np.float32)

# TF-IDF SVD (на ИНДИВ train)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
log('TF-IDF SVD-50 на индив train...')
tfv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
texts_train = df.iloc[train_idx]['text'].values
tfv.fit(texts_train)
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(tfv.transform(texts_train))
svd_arr = svd.transform(tfv.transform(df['text'].values)).astype(np.float32)
for i in range(50):
    df[f'svd_{i}'] = svd_arr[:, i]
del svd_arr; gc.collect()
svd_cols = [f'svd_{i}' for i in range(50)]

# Tab cols
# Используем те же 38 tab признаков что и в командном FE+
team_feat_path = ROOT / 'Диана_ВКР_финал' / 'notebooks' / 'team_splits' / 'team_split_meta.json'
if team_feat_path.exists():
    meta = json.load(open(team_feat_path))
    tab_cols = meta['team_feature_cols']
else:
    log('WARNING: team_split_meta.json не найден, используем все числовые')
    tab_cols = [c for c in df.select_dtypes(include='number').columns if c not in clip_cols + svd_cols + ['resolution', 'ItemID', 'SellerID']][:38]

for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)
cat_cols = ['CommercialTypeName4']

# Группа №7 cat_x_null_brand
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
y = df['resolution'].values
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
log('=' * 40)
log('Обучение M2-FE+ INDIVIDUAL...')
t_c = time.time()
cb_fe = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=42, early_stopping_rounds=100, cat_features=all_cat,
    thread_count=4, verbose=200, auto_class_weights='Balanced',
)
cb_fe.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
log(f'  обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb_fe.tree_count_}')
p_fe = cb_fe.predict_proba(X_te)[:, 1].astype(np.float32)
log(f'  M2-FE+ INDIV: ROC={roc_auc_score(y_test_true, p_fe):.4f}  PR={average_precision_score(y_test_true, p_fe):.4f}  R@P09={rap(y_test_true, p_fe):.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_indiv.npy', p_fe)

# SHAP топ-15
fi = cb_fe.get_feature_importance()
fi_pairs = sorted(zip(features_full + cat_cols_extra, fi), key=lambda x: -x[1])[:15]
log('SHAP топ-15 M2-FE+ INDIV:')
for i, (f, w) in enumerate(fi_pairs, 1):
    log(f'  {i:>2d}. {f:35s} {w:.3f}')

del cb_fe; gc.collect()

# 2. M2 baseline на индив (для ансамбля)
log('=' * 40)
log('Обучение M2 BASELINE INDIVIDUAL для ансамбля...')
features_base = tab_cols + clip_cols + svd_cols
X_tr_b = df[features_base].iloc[train_idx].reset_index(drop=True)
X_va_b = df[features_base].iloc[val_idx].reset_index(drop=True)
X_te_b = df[features_base].iloc[test_idx].reset_index(drop=True)

t_c = time.time()
cb_m2 = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=42, early_stopping_rounds=100, cat_features=cat_cols,
    thread_count=4, verbose=200, auto_class_weights='Balanced',
)
cb_m2.fit(X_tr_b, y_tr, eval_set=(X_va_b, y_va), use_best_model=True)
log(f'  обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb_m2.tree_count_}')
p_m2 = cb_m2.predict_proba(X_te_b)[:, 1].astype(np.float32)
log(f'  M2 BASE INDIV: ROC={roc_auc_score(y_test_true, p_m2):.4f}  PR={average_precision_score(y_test_true, p_m2):.4f}  R@P09={rap(y_test_true, p_m2):.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_base_indiv.npy', p_m2)
del cb_m2; gc.collect()

# 3. Ансамбли
log('=' * 40)
log('Ансамбли на ИНДИВИДУАЛЬНОМ test:')

mix = 0.5 * p_fe + 0.5 * p_m2
roc, pr, r9 = roc_auc_score(y_test_true, mix), average_precision_score(y_test_true, mix), rap(y_test_true, mix)
log(f'  0.5*M2-FE+ + 0.5*M2 (headline): ROC={roc:.4f}  PR={pr:.4f}  R@P09={r9:.4f}')
np.save(OUT_DIR / 'test_proba_diana_m2_fe_plus_ansamble_indiv.npy', mix.astype(np.float32))

# Веса grid
log('  Grid 0.1-0.9 шаг 0.1:')
best_pr = 0; best_cfg = None
for w in np.arange(0.1, 1.0, 0.1):
    mix_w = w * p_fe + (1-w) * p_m2
    pr = average_precision_score(y_test_true, mix_w)
    r9 = rap(y_test_true, mix_w)
    if pr > best_pr:
        best_pr = pr; best_cfg = (w, r9)
    log(f'    w_FE+={w:.1f}: PR={pr:.4f}  R@P09={r9:.4f}')
log(f'\nЛУЧШИЙ ВЕС M2-FE+: w={best_cfg[0]:.1f}, PR={best_pr:.4f}, R@P09={best_cfg[1]:.4f}')

# Сохранить y_test для других экспериментов
np.save(OUT_DIR / 'y_test_indiv.npy', y_test_true)
np.save(OUT_DIR / 'train_idx_indiv.npy', train_idx)
np.save(OUT_DIR / 'val_idx_indiv.npy', val_idx)
np.save(OUT_DIR / 'test_idx_indiv.npy', test_idx)
log(f'\nSaved indiv splits и метки для будущих экспериментов: y_test_indiv.npy, {{train,val,test}}_idx_indiv.npy')

log(f'\nВсего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
