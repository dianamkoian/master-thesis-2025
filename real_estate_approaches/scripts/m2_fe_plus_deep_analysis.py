"""
Глубокий анализ M2-FE+:
1. Component ablation по 6 группам FE (какая группа главная)
2. Counterfactual / semantic ablation на новых FE-фичах
"""
import gc, time, os, json, warnings
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
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'm2_fe_plus_deep_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2-FE+ глубокий анализ: component ablation + counterfactual')

# --- Данные + базовая обработка (как в m2_fe_plus.py) ---
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['name_rus_str'] = df['name_rus'].fillna('').astype(str)
df['description_str'] = df['description'].fillna('').astype(str)
df['brand_str'] = df['brand_name'].fillna('').astype(str)
df['text'] = (df['name_rus_str'] + ' ' + df['description_str'] + ' ' + df['brand_str']).str.slice(0, 512)

train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')

# Группа №1 text statistics (FADAML domain-text, Nguyen 2025 [4])
log('Группа №1: text statistics (FADAML [4])')
for s in ['name_rus', 'description', 'brand']:
    df[f'{s}_char_len'] = df[f'{s}_str'].str.len().astype(np.float32)
    df[f'{s}_word_count'] = df[f'{s}_str'].str.split().str.len().fillna(0).astype(np.float32)
    df[f'{s}_digit_ratio'] = df[f'{s}_str'].apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)).astype(np.float32)
    df[f'{s}_upper_ratio'] = df[f'{s}_str'].apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)).astype(np.float32)
    df[f'{s}_special_count'] = df[f'{s}_str'].apply(lambda x: sum(c in '!?*#$%&@<>/\\' for c in x)).astype(np.float32)
    df[f'{s}_avg_word_len'] = df[f'{s}_str'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0).astype(np.float32)

# Группа №2 brand-level
log('Группа №2: brand aggregates')
df['brand_key'] = df['brand_str'].str.lower().str.strip()
df_train = df.iloc[train_idx]
brand_stats = df_train.groupby('brand_key').agg(
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
log('Группа №3: coherence')
df['nd_word_overlap'] = df.apply(lambda r: len(set(r['name_rus_str'].lower().split()) & set(r['description_str'].lower().split())) / max(len(set(r['name_rus_str'].lower().split())), 1), axis=1).astype(np.float32)
df['nd_len_ratio'] = (df['description_char_len'] / np.maximum(df['name_rus_char_len'], 1)).astype(np.float32)
df['nd_both_present'] = ((df['name_rus_char_len'] > 5) & (df['description_char_len'] > 10)).astype(np.float32)

# Группа №4 interactions
log('Группа №4: interactions')
for c in ['PriceDiscounted', 'item_time_alive', 'seller_time_alive', 'item_count_returns30', 'item_count_sales30']:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median()).astype(np.float32)
df['price_per_age'] = (df['PriceDiscounted'] / np.maximum(df['item_time_alive'], 1)).astype(np.float32)
df['returns_per_sales'] = (df['item_count_returns30'] / np.maximum(df['item_count_sales30'], 1)).astype(np.float32)
df['item_seller_age_ratio'] = (df['item_time_alive'] / np.maximum(df['seller_time_alive'], 1)).astype(np.float32)
df['age_brand_concentration'] = (df['item_time_alive'] * df['brand_seller_concentration']).astype(np.float32)

# CLIP
log('Загрузка CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()

# Группа №6 CLIP-derived structural
log('Группа №6: CLIP-derived structural')
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

# TF-IDF SVD
log('TF-IDF SVD-50...')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)
cat_cols = ['CommercialTypeName4']

# Группа №7 categorical interaction
df['cat_x_null_brand'] = (df['CommercialTypeName4'] + '_' + (df['brand_str'] == '').astype(int).astype(str)).astype(str)
cat_cols_extra = ['cat_x_null_brand']

# Группы признаков
groups = {
    'text_stats': [c for c in df.columns if any(s in c for s in ['_char_len', '_word_count', '_digit_ratio', '_upper_ratio', '_special_count', '_avg_word_len'])],
    'brand_agg': ['brand_n_items', 'brand_n_sellers', 'brand_mean_price', 'brand_std_price', 'brand_seller_concentration'],
    'coherence': ['nd_word_overlap', 'nd_len_ratio', 'nd_both_present'],
    'interactions': ['price_per_age', 'returns_per_sales', 'item_seller_age_ratio', 'age_brand_concentration'],
    'clip_structural': ['clip_dist_to_pos_centroid', 'clip_dist_to_neg_centroid', 'clip_pos_neg_ratio', 'clip_diff_to_pos_neg', 'clip_cluster_id', 'clip_dist_to_own_cluster'],
    'cat_interactions': cat_cols_extra,
}
all_new = sum(groups.values(), [])
base_feats = tab_cols + clip_cols + svd_cols

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

# === 1. M2-FE+ full retrain (для counterfactual анализа) ===
log('=' * 50)
log('M2-FE+ FULL retrain для counterfactual...')
features_full = base_feats + all_new
all_cat = cat_cols + cat_cols_extra
X = df[features_full].copy()
y = df['resolution'].values
X_tr = X.iloc[train_idx].reset_index(drop=True); y_tr = y[train_idx]
X_va = X.iloc[val_idx].reset_index(drop=True); y_va = y[val_idx]
X_te = X.iloc[test_idx].reset_index(drop=True)

t_c = time.time()
cb_full = CatBoostClassifier(
    iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=42, early_stopping_rounds=100, cat_features=all_cat,
    thread_count=4, verbose=0, auto_class_weights='Balanced',
)
cb_full.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
p_te = cb_full.predict_proba(X_te)[:, 1].astype(np.float32)
log(f'  M2-FE+ обучен за {(time.time()-t_c)/60:.1f} мин, best iter {cb_full.tree_count_}')
log(f'  M2-FE+ test: ROC={roc_auc_score(y_test_true, p_te):.4f}  PR={average_precision_score(y_test_true, p_te):.4f}  R@P09={rap(y_test_true, p_te):.4f}')

# === 2. Counterfactual для M2-FE+ ===
log('=' * 50)
log('=== COUNTERFACTUAL для M2-FE+ ===')
high_conf_mask = (y_test_true == 1) & (p_te > 0.7)
log(f'high-confidence TP подвыборка: {high_conf_mask.sum()}')
X_high = X_te.loc[high_conf_mask].copy().reset_index(drop=True)
p_high_base = p_te[high_conf_mask]
log(f'baseline mean p = {p_high_base.mean():.4f}')

# Median values для negative класса в train
neg_mask = (y_tr == 0)
median_neg = {}
for c in features_full:
    if df[c].dtype.kind in 'fi':
        try:
            median_neg[c] = float(np.median(X_tr.loc[neg_mask, c]))
        except Exception:
            pass

# Counterfactual per group
log(f"\n{'Группа':30s} {'размер':>10s} {'mean ΔP':>10s} {'%resolved':>10s}")
for grp_name, feats in groups.items():
    X_mut = X_high.copy()
    for f in feats:
        if f in median_neg:
            X_mut[f] = median_neg[f]
        elif f in cat_cols_extra:
            # Replace categorical with most common value among negatives
            mode_val = X_tr.loc[neg_mask, f].mode().iloc[0] if not X_tr.loc[neg_mask, f].mode().empty else 'NA'
            X_mut[f] = mode_val
    p_mut = cb_full.predict_proba(X_mut)[:, 1]
    delta_p = p_high_base - p_mut
    log(f'{grp_name:30s} {len(feats):>10d} {delta_p.mean():>10.4f} {(p_mut<0.5).mean()*100:>9.1f}%')

# Top counterfactual features (single)
log('\nTop-10 single-feature counterfactual ΔP (только новые FE):')
single_results = []
for f in all_new:
    if f in median_neg:
        X_mut = X_high.copy()
        X_mut[f] = median_neg[f]
        p_mut = cb_full.predict_proba(X_mut)[:, 1]
        single_results.append((f, (p_high_base - p_mut).mean()))
single_results.sort(key=lambda x: -x[1])
for f, dp in single_results[:10]:
    log(f'  {f:35s} ΔP={dp:.4f}')

# === 3. Component ablation ===
log('=' * 50)
log('=== COMPONENT ABLATION (исключение одной FE-группы из M2-FE+ full) ===')
ablation_results = {}
log(f"{'config':25s} {'ROC':>8s} {'PR-AUC':>8s} {'R@P09':>8s} {'ΔPR vs FE+':>12s} {'time':>8s}")
# baseline FE+ full (уже посчитан выше)
pr_full = average_precision_score(y_test_true, p_te)
ablation_results['M2-FE+ full'] = (roc_auc_score(y_test_true, p_te), pr_full, rap(y_test_true, p_te))
log(f'{"M2-FE+ full":25s} {ablation_results["M2-FE+ full"][0]:8.4f} {ablation_results["M2-FE+ full"][1]:8.4f} {ablation_results["M2-FE+ full"][2]:8.4f}    0.0000')

del cb_full; gc.collect()

for grp_name, feats_to_remove in groups.items():
    feats_ablation = [f for f in features_full if f not in feats_to_remove]
    cat_ablation = [c for c in all_cat if c not in feats_to_remove]
    t_c = time.time()
    X_tr_a = df[feats_ablation].iloc[train_idx].reset_index(drop=True)
    X_va_a = df[feats_ablation].iloc[val_idx].reset_index(drop=True)
    X_te_a = df[feats_ablation].iloc[test_idx].reset_index(drop=True)
    cb_a = CatBoostClassifier(
        iterations=2500, depth=8, learning_rate=0.05, eval_metric='AUC',
        random_seed=42, early_stopping_rounds=100, cat_features=cat_ablation,
        thread_count=4, verbose=0, auto_class_weights='Balanced',
    )
    cb_a.fit(X_tr_a, y_tr, eval_set=(X_va_a, y_va), use_best_model=True)
    p_a = cb_a.predict_proba(X_te_a)[:, 1]
    roc, pr, r9 = roc_auc_score(y_test_true, p_a), average_precision_score(y_test_true, p_a), rap(y_test_true, p_a)
    ablation_results[f'M2-FE+ без {grp_name}'] = (roc, pr, r9)
    log(f'{"M2-FE+ без "+grp_name:25s} {roc:8.4f} {pr:8.4f} {r9:8.4f}    {pr-pr_full:+8.4f}    {(time.time()-t_c)/60:.1f}')
    del cb_a; gc.collect()

# Сводка
log('=' * 50)
log('=== ИТОГОВАЯ СВОДКА: какая группа FE наиболее важна ===')
log('(чем больше |ΔPR| при исключении, тем важнее группа)')
for grp_name in groups:
    if f'M2-FE+ без {grp_name}' in ablation_results:
        roc, pr, r9 = ablation_results[f'M2-FE+ без {grp_name}']
        delta = pr - pr_full
        log(f'  {grp_name:25s}: ΔPR при исключении = {delta:+.4f}')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
