"""
Counterfactual / semantic ablation на уровне отдельных признаков.
Для модели M2 на team split: берём правильно классифицированные
контрафактные объекты с высокой уверенностью и поочерёдно мутируем
один ключевой признак на «нормальное» значение (медиана у класса 0).
Замеряем ΔP — насколько падает предсказание модели.
Это даёт количественную семантическую важность каждого признака.
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
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'counterfactual_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('Counterfactual / semantic ablation')

# 1. Данные
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
df['text'] = df['text'].str.slice(0, 512)
train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')

# 2. CLIP
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()
log(f'CLIP merged: {len(clip_cols)} cols')

# 3. TF-IDF SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
log('TF-IDF + SVD-50...')
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

# 4. Tab features + impute
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)
cat_cols = ['CommercialTypeName4']

# 5. Train M2 (упрощённый, без K-means для краткости)
features = tab_cols + clip_cols + svd_cols
log(f'Признаков: {len(features)}')

X_tr = df[features].iloc[train_idx].reset_index(drop=True)
y_tr = df['resolution'].values[train_idx]
X_va = df[features].iloc[val_idx].reset_index(drop=True)
y_va = df['resolution'].values[val_idx]
X_te = df[features].iloc[test_idx].reset_index(drop=True)

from catboost import CatBoostClassifier
log('Обучение M2 (упрощённой)...')
cb = CatBoostClassifier(
    iterations=1500, depth=8, learning_rate=0.05, eval_metric='AUC',
    random_seed=42, early_stopping_rounds=100, cat_features=cat_cols,
    thread_count=4, verbose=0, auto_class_weights='Balanced',
)
t_cb = time.time()
cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
log(f'  обучена за {(time.time()-t_cb)/60:.1f} мин, best iter={cb.tree_count_}')

p_test_base = cb.predict_proba(X_te)[:, 1]
from sklearn.metrics import roc_auc_score, average_precision_score
log(f'  baseline test: ROC={roc_auc_score(y_test_true, p_test_base):.4f}  PR={average_precision_score(y_test_true, p_test_base):.4f}')

# 6. Counterfactual: для high-conf TP, мутируем по одному признаку
#    «Нормальное» значение = медиана этого признака у negatives в train
log('=' * 50)
log('Counterfactual ablation на high-confidence TP (y=1, p>0.7):')

high_conf_mask = (y_test_true == 1) & (p_test_base > 0.7)
log(f'  размер high-conf TP подвыборки: {high_conf_mask.sum()}')

# Median negative values из train
neg_mask = (y_tr == 0)
median_neg = {}
for c in tab_cols + svd_cols:
    if df[c].dtype.kind in 'fi':
        median_neg[c] = float(np.median(X_tr.loc[neg_mask, c]))

# Ключевые признаки для counterfactual анализа
key_features = [
    # FADAML ценовые
    'PriceDiscounted',
    # поведенческие
    'item_count_fake_returns30', 'item_count_returns30', 'item_count_sales30',
    'GmvTotal30', 'ExemplarReturnedCountTotal30',
    # (table-only features, текстовые лежат в SVD/CLIP)
    # временные
    'item_time_alive', 'seller_time_alive',
    # рейтинги
    'rating_1_count', 'rating_5_count',
    # фото
    'photos_published_count', 'comments_published_count',
]
key_features = [f for f in key_features if f in features]

# Для каждой группы признаков считаем ΔP
log(f"{'Группа признаков':40s} {'mean ΔP':>10s} {'std ΔP':>10s} {'median ΔP':>10s}")

# Single-feature counterfactual
X_high = X_te.loc[high_conf_mask].copy().reset_index(drop=True)
p_high_base = p_test_base[high_conf_mask]

for feat in key_features:
    X_mut = X_high.copy()
    X_mut[feat] = median_neg.get(feat, np.median(X_te[feat]))
    p_mut = cb.predict_proba(X_mut)[:, 1]
    delta_p = p_high_base - p_mut  # положительно = признак поддерживал решение
    log(f'  {feat:40s} {delta_p.mean():>10.4f} {delta_p.std():>10.4f} {np.median(delta_p):>10.4f}')

# Группы признаков (mass ablation)
log('--- Group-level counterfactual ---')
groups = {
    'returns_block':   [c for c in features if 'return' in c.lower()],
    'sales_block':     [c for c in features if 'sales' in c.lower() or 'Gmv' in c],
    'rating_block':    [c for c in features if c.startswith('rating_')],
    'time_block':      [c for c in features if 'time_alive' in c],
    'price_block':     [c for c in features if 'Price' in c or 'price' in c.lower()],
    'photos_comments': [c for c in features if 'photo' in c.lower() or 'comment' in c.lower() or 'video' in c.lower()],
    'CLIP_image':      clip_cols,
    'TF-IDF SVD':      svd_cols,
}
for group_name, feats in groups.items():
    X_mut = X_high.copy()
    for f in feats:
        if f in median_neg:
            X_mut[f] = median_neg[f]
        elif f in clip_cols + svd_cols:
            X_mut[f] = 0.0
    p_mut = cb.predict_proba(X_mut)[:, 1]
    delta_p = p_high_base - p_mut
    log(f'  group {group_name:25s} (n={len(feats):3d}): mean ΔP={delta_p.mean():.4f}, std={delta_p.std():.4f}, %resolved={(p_mut < 0.5).mean()*100:.1f}%')

# 7. Counterfactual: что если все «структурные сигналы контрафакта» зануляются — модель «верит»?
log('--- Полный «de-counterfeit» mutation на high-conf TP ---')
X_mut = X_high.copy()
for f in tab_cols:
    if f in median_neg:
        X_mut[f] = median_neg[f]
for f in svd_cols:
    X_mut[f] = 0.0
for f in clip_cols:
    X_mut[f] = 0.0
p_mut_all = cb.predict_proba(X_mut)[:, 1]
log(f'  baseline mean p = {p_high_base.mean():.4f}, после полной мутации mean p = {p_mut_all.mean():.4f}')
log(f'  % переклассифицировано в "оригинал" (p<0.5): {(p_mut_all<0.5).mean()*100:.1f}%')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
