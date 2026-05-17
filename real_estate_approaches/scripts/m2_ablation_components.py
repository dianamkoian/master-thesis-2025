"""
M2 Ablation вклада каждой компоненты: переобучение M2 на team split
с поочерёдным исключением одного из блоков признаков.

Конфигурации:
1. M2 full (baseline)              — tab + K-means + CLIP + TF-IDF SVD
2. M2 без K-means structural        — убираем cluster_id, dist_centroid
3. M2 без CLIP                       — убираем все 512 img_*
4. M2 без TF-IDF SVD                 — убираем все svd_*

Выход: метрики каждой конфигурации + npy probas.
"""
import gc, time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
CLIP_PARQUET = ROOT / 'counterfeit_service' / 'clip_embeddings.parquet'
SPLITS = ROOT / 'real_estate_approaches' / 'notebooks' / 'team_splits'
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm2_ablation_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M2 ablation вклада компонент: старт')

# 1. Данные
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
df['text'] = df['text'].str.slice(0, 512)

train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')
log(f'train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

# 2. CLIP
log('Загрузка CLIP...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
log(f'CLIP merged: {len(clip_cols)} cols')
del clip_df, clip_matrix, clip_expanded; gc.collect()

# 3. TF-IDF SVD-50
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

# 4. K-means structural features (на train fold)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
log('K-means structural features...')
meta = json.load(open(SPLITS / 'team_split_meta.json'))
base_tab_cols = meta['team_feature_cols']
for c in base_tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')

km_features_input = df[base_tab_cols].select_dtypes(include='number').iloc[train_idx]
scaler = StandardScaler()
km_scaled_train = scaler.fit_transform(km_features_input)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(km_scaled_train)
all_numeric = df[base_tab_cols].select_dtypes(include='number')
km_scaled_all = scaler.transform(all_numeric)
df['cluster_id'] = km.predict(km_scaled_all).astype(np.float32)
distances = km.transform(km_scaled_all)
df['dist_centroid'] = distances.min(axis=1).astype(np.float32)
kmeans_cols = ['cluster_id', 'dist_centroid']
log(f'  cluster distribution: {pd.Series(df["cluster_id"]).value_counts().to_dict()}')

# Cat cols
cat_cols = ['CommercialTypeName4']
df[cat_cols[0]] = df[cat_cols[0]].astype(str)

# 5. Конфигурации ablation
configs = {
    'M2 full':           base_tab_cols + kmeans_cols + clip_cols + svd_cols,
    'M2 без K-means':    base_tab_cols + clip_cols + svd_cols,
    'M2 без CLIP':       base_tab_cols + kmeans_cols + svd_cols,
    'M2 без TF-IDF SVD': base_tab_cols + kmeans_cols + clip_cols,
}

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

results = {}
for cfg_name, feats in configs.items():
    log(f'--- {cfg_name} (n_features={len(feats)}) ---')
    t_c = time.time()
    X_tr = df[feats].iloc[train_idx].reset_index(drop=True)
    y_tr = df['resolution'].values[train_idx]
    X_va = df[feats].iloc[val_idx].reset_index(drop=True)
    y_va = df['resolution'].values[val_idx]
    X_te = df[feats].iloc[test_idx].reset_index(drop=True)
    cb = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.05,
        eval_metric='AUC', random_seed=42, early_stopping_rounds=100,
        cat_features=cat_cols, thread_count=4, verbose=0,
        auto_class_weights='Balanced',
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    p_te = cb.predict_proba(X_te)[:, 1].astype(np.float32)
    roc = roc_auc_score(y_test_true, p_te)
    pr = average_precision_score(y_test_true, p_te)
    r09 = rap(y_test_true, p_te)
    log(f'  {cfg_name}: ROC={roc:.4f}  PR={pr:.4f}  R@P09={r09:.4f}  (best iter={cb.tree_count_}, {(time.time()-t_c)/60:.1f} мин)')
    results[cfg_name] = (roc, pr, r09)
    # Save probas
    safe_name = cfg_name.replace(' ', '_').replace('M2_full', 'm2_full').replace('M2_без_', 'm2_no_')
    np.save(OUT_DIR / f'test_proba_diana_ablation_{safe_name}.npy', p_te)

log('=' * 50)
log('=== Сводка ablation вклада компонент M2 ===')
log(f'{"Конфигурация":25s} {"ROC":>8s} {"PR-AUC":>8s} {"R@P09":>8s} {"ΔPR":>8s}')
full_pr = results['M2 full'][1]
for cfg, (roc, pr, r09) in results.items():
    delta = pr - full_pr if cfg != 'M2 full' else 0
    log(f'{cfg:25s} {roc:8.4f} {pr:8.4f} {r09:8.4f} {delta:+8.4f}')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
