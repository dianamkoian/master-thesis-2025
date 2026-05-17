"""
Альтернативные методы кластеризации для § 4.4.2: DBSCAN, GMM, IsolationForest
вместо K-means, чтобы зафиксировать что вырождение K-means — не артефакт метода,
а свойство данных, и проверить, дадут ли альтернативы менее вырожденную картину.

Каждая методика даёт пару структурных признаков (cluster_id / score),
добавляем к M2-pipeline и сравниваем PR-AUC.
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
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'clustering_alt_indiv_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('Alternative clustering: DBSCAN, GMM, IsolationForest')

df = pd.read_csv(DATA_CSV, encoding='utf-8')
train_idx = np.load(OUT_DIR / 'train_idx_indiv.npy')
val_idx = np.load(OUT_DIR / 'val_idx_indiv.npy')
test_idx = np.load(OUT_DIR / 'test_idx_indiv.npy')
y_test_true = np.load(OUT_DIR / 'y_test_indiv.npy')

# Tab impute
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())

# Численные tab для кластеризации
num_cols = [c for c in tab_cols if df[c].dtype.kind in 'fi']
log(f'численных колонок для кластеризации: {len(num_cols)}')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all_num = df[num_cols].values
scaler.fit(X_all_num[train_idx])
X_scaled = scaler.transform(X_all_num)
X_train_scaled = X_scaled[train_idx]

# ---------- KMeans baseline ----------
from sklearn.cluster import KMeans
log('K-means k=2 (baseline):')
km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_train_scaled)
cluster_km = km.predict(X_scaled).astype(np.int8)
dist_km = km.transform(X_scaled).min(axis=1).astype(np.float32)
log(f'  cluster sizes: {pd.Series(cluster_km).value_counts().to_dict()}')

# ---------- KMeans k=5 ----------
log('K-means k=5:')
km5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_train_scaled)
cluster_km5 = km5.predict(X_scaled).astype(np.int8)
dist_km5 = km5.transform(X_scaled).min(axis=1).astype(np.float32)
log(f'  cluster sizes: {pd.Series(cluster_km5).value_counts().to_dict()}')

# ---------- DBSCAN (on subsample due to O(n^2) complexity) ----------
log('DBSCAN (на сэмпле train 20k для скорости):')
from sklearn.cluster import DBSCAN
sample_size = min(20000, len(train_idx))
sample_subset = np.random.default_rng(42).choice(train_idx, sample_size, replace=False)
X_sample = X_scaled[sample_subset]
db = DBSCAN(eps=2.0, min_samples=20, n_jobs=4).fit(X_sample)
log(f'  DBSCAN на сэмпле {sample_size}: классов {len(set(db.labels_))}, шум {(db.labels_ == -1).sum()}/{sample_size}')

# DBSCAN predict для full — через nearest sample
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(X_sample)
_, nn_idx = nn.kneighbors(X_scaled)
dbscan_label = db.labels_[nn_idx.flatten()].astype(np.int16)
# Distance to nearest core sample
_, dists = nn.kneighbors(X_scaled, n_neighbors=1, return_distance=True), 0
dbscan_dist = nn.kneighbors(X_scaled, n_neighbors=1)[0].flatten().astype(np.float32)
log(f'  DBSCAN distribution в full: {pd.Series(dbscan_label).value_counts().head().to_dict()}')

# ---------- Gaussian Mixture ----------
from sklearn.mixture import GaussianMixture
log('Gaussian Mixture (n_components=5):')
gm = GaussianMixture(n_components=5, random_state=42, max_iter=200, n_init=2)
gm.fit(X_train_scaled)
gmm_label = gm.predict(X_scaled).astype(np.int8)
gmm_logprob = gm.score_samples(X_scaled).astype(np.float32)
log(f'  GMM cluster sizes: {pd.Series(gmm_label).value_counts().to_dict()}')

# ---------- Isolation Forest ----------
from sklearn.ensemble import IsolationForest
log('Isolation Forest:')
iso = IsolationForest(n_estimators=200, contamination=0.066, random_state=42, n_jobs=4)
iso.fit(X_train_scaled)
iso_score = -iso.score_samples(X_scaled).astype(np.float32)  # выше = аномальнее
iso_label = (iso.predict(X_scaled) == -1).astype(np.int8)  # 1 = аномалия
log(f'  IsoForest anomaly: {iso_label.sum()}/{len(iso_label)}')

# ---------- Сравнение через M2 + только структурные ----------
# Берём только CLIP + структурные (без tab base) — для чистоты сравнения
log('Загрузка CLIP для совмещения...')
clip_df = pd.read_parquet(CLIP_PARQUET)
clip_matrix = np.stack(clip_df['embedding'].values).astype(np.float32)
clip_expanded = pd.DataFrame(clip_matrix, columns=[f'img_{i}' for i in range(clip_matrix.shape[1])])
clip_expanded['ItemID'] = clip_df['ItemID'].values
df = df.merge(clip_expanded, on='ItemID', how='left')
clip_cols = [c for c in df.columns if c.startswith('img_')]
df[clip_cols] = df[clip_cols].fillna(0.0)
del clip_df, clip_matrix, clip_expanded; gc.collect()

# Tab cols стрингификация
for c in tab_cols:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).fillna('NA')
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype(str)

# Конфигурации с разными структурными
df['km2_cluster'] = cluster_km
df['km2_dist'] = dist_km
df['km5_cluster'] = cluster_km5
df['km5_dist'] = dist_km5
df['dbscan_cluster'] = dbscan_label
df['dbscan_dist'] = dbscan_dist
df['gmm_cluster'] = gmm_label
df['gmm_logprob'] = gmm_logprob
df['iso_anomaly'] = iso_label
df['iso_score'] = iso_score

configs = {
    'No structural':   tab_cols + clip_cols,
    'K-means k=2':     tab_cols + ['km2_cluster', 'km2_dist'] + clip_cols,
    'K-means k=5':     tab_cols + ['km5_cluster', 'km5_dist'] + clip_cols,
    'DBSCAN':           tab_cols + ['dbscan_cluster', 'dbscan_dist'] + clip_cols,
    'GMM':              tab_cols + ['gmm_cluster', 'gmm_logprob'] + clip_cols,
    'IsolationForest':  tab_cols + ['iso_anomaly', 'iso_score'] + clip_cols,
}

cat_cols = ['CommercialTypeName4']

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

log('=' * 50)
log('=== Сводка структурных альтернатив ===')
results = {}
for cfg_name, feats in configs.items():
    t_c = time.time()
    X_tr = df[feats].iloc[train_idx].reset_index(drop=True)
    y_tr = df['resolution'].values[train_idx]
    X_va = df[feats].iloc[val_idx].reset_index(drop=True)
    y_va = df['resolution'].values[val_idx]
    X_te = df[feats].iloc[test_idx].reset_index(drop=True)
    cb = CatBoostClassifier(
        iterations=1500, depth=8, learning_rate=0.05, eval_metric='AUC',
        random_seed=42, early_stopping_rounds=100, cat_features=cat_cols,
        thread_count=4, verbose=0, auto_class_weights='Balanced',
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    p_te = cb.predict_proba(X_te)[:, 1].astype(np.float32)
    roc = roc_auc_score(y_test_true, p_te)
    pr = average_precision_score(y_test_true, p_te)
    r09 = rap(y_test_true, p_te)
    results[cfg_name] = (roc, pr, r09)
    log(f'  {cfg_name:18s}: ROC={roc:.4f}  PR={pr:.4f}  R@P09={r09:.4f}  ({(time.time()-t_c)/60:.1f} мин)')

log('-' * 50)
log('=== Итоговая таблица (ΔPR vs No structural) ===')
base_pr = results['No structural'][1]
for cfg, (roc, pr, r09) in results.items():
    delta = pr - base_pr
    log(f'  {cfg:18s}: ROC={roc:.4f}  PR={pr:.4f}  R@P09={r09:.4f}  ΔPR={delta:+.4f}')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
