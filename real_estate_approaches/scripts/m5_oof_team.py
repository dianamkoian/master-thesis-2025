"""
M5 OOF Stacking на едином командном split: 5-fold StratifiedGroupKFold
по SellerID, две базовых модели (CatBoost tab-only и CatBoost multi),
LR meta поверх OOF predictions.

Выход: test_proba_diana_m5_team.npy — финальные вероятности на team test.
Цель: вторая новая строка в Таблице 5.1 как "real_estate M5 OOF Stacking".
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
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'm5_oof_team_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('M5 OOF Stacking на team split: старт')

# 1. Данные + индексы
df = pd.read_csv(DATA_CSV, encoding='utf-8')
train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')
log(f'train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

# 2. Текст и CLIP
df['text'] = (df['name_rus'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['brand_name'].fillna(''))
df['text'] = df['text'].str.slice(0, 512)

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

# 3. TF-IDF SVD-50 на текстах train (как в исходном M5)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
log('TF-IDF + SVD-50 на текстах...')
tfv = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
texts_train = df.iloc[train_idx]['text'].values
tfv.fit(texts_train)
X_text_all = tfv.transform(df['text'].values)
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(tfv.transform(texts_train))
svd_arr = svd.transform(X_text_all).astype(np.float32)
log(f'  SVD explained: {svd.explained_variance_ratio_.sum():.4f}')
for i in range(50):
    df[f'svd_{i}'] = svd_arr[:, i]
del X_text_all, svd_arr; gc.collect()

# 4. Tab признаки + категория
meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
cat_cols = ['CommercialTypeName4']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('NA')
for c in cat_cols:
    df[c] = df[c].astype(str)

svd_cols = [f'svd_{i}' for i in range(50)]
features_tab = tab_cols
features_multi = tab_cols + clip_cols + svd_cols
log(f'tab features: {len(features_tab)}, multi features: {len(features_multi)}')

# 5. Подготовка train/val/test данных
X_train_tab = df[features_tab].iloc[train_idx].reset_index(drop=True)
X_train_multi = df[features_multi].iloc[train_idx].reset_index(drop=True)
y_train = df['resolution'].values[train_idx]

X_val_tab = df[features_tab].iloc[val_idx].reset_index(drop=True)
X_val_multi = df[features_multi].iloc[val_idx].reset_index(drop=True)
y_val = df['resolution'].values[val_idx]

X_test_tab = df[features_tab].iloc[test_idx].reset_index(drop=True)
X_test_multi = df[features_multi].iloc[test_idx].reset_index(drop=True)
groups_train = df['SellerID'].values[train_idx]

# 6. 5-fold StratifiedGroupKFold OOF на train
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier

n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_tab = np.zeros(len(y_train), dtype=np.float32)
oof_multi = np.zeros(len(y_train), dtype=np.float32)

for fold, (tr_i, va_i) in enumerate(sgkf.split(X_train_tab, y_train, groups=groups_train)):
    log(f'fold {fold+1}/{n_splits}: train={len(tr_i)} val={len(va_i)}')
    t_fold = time.time()
    # tab base
    m_tab = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.05,
        eval_metric='AUC', random_seed=42 + fold, early_stopping_rounds=80,
        cat_features=cat_cols, thread_count=4, verbose=0, auto_class_weights='Balanced')
    m_tab.fit(X_train_tab.iloc[tr_i], y_train[tr_i],
              eval_set=(X_train_tab.iloc[va_i], y_train[va_i]), use_best_model=True)
    oof_tab[va_i] = m_tab.predict_proba(X_train_tab.iloc[va_i])[:, 1]
    # multi base
    m_multi = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.05,
        eval_metric='AUC', random_seed=142 + fold, early_stopping_rounds=80,
        cat_features=cat_cols, thread_count=4, verbose=0, auto_class_weights='Balanced')
    m_multi.fit(X_train_multi.iloc[tr_i], y_train[tr_i],
                eval_set=(X_train_multi.iloc[va_i], y_train[va_i]), use_best_model=True)
    oof_multi[va_i] = m_multi.predict_proba(X_train_multi.iloc[va_i])[:, 1]
    del m_tab, m_multi; gc.collect()
    log(f'  fold {fold+1} done in {(time.time()-t_fold)/60:.1f} мин')

# 7. Финальные base модели на полном train
log('Финальные base модели на полном train...')
m_tab_full = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.05,
    eval_metric='AUC', random_seed=42, early_stopping_rounds=80,
    cat_features=cat_cols, thread_count=4, verbose=0, auto_class_weights='Balanced')
m_tab_full.fit(X_train_tab, y_train, eval_set=(X_val_tab, y_val), use_best_model=True)
p_tab_test = m_tab_full.predict_proba(X_test_tab)[:, 1].astype(np.float32)

m_multi_full = CatBoostClassifier(iterations=1500, depth=7, learning_rate=0.05,
    eval_metric='AUC', random_seed=142, early_stopping_rounds=80,
    cat_features=cat_cols, thread_count=4, verbose=0, auto_class_weights='Balanced')
m_multi_full.fit(X_train_multi, y_train, eval_set=(X_val_multi, y_val), use_best_model=True)
p_multi_test = m_multi_full.predict_proba(X_test_multi)[:, 1].astype(np.float32)

# 8. LR meta на OOF features
from sklearn.linear_model import LogisticRegression
log('LR meta на OOF features...')
meta_X = np.column_stack([oof_tab, oof_multi])
meta_y = y_train
lr_meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
lr_meta.fit(meta_X, meta_y)
log(f'  LR coefs: {lr_meta.coef_[0]}, intercept: {lr_meta.intercept_[0]:.4f}')

# 9. Финальное предсказание
test_meta_X = np.column_stack([p_tab_test, p_multi_test])
p_test_m5 = lr_meta.predict_proba(test_meta_X)[:, 1].astype(np.float32)

# 10. Метрики
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

log('=' * 50)
log('Метрики на team test (истинные метки):')
log(f'  base tab only:    ROC={roc_auc_score(y_test_true, p_tab_test):.4f}  PR={average_precision_score(y_test_true, p_tab_test):.4f}  R@P>=0.9={rap(y_test_true, p_tab_test):.4f}')
log(f'  base multi:       ROC={roc_auc_score(y_test_true, p_multi_test):.4f}  PR={average_precision_score(y_test_true, p_multi_test):.4f}  R@P>=0.9={rap(y_test_true, p_multi_test):.4f}')
log(f'  M5 LR-meta stack: ROC={roc_auc_score(y_test_true, p_test_m5):.4f}  PR={average_precision_score(y_test_true, p_test_m5):.4f}  R@P>=0.9={rap(y_test_true, p_test_m5):.4f}')

p_m2 = np.load(OUT_DIR / 'test_proba_diana_team.npy')
log(f'  M2 (current headline для справки): ROC={roc_auc_score(y_test_true, p_m2):.4f}  PR={average_precision_score(y_test_true, p_m2):.4f}')

# 11. Save
np.save(OUT_DIR / 'test_proba_diana_m5_team.npy', p_test_m5)
np.save(OUT_DIR / 'oof_tab_diana_m5.npy', oof_tab)
np.save(OUT_DIR / 'oof_multi_diana_m5.npy', oof_multi)
log(f'Saved → test_proba_diana_m5_team.npy')
log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
