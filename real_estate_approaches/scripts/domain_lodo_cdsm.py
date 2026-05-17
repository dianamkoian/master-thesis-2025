"""
Настоящий domain-LODO для CDSM v3 4-ch.

Для каждого автора переобучаем CDSM с убранными feature-группами + убираем external channel.

Distribution feature-групп по авторам:
- Diana (RE):   H (CLIP-structural)
- Soni (FT):    F (e5), G (CLIP-PCA)
- Karina (MA):  D (typosquat), I (KL feature)
- Albina (SN):  архитектурные элементы (sample_weights, 3-mode разделение)
- Cross-domain (e-commerce fraud): C (FADAML, симметрично Diana + Karina)
- Общекомандный baseline: A (38 tab), E (TF-IDF SVD)

LODO-конфигурации:
  base:      все группы (v3 baseline) — для подтверждения
  no_diana:  CDSM без H и C; final LR без M2-FE+ external channel
  no_soni:   CDSM без F и G; final LR без Fusion external channel
  no_karina: CDSM без D и I и C; final LR на 4 channels (CDSM_no_karina + 3 external — без Karina-external, т.к. у неё нет ext)
  no_albina: CDSM без sample_weights (uniform weights в meta-LR); final LR без MMD-v4 external

Note про FADAML (C): симметричный transfer. Для no_diana и no_karina удаляем C полностью
(методология не приписывается к одному автору). Для no_soni и no_albina C остаётся.
"""
from __future__ import annotations
import os, time, json, warnings
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from catboost import CatBoostClassifier
from rapidfuzz.fuzz import partial_ratio
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'packages'
NB  = ROOT / 'real_estate_approaches' / 'notebooks'
OUT_DIR = NB / 'cdsm'
SEED = 42

def rap90(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    m = pr >= 0.9
    return float(rc[m].max()) if m.any() else 0.0

def metrics(y, p):
    return {'PR': float(average_precision_score(y, p)),
            'ROC': float(roc_auc_score(y, p)),
            'R@P': rap90(y, p)}

# ============ Load ============
df = pd.read_csv(ROOT/"data"/'ml_ozon_ounterfeit_train.csv', encoding='utf-8')
train_idx = np.load(PKG/'package_diana'/'3_probas_team_split'/'team_train_idx.npy')
val_idx   = np.load(PKG/'package_diana'/'3_probas_team_split'/'team_val_idx.npy')
test_idx  = np.load(PKG/'package_diana'/'3_probas_team_split'/'team_test_idx.npy')
y_test = np.load(PKG/'package_diana'/'3_probas_team_split'/'y_test_team.npy').astype(np.int8)
y_train = df.iloc[train_idx]['resolution'].astype('int8').values
y_val   = df.iloc[val_idx]['resolution'].astype('int8').values
y_tv = np.concatenate([y_train, y_val])
df_tr, df_va, df_te = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
df_tv = pd.concat([df_tr, df_va], ignore_index=True)
n_tv = len(df_tv)
scale_pos = (y_tv == 0).sum() / max((y_tv == 1).sum(), 1)
sellers = np.load(NB/'team_test_sellers.npy')

# External channels (pre-computed)
p_diana_ext  = np.load(PKG/'package_diana'/'3_probas_team_split'/'test_proba_diana_team.npy').astype(np.float64)
p_soni_ext   = np.load(PKG/'package_soni'/'3_probas_team_split'/'test_proba_fusion_team.npy').astype(np.float64)
p_albina_ext = np.load(PKG/'package_albina'/'3_probas_team_split'/'test_proba_albina_team.npy').astype(np.float64)

print(f'[{time.strftime("%H:%M:%S")}] data loaded, scale_pos={scale_pos:.2f}')

# ============ Feature builders ============
DROP = {'id','ItemID','SellerID','name_rus','description','brand_name','text','resolution'}
TAB38 = [c for c in df.columns if c not in DROP]
CAT_TAB38 = [c for c in TAB38 if df[c].dtype == 'object']

def build_tab(part):
    X = part[TAB38].copy()
    for c in TAB38:
        if c in CAT_TAB38:
            X[c] = X[c].fillna('nan').astype(str)
        else:
            X[c] = X[c].fillna(0)
    return X

def build_text(part):
    parts = part[['brand_name','description','name_rus']].fillna('').astype(str)
    return (parts['brand_name'] + ' ' + parts['description'] + ' ' + parts['name_rus']).values

# FADAML
def build_fadaml(part, cat_med):
    cm = part['CommercialTypeName4'].map(cat_med).fillna(part['PriceDiscounted'].median())
    pr = (part['PriceDiscounted'].fillna(0) / (cm + 1)).astype(np.float32)
    return np.column_stack([pr, (pr<0.3).astype(np.float32), (pr>1.5).astype(np.float32)])
cat_med = df_tv.groupby('CommercialTypeName4')['PriceDiscounted'].median().to_dict()

# Deng
def build_deng(part):
    b = part['brand_name'].fillna('').astype(str).str.lower()
    n = part['name_rus'].fillna('').astype(str).str.lower()
    be = np.array([(x in y) and len(x)>0 for x,y in zip(b,n)], dtype=np.float32)
    bf = np.array([partial_ratio(x,y)/100.0 if (x and y) else 0.0 for x,y in zip(b,n)], dtype=np.float32)
    ts = np.maximum(0, bf - 0.5*be)
    return np.column_stack([be, bf, ts])

# TF-IDF
text_tv = build_text(df_tv); text_te = build_text(df_te)
tf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=5)
X_tf_tv = tf.fit_transform(text_tv); X_tf_te = tf.transform(text_te)
svd = TruncatedSVD(n_components=50, random_state=SEED)
E_tv = svd.fit_transform(X_tf_tv).astype(np.float32)
E_te = svd.transform(X_tf_te).astype(np.float32)

# e5 (precomputed cache)
e5_all = np.load(OUT_DIR/'e5_small_embeddings.npy')
e5_tv = e5_all[np.concatenate([train_idx, val_idx])]
e5_te = e5_all[test_idx]
pca_e5 = PCA(n_components=25, random_state=SEED)
F_tv = pca_e5.fit_transform(e5_tv).astype(np.float32)
F_te = pca_e5.transform(e5_te).astype(np.float32)

# CLIP
clip = pd.read_parquet(ROOT/'counterfeit_service'/'clip_embeddings.parquet')
clip_mat = np.stack(clip['embedding'].values).astype(np.float32)
clip_ids = clip['ItemID'].values
id2row = {int(i): r for r, i in enumerate(clip_ids)}
clip_dim = clip_mat.shape[1]
def merge_clip(part):
    n = len(part); X = np.zeros((n, clip_dim), dtype=np.float32)
    ids = part['id'].values
    for i, item_id in enumerate(ids):
        r = id2row.get(int(item_id))
        if r is not None: X[i] = clip_mat[r]
    return X
X_clip_tv_raw = merge_clip(df_tv); X_clip_te_raw = merge_clip(df_te)
pca_clip = PCA(n_components=25, random_state=SEED)
G_tv = pca_clip.fit_transform(X_clip_tv_raw).astype(np.float32)
G_te = pca_clip.transform(X_clip_te_raw).astype(np.float32)

# CLIP-structural
clip_tv_n = X_clip_tv_raw / (np.linalg.norm(X_clip_tv_raw, axis=1, keepdims=True) + 1e-9)
clip_te_n = X_clip_te_raw / (np.linalg.norm(X_clip_te_raw, axis=1, keepdims=True) + 1e-9)
cats = df_tv['CommercialTypeName4'].fillna('NA').astype(str).values
centroids = {c: clip_tv_n[cats==c].mean(axis=0) for c in np.unique(cats)}
def build_clip_struct(part, X_clip_n):
    cats_p = part['CommercialTypeName4'].fillna('NA').astype(str).values
    dist = np.zeros(len(part), dtype=np.float32)
    for i, c in enumerate(cats_p):
        if c in centroids: dist[i] = float(np.dot(X_clip_n[i], centroids[c]))
    norms = np.linalg.norm(X_clip_n, axis=1).astype(np.float32)
    return np.column_stack([dist, norms, np.abs(dist - 0.5), dist * norms])
H_tv = build_clip_struct(df_tv, clip_tv_n); H_te = build_clip_struct(df_te, clip_te_n)

# KL via OOF
print(f'[{time.strftime("%H:%M:%S")}] computing KL OOF (5-fold tab+text)...')
X_tab_tv_pd = build_tab(df_tv); X_tab_te_pd = build_tab(df_te)
skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
p_tab_oof = np.zeros(n_tv); p_text_oof = np.zeros(n_tv)
for tr_i, va_i in skf_oof.split(X_tab_tv_pd, y_tv):
    cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, scale_pos_weight=scale_pos,
                            random_seed=SEED, verbose=0, early_stopping_rounds=30)
    cb.fit(X_tab_tv_pd.iloc[tr_i], y_tv[tr_i], cat_features=CAT_TAB38,
           eval_set=(X_tab_tv_pd.iloc[va_i], y_tv[va_i]), use_best_model=True)
    p_tab_oof[va_i] = cb.predict_proba(X_tab_tv_pd.iloc[va_i])[:, 1]
    lr_t = LogisticRegression(class_weight='balanced', max_iter=500, random_state=SEED)
    tf_fold = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=5)
    lr_t.fit(tf_fold.fit_transform(text_tv[tr_i]), y_tv[tr_i])
    p_text_oof[va_i] = lr_t.predict_proba(tf_fold.transform(text_tv[va_i]))[:, 1]

def kl(p, q, eps=1e-6):
    p = np.clip(p, eps, 1-eps); q = np.clip(q, eps, 1-eps)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
I_tv = kl(p_text_oof, p_tab_oof).astype(np.float32).reshape(-1, 1)

cb_full = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, scale_pos_weight=scale_pos,
                              random_seed=SEED, verbose=0)
cb_full.fit(X_tab_tv_pd, y_tv, cat_features=CAT_TAB38)
lr_t_full = LogisticRegression(class_weight='balanced', max_iter=500, random_state=SEED)
tf_full = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=5)
lr_t_full.fit(tf_full.fit_transform(text_tv), y_tv)
p_tab_te = cb_full.predict_proba(X_tab_te_pd)[:, 1]
p_text_te = lr_t_full.predict_proba(tf_full.transform(text_te))[:, 1]
I_te = kl(p_text_te, p_tab_te).astype(np.float32).reshape(-1, 1)
print(f'[{time.strftime("%H:%M:%S")}] KL OOF done')

# C (FADAML)
C_tv = build_fadaml(df_tv, cat_med)
C_te = build_fadaml(df_te, cat_med)
# D (Deng)
D_tv = build_deng(df_tv)
D_te = build_deng(df_te)

# ============ Helper: assemble feature matrix from group set ============
def assemble(groups):
    parts_tv, parts_te = [], []; cat_idx = []
    if 'A' in groups:
        A_tv = build_tab(df_tv).values
        A_te = build_tab(df_te).values
        cat_idx.extend([TAB38.index(c) for c in CAT_TAB38])
        parts_tv.append(A_tv); parts_te.append(A_te)
    if 'C' in groups: parts_tv.append(C_tv); parts_te.append(C_te)
    if 'D' in groups: parts_tv.append(D_tv); parts_te.append(D_te)
    if 'E' in groups: parts_tv.append(E_tv); parts_te.append(E_te)
    if 'F' in groups: parts_tv.append(F_tv); parts_te.append(F_te)
    if 'G' in groups: parts_tv.append(G_tv); parts_te.append(G_te)
    if 'H' in groups: parts_tv.append(H_tv); parts_te.append(H_te)
    if 'I' in groups: parts_tv.append(I_tv); parts_te.append(I_te)
    return np.hstack(parts_tv), np.hstack(parts_te), cat_idx

def train_mode_oof(groups, label, cb_iters=600):
    X_tv, X_te, cat_idx = assemble(groups)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(n_tv, dtype=np.float32)
    tp  = np.zeros(len(y_test), dtype=np.float32)
    t0 = time.time()
    for tr, va in skf.split(X_tv, y_tv):
        cb = CatBoostClassifier(iterations=cb_iters, depth=6, learning_rate=0.05,
                                 scale_pos_weight=scale_pos, random_seed=SEED, verbose=0,
                                 early_stopping_rounds=40, cat_features=cat_idx)
        cb.fit(X_tv[tr], y_tv[tr], eval_set=(X_tv[va], y_tv[va]), use_best_model=True)
        oof[va] = cb.predict_proba(X_tv[va])[:, 1]
        tp += cb.predict_proba(X_te)[:, 1] / 5
    print(f'  {label}: {time.time()-t0:.0f}s')
    return oof, tp

def meta(oof_set, test_set, with_sw=True):
    M_tv = np.column_stack([oof_set[0], oof_set[1], oof_set[2],
                              np.abs(oof_set[0]-oof_set[1]), np.abs(oof_set[1]-oof_set[2]), np.abs(oof_set[0]-oof_set[2]),
                              oof_set[0]*oof_set[1], oof_set[0]*oof_set[2], oof_set[1]*oof_set[2]])
    M_te = np.column_stack([test_set[0], test_set[1], test_set[2],
                              np.abs(test_set[0]-test_set[1]), np.abs(test_set[1]-test_set[2]), np.abs(test_set[0]-test_set[2]),
                              test_set[0]*test_set[1], test_set[0]*test_set[2], test_set[1]*test_set[2]])
    sc = StandardScaler().fit(M_tv)
    sw = (1 + (1 - 2*np.abs(oof_set[0]-0.5))) if with_sw else None
    lr = LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced', random_state=SEED)
    lr.fit(sc.transform(M_tv), y_tv, sample_weight=sw)
    return lr.predict_proba(sc.transform(M_te))[:, 1]

def build_cdsm(mode1_g, mode2_g, mode3_g, with_sw=True, label=''):
    print(f'[{time.strftime("%H:%M:%S")}] {label}: m1={sorted(mode1_g)}, m2={sorted(mode2_g)}, m3={sorted(mode3_g)}, sw={with_sw}')
    oof1, te1 = train_mode_oof(mode1_g, f'  m1')
    oof2, te2 = train_mode_oof(mode2_g, f'  m2')
    oof3, te3 = train_mode_oof(mode3_g, f'  m3')
    return meta((oof1, oof2, oof3), (te1, te2, te3), with_sw=with_sw)

# Baseline v3 reuse
p_cdsm_v3 = np.load(OUT_DIR/'p_cdsm.npy').astype(np.float64)

def final_lr(channels, C=0.1):
    X = np.column_stack(channels)
    g = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    fi, _ = next(g.split(X, y_test, groups=sellers))
    sc = StandardScaler().fit(X[fi])
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=SEED)
    lr.fit(sc.transform(X[fi]), y_test[fi])
    return lr.predict_proba(sc.transform(X))[:, 1], lr.coef_[0]

# ============ LODO configurations ============
# v3 baseline groups
G3_M1 = {'A','C','D'}
G3_M2 = {'A','C','D','E','F'}
G3_M3 = {'A','C','D','E','F','G','H','I'}

results = {}
p_base, coefs_base = final_lr([p_cdsm_v3, p_diana_ext, p_soni_ext, p_albina_ext])
results['baseline_v3_4ch'] = {'metrics': metrics(y_test, p_base), 'coefs': list(coefs_base)}
print(f'\nBASELINE v3 4-ch: {results["baseline_v3_4ch"]["metrics"]}')

# no_diana — remove H from CDSM + C symmetric (½ Diana) + drop Diana M2-FE+ external
print(f'\n[{time.strftime("%H:%M:%S")}] ===== NO_DIANA (remove H from CDSM, also C symmetric, drop M2-FE+ external) =====')
p_no_diana = build_cdsm({'A','D'}, {'A','D','E','F'}, {'A','D','E','F','G','I'}, label='CDSM_no_diana')
p, coefs = final_lr([p_no_diana, p_soni_ext, p_albina_ext])
results['no_diana'] = {'metrics': metrics(y_test, p), 'coefs': list(coefs)}
print(f'no_diana: {results["no_diana"]["metrics"]}')

# no_soni — remove F, G from CDSM + drop Soni Fusion external
print(f'\n[{time.strftime("%H:%M:%S")}] ===== NO_SONI (remove F, G from CDSM, drop Fusion external) =====')
p_no_soni = build_cdsm({'A','C','D'}, {'A','C','D','E'}, {'A','C','D','E','H','I'}, label='CDSM_no_soni')
p, coefs = final_lr([p_no_soni, p_diana_ext, p_albina_ext])
results['no_soni'] = {'metrics': metrics(y_test, p), 'coefs': list(coefs)}
print(f'no_soni: {results["no_soni"]["metrics"]}')

# no_karina — remove D, I, C from CDSM + Karina has no external channel
print(f'\n[{time.strftime("%H:%M:%S")}] ===== NO_KARINA (remove D, I, C from CDSM) =====')
p_no_karina = build_cdsm({'A'}, {'A','E','F'}, {'A','E','F','G','H'}, label='CDSM_no_karina')
p, coefs = final_lr([p_no_karina, p_diana_ext, p_soni_ext, p_albina_ext])
results['no_karina'] = {'metrics': metrics(y_test, p), 'coefs': list(coefs)}
print(f'no_karina: {results["no_karina"]["metrics"]}')

# no_albina — CDSM без sample_weights + drop MMD-v4 external
print(f'\n[{time.strftime("%H:%M:%S")}] ===== NO_ALBINA (CDSM без sample_weights, drop MMD-v4 external) =====')
p_no_albina = build_cdsm(G3_M1, G3_M2, G3_M3, with_sw=False, label='CDSM_no_albina')
p, coefs = final_lr([p_no_albina, p_diana_ext, p_soni_ext])
results['no_albina'] = {'metrics': metrics(y_test, p), 'coefs': list(coefs)}
print(f'no_albina: {results["no_albina"]["metrics"]}')

# Save
with open(OUT_DIR/'domain_lodo_full.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nsaved {OUT_DIR/"domain_lodo_full.json"}')

# Summary
print(f'\n=== SUMMARY: domain-LODO ===')
print(f'{"config":<28} {"PR":>8} {"R@P":>8} {"ROC":>8} {"dPR":>10} {"dR@P":>10}')
print('-'*80)
m_b = results['baseline_v3_4ch']['metrics']
print('{:<28} {:>8.4f} {:>8.4f} {:>8.4f} {:>10} {:>10}'.format('baseline', m_b['PR'], m_b['R@P'], m_b['ROC'], '—', '—'))
for k in ['no_diana','no_soni','no_karina','no_albina']:
    m = results[k]['metrics']
    print('{:<28} {:>8.4f} {:>8.4f} {:>8.4f} {:>+10.4f} {:>+10.4f}'.format(k, m['PR'], m['R@P'], m['ROC'], m['PR']-m_b['PR'], m['R@P']-m_b['R@P']))
