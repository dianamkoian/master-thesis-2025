"""
Mondrian (group-conditional) conformal predictor для § 4.4.9.5.
Альтернатива split conformal: квантиль рассчитывается отдельно
для каждой группы (категории товара CommercialTypeName4) на val,
затем применяется к test-объектам соответствующей группы.

Цель: проверить, выполняются ли формальные FPR-гарантии лучше
в условиях covariate shift между val и test.

Также: adaptive conformal (CQR-style) через нонкомформ-оценку
|y - p|, как в Romano et al. 2020 [43].
"""
import time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path('/Users/diana/master-thesis-2025')
DATA_CSV = ROOT / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv'
SPLITS = ROOT / 'Диана_ВКР_финал' / 'notebooks' / 'team_splits'
OUT_DIR = ROOT / 'Диана_ВКР_финал' / 'notebooks'
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'mondrian_mainline_indiv_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('Mondrian conformal + Adaptive conformal: старт')

# 1. Загрузка
df = pd.read_csv(DATA_CSV, encoding='utf-8', usecols=['ItemID', 'resolution', 'SellerID', 'CommercialTypeName4'])
val_idx = np.load(OUT_DIR / 'val_idx_indiv.npy')
test_idx = np.load(OUT_DIR / 'test_idx_indiv.npy')
y_test_true = np.load(OUT_DIR / 'y_test_indiv.npy')

y_val = df['resolution'].values[val_idx]
cat_val = df['CommercialTypeName4'].fillna('NA').values[val_idx]
cat_test = df['CommercialTypeName4'].fillna('NA').values[test_idx]

p_val = np.load(OUT_DIR / 'val_proba_diana_m2_mainline_indiv.npy').astype(np.float32)
p_test = np.load(OUT_DIR / 'test_proba_diana_m2_mainline_indiv.npy').astype(np.float32)

log(f'val={len(p_val)} test={len(p_test)}  y_val positive {y_val.mean():.4f}  y_test positive {y_test_true.mean():.4f}')
log(f'категорий в val: {len(set(cat_val))}, в test: {len(set(cat_test))}')

# 2. Split conformal (baseline, оригинальный § 4.4.9.5)
log('=' * 50)
log('--- Split conformal (baseline) ---')
neg_val_mask = (y_val == 0)
p_val_neg = p_val[neg_val_mask]

def split_conformal(target_fpr):
    threshold = np.quantile(p_val_neg, 1 - target_fpr)
    pred_test = (p_test >= threshold).astype(int)
    tp = ((pred_test == 1) & (y_test_true == 1)).sum()
    fp = ((pred_test == 1) & (y_test_true == 0)).sum()
    fn = ((pred_test == 0) & (y_test_true == 1)).sum()
    tn = ((pred_test == 0) & (y_test_true == 0)).sum()
    observed_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return threshold, observed_fpr, precision, recall

log(f"{'target FPR':>12s} {'threshold':>10s} {'obs FPR':>10s} {'precision':>10s} {'recall':>10s}")
for ε in [0.01, 0.05, 0.10]:
    th, obs, prec, rec = split_conformal(ε)
    mark = '✓' if obs <= ε else '✗'
    log(f'  {ε:>10.2f}   {th:>10.4f}   {obs:>10.4f} {mark}  {prec:>10.4f} {rec:>10.4f}')

# 3. Mondrian conformal: per-group threshold
log('=' * 50)
log('--- Mondrian conformal (per CommercialTypeName4) ---')
log('Для каждой категории — свой quantile на val negatives, на test применяем threshold категории')

# Все категории, встречающиеся в val или test
all_cats = set(cat_val) | set(cat_test)
log(f'Уникальных категорий: {len(all_cats)}')

# Для каждой категории строим threshold на val_neg
def mondrian_conformal(target_fpr, min_val_size=50):
    """Возвращает обобщённую threshold для test-объектов по группе.

    Если в группе мало val-объектов — fallback на глобальный threshold."""
    global_thr = np.quantile(p_val_neg, 1 - target_fpr)
    group_thr = {}
    for cat in all_cats:
        val_mask = (cat_val == cat) & (y_val == 0)
        if val_mask.sum() >= min_val_size:
            group_thr[cat] = np.quantile(p_val[val_mask], 1 - target_fpr)
        else:
            group_thr[cat] = global_thr
    # Применить к test
    test_thresholds = np.array([group_thr.get(c, global_thr) for c in cat_test])
    pred_test = (p_test >= test_thresholds).astype(int)
    tp = ((pred_test == 1) & (y_test_true == 1)).sum()
    fp = ((pred_test == 1) & (y_test_true == 0)).sum()
    fn = ((pred_test == 0) & (y_test_true == 1)).sum()
    tn = ((pred_test == 0) & (y_test_true == 0)).sum()
    observed_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    n_with_group_thr = sum(1 for c in cat_test if c in group_thr and group_thr[c] != global_thr)
    return observed_fpr, precision, recall, n_with_group_thr

log(f"{'target FPR':>12s} {'obs FPR (Mondrian)':>20s} {'precision':>10s} {'recall':>10s} {'#group-thr':>11s}")
for ε in [0.01, 0.05, 0.10]:
    obs, prec, rec, n_gt = mondrian_conformal(ε)
    mark = '✓' if obs <= ε else '✗'
    log(f'  {ε:>10.2f}   {obs:>20.4f} {mark}  {prec:>10.4f} {rec:>10.4f}  {n_gt:>11d}')

# 4. Adaptive conformal (по Romano 2020 [43]): non-conformity = max(p - 1, -p)
#    Здесь упрощаем: используем absolute residual |y - p|, перцентиль 1-α
log('=' * 50)
log('--- Adaptive conformal (split-style на |y - p|) ---')
# Non-conformity for negatives = p_val[neg]  (более вероятно у positive объектов)
# Эквивалентно split conformal по сути; но adaptive вариант делает quantile регрессию на признаках,
# что требует обучения отдельной модели. В упрощённом split-варианте =  split conformal.

# Lightweight CQR-style: используем difficulty-conditional thresholds.
# Группируем val по биннам предсказанной вероятности (4 бина), threshold per bin.
log('Difficulty-conditional thresholds по биннам p_val:')
bin_edges = np.quantile(p_val, [0, 0.25, 0.5, 0.75, 1.0])
val_bins = np.clip(np.searchsorted(bin_edges[1:-1], p_val), 0, 3)
test_bins = np.clip(np.searchsorted(bin_edges[1:-1], p_test), 0, 3)

def adaptive_split_conformal(target_fpr):
    bin_thr = {}
    for b in range(4):
        mask = (val_bins == b) & (y_val == 0)
        if mask.sum() >= 50:
            bin_thr[b] = np.quantile(p_val[mask], 1 - target_fpr)
        else:
            bin_thr[b] = np.quantile(p_val_neg, 1 - target_fpr)
    test_thresholds = np.array([bin_thr[b] for b in test_bins])
    pred_test = (p_test >= test_thresholds).astype(int)
    tp = ((pred_test == 1) & (y_test_true == 1)).sum()
    fp = ((pred_test == 1) & (y_test_true == 0)).sum()
    tn = ((pred_test == 0) & (y_test_true == 0)).sum()
    fn = ((pred_test == 0) & (y_test_true == 1)).sum()
    return (fp / (fp + tn) if (fp + tn) > 0 else 0,
            tp / (tp + fp) if (tp + fp) > 0 else 0,
            tp / (tp + fn) if (tp + fn) > 0 else 0)

log(f"{'target FPR':>12s} {'obs FPR (adaptive)':>20s} {'precision':>10s} {'recall':>10s}")
for ε in [0.01, 0.05, 0.10]:
    obs, prec, rec = adaptive_split_conformal(ε)
    mark = '✓' if obs <= ε else '✗'
    log(f'  {ε:>10.2f}   {obs:>20.4f} {mark}  {prec:>10.4f} {rec:>10.4f}')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
