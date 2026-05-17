"""
Bootstrap CI для ансамблей M2-FE+ × M2 mainline (calibrated) на индив тесте.
Дополнение к bootstrap_ci_grid_indiv.py: те модели и конфиги, которые используют
M2 mainline calibrated proba, а не simplified M2 baseline.
"""
import time, os, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'bootstrap_ci_mainline_indiv_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

def bootstrap_ci(y_true, p, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], p[idx]))
        except Exception:
            continue
    a = np.array(vals)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

t0 = time.time()
log('=' * 60)
log('Bootstrap CI на mainline-ансамблях M2-FE+ × M2 mainline INDIV')

y_test = np.load(OUT_DIR / 'y_test_indiv.npy')
log(f'y_test_indiv: n={len(y_test)}, positive={int(y_test.sum())} ({y_test.mean()*100:.2f}%)')

p_fe = np.load(OUT_DIR / 'test_proba_diana_m2_fe_plus_indiv.npy').astype(np.float64)
p_m2_mainline = np.load(OUT_DIR / 'test_proba_diana_m2_mainline_indiv.npy').astype(np.float64)
log(f'loaded M2-FE+ INDIV n={len(p_fe)}; M2 mainline INDIV n={len(p_m2_mainline)}')

models = {
    'M2 mainline INDIV (calibrated)': p_m2_mainline,
    'M2-FE+ INDIV': p_fe,
    'M2-FE+ + M2 mainline (50/50, headline по PR)': 0.50 * p_fe + 0.50 * p_m2_mainline,
    'M2-FE+ + M2 mainline (w=0.75, headline по R@P)': 0.75 * p_fe + 0.25 * p_m2_mainline,
}

log('=' * 60)
log(f'{"Модель":50s} {"ROC [95% CI]":>26s} {"PR [95% CI]":>26s} {"R@P09 [95% CI]":>26s}')
results = {}
for name, p in models.items():
    roc_m, roc_lo, roc_hi = bootstrap_ci(y_test, p, roc_auc_score)
    pr_m, pr_lo, pr_hi = bootstrap_ci(y_test, p, average_precision_score)
    r9_m, r9_lo, r9_hi = bootstrap_ci(y_test, p, rap)
    results[name] = {
        'ROC': {'mean': roc_m, 'ci_lo': roc_lo, 'ci_hi': roc_hi},
        'PR-AUC': {'mean': pr_m, 'ci_lo': pr_lo, 'ci_hi': pr_hi},
        'R@P09': {'mean': r9_m, 'ci_lo': r9_lo, 'ci_hi': r9_hi},
    }
    log(f'{name:50s} {roc_m:.4f}[{roc_lo:.4f},{roc_hi:.4f}] {pr_m:.4f}[{pr_lo:.4f},{pr_hi:.4f}] {r9_m:.4f}[{r9_lo:.4f},{r9_hi:.4f}]')

with open(OUT_DIR / 'bootstrap_ci_mainline_indiv_summary.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log(f'Summary JSON: {OUT_DIR / "bootstrap_ci_mainline_indiv_summary.json"}')
log(f'Всего: {(time.time()-t0)/60:.2f} мин')
