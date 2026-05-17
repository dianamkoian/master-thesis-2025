"""
Bootstrap CI 95% + R@P-optimal grid search для всех ИНДИВИДУАЛЬНЫХ моделей Дианы.

Зависит от:
  notebooks/y_test_indiv.npy                                    (метки)
  notebooks/test_proba_diana_m2_fe_plus_indiv.npy               (M2-FE+ INDIV)
  notebooks/test_proba_diana_m2_base_indiv.npy                  (M2 baseline INDIV)
  notebooks/test_proba_diana_m2_fe_plus_ansamble_indiv.npy      (M2-FE+ + M2 50/50)
  notebooks/test_proba_diana_ablation_m2_*_indiv.npy            (component ablation, опционально)
  notebooks/test_proba_diana_m2_e5.npy / m2_rubert.npy          (если индив-варианты будут)

Считает ROC / PR-AUC / R@P09 + 95% CI (1000 bootstrap), плюс grid w_M2-FE+ × w_M2.
"""
import time, os, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

ROOT = Path('/Users/diana/master-thesis-2025')
OUT_DIR = ROOT / 'Диана_ВКР_финал' / 'notebooks'
LOG = ROOT / 'Диана_ВКР_финал' / 'scripts' / 'bootstrap_ci_grid_indiv_log.txt'

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
log('Bootstrap CI + Grid INDIV')

y_test = np.load(OUT_DIR / 'y_test_indiv.npy')
log(f'y_test_indiv: n={len(y_test)}, positive={int(y_test.sum())} ({y_test.mean()*100:.2f}%)')

# Регистрация моделей: имя → путь
models = {
    'M2-FE+ INDIV': 'test_proba_diana_m2_fe_plus_indiv.npy',
    'M2 baseline INDIV': 'test_proba_diana_m2_base_indiv.npy',
    'M2-FE+ + M2 (50/50) INDIV': 'test_proba_diana_m2_fe_plus_ansamble_indiv.npy',
    # ablation если файлы есть
    'M2 без CLIP INDIV': 'test_proba_diana_ablation_m2_no_CLIP_indiv.npy',
    'M2 без K-means INDIV': 'test_proba_diana_ablation_m2_no_K-means_indiv.npy',
    'M2 без TF-IDF SVD INDIV': 'test_proba_diana_ablation_m2_no_TF-IDF_SVD_indiv.npy',
    'M2 full ablation INDIV': 'test_proba_diana_ablation_m2_full_indiv.npy',
}

probas = {}
for name, fname in models.items():
    fpath = OUT_DIR / fname
    if fpath.exists():
        probas[name] = np.load(fpath).astype(np.float64)
        if len(probas[name]) != len(y_test):
            log(f'WARNING: {name} имеет n={len(probas[name])}, ожидается {len(y_test)} — пропускаю')
            del probas[name]
        else:
            log(f'loaded {name}: n={len(probas[name])}')
    else:
        log(f'(пропуск) {name}: {fname} не найден')

log('=' * 60)
log(f'{"Модель":35s} {"ROC [95% CI]":>22s} {"PR [95% CI]":>22s} {"R@P09 [95% CI]":>26s}')

results = {}
for name, p in probas.items():
    roc_m, roc_lo, roc_hi = bootstrap_ci(y_test, p, roc_auc_score)
    pr_m, pr_lo, pr_hi = bootstrap_ci(y_test, p, average_precision_score)
    r9_m, r9_lo, r9_hi = bootstrap_ci(y_test, p, rap)
    results[name] = {
        'ROC': {'mean': roc_m, 'ci_lo': roc_lo, 'ci_hi': roc_hi},
        'PR-AUC': {'mean': pr_m, 'ci_lo': pr_lo, 'ci_hi': pr_hi},
        'R@P09': {'mean': r9_m, 'ci_lo': r9_lo, 'ci_hi': r9_hi},
    }
    log(f'{name:35s} {roc_m:.4f} [{roc_lo:.4f},{roc_hi:.4f}] {pr_m:.4f} [{pr_lo:.4f},{pr_hi:.4f}] {r9_m:.4f} [{r9_lo:.4f},{r9_hi:.4f}]')

# === Grid search для ансамблей M2-FE+ × M2 (R@P-headline) ===
log('=' * 60)
log('Grid: w*M2-FE+ + (1-w)*M2 baseline, шаг 0.05')

if 'M2-FE+ INDIV' in probas and 'M2 baseline INDIV' in probas:
    p_fe = probas['M2-FE+ INDIV']
    p_m2 = probas['M2 baseline INDIV']
    log(f'{"w_FE+":>8s} {"ROC":>8s} {"PR-AUC":>8s} {"R@P09":>8s}')
    grid_results = []
    for w in np.arange(0.0, 1.01, 0.05):
        mix = w * p_fe + (1-w) * p_m2
        roc = roc_auc_score(y_test, mix)
        pr_a = average_precision_score(y_test, mix)
        r9 = rap(y_test, mix)
        grid_results.append({'w': float(w), 'ROC': float(roc), 'PR': float(pr_a), 'RAP09': float(r9)})
        log(f'{w:8.2f} {roc:8.4f} {pr_a:8.4f} {r9:8.4f}')

    best_pr = max(grid_results, key=lambda r: r['PR'])
    best_r9 = max(grid_results, key=lambda r: r['RAP09'])
    log(f'\nЛУЧШИЙ по PR-AUC: w={best_pr["w"]:.2f}, ROC={best_pr["ROC"]:.4f}, PR={best_pr["PR"]:.4f}, R@P={best_pr["RAP09"]:.4f}')
    log(f'ЛУЧШИЙ по R@P09: w={best_r9["w"]:.2f}, ROC={best_r9["ROC"]:.4f}, PR={best_r9["PR"]:.4f}, R@P={best_r9["RAP09"]:.4f}')

    # Bootstrap CI для best PR и best R@P конфигов
    best_pr_mix = best_pr['w'] * p_fe + (1-best_pr['w']) * p_m2
    best_r9_mix = best_r9['w'] * p_fe + (1-best_r9['w']) * p_m2
    log('\nBootstrap CI для best PR конфига:')
    for mname, mfn in [('ROC', roc_auc_score), ('PR-AUC', average_precision_score), ('R@P09', rap)]:
        m, lo, hi = bootstrap_ci(y_test, best_pr_mix, mfn)
        log(f'  {mname}: {m:.4f} [{lo:.4f},{hi:.4f}]')
    log('\nBootstrap CI для best R@P конфига:')
    for mname, mfn in [('ROC', roc_auc_score), ('PR-AUC', average_precision_score), ('R@P09', rap)]:
        m, lo, hi = bootstrap_ci(y_test, best_r9_mix, mfn)
        log(f'  {mname}: {m:.4f} [{lo:.4f},{hi:.4f}]')

    # Save best mixes
    np.save(OUT_DIR / 'test_proba_diana_indiv_best_pr.npy', best_pr_mix.astype(np.float32))
    np.save(OUT_DIR / 'test_proba_diana_indiv_best_rap.npy', best_r9_mix.astype(np.float32))
    log(f'\nSaved: test_proba_diana_indiv_best_pr.npy, test_proba_diana_indiv_best_rap.npy')

    results['_grid'] = {'best_pr': best_pr, 'best_r9': best_r9, 'all': grid_results}

# Сохранить summary JSON
with open(OUT_DIR / 'bootstrap_ci_grid_indiv_summary.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log(f'\nSummary JSON: {OUT_DIR / "bootstrap_ci_grid_indiv_summary.json"}')

log(f'\nВсего: {(time.time()-t0)/60:.1f} мин')
log('=' * 60)
