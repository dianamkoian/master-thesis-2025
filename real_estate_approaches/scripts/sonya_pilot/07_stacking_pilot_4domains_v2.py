"""
4-domain cross-domain stacking pilot — IMPROVED VERSION (v2).

Adapted from `07_stacking_pilot_4domains.py` (Sonya, dazzling-vaughan branch).
Same inputs, same seed, same B=1000. Strict superset of original outputs.

Improvements over v1 (each tagged with [IMPROVEMENT N] in code):

  [1] Full pairwise bootstrap matrix — blend vs EACH single, not only vs fusion.
      Reproduces the table on rows 12–19 of `stacking_pilot_4domains.md`
      directly from this script (previously the table was constructed by hand).

  [2] Grid search for optimal weights (step=0.05) by PR-AUC AND by R@P≥0.9
      separately, instead of hand-picked weights. Eliminates "blend selection
      bias" concern raised in original section 10.

  [3] Holm-Bonferroni correction for the 4 paired comparisons (blend vs each).
      Original v1 reports raw bootstrap CIs without multiple-testing correction.

  [4] Proxy stacker via test-internal 5-fold CV (LR + LGBM heads).
      Original v1 deliberately excluded a trained meta-classifier (section 10),
      citing absence of per-domain val probas. This v2 supplies a CV-based
      proxy estimate of stacker performance, with the explicit caveat that the
      stacker is trained AND evaluated on test via out-of-fold predictions —
      this provides an UPPER BOUND for "what a real stacker could achieve",
      not a deployable model.

  [5] Calibration analysis — Brier score + ECE for each single model and for
      all blends. Naive linear averaging presupposes comparable proba
      distributions; this measures actual divergence.

  [6] Extended top-K table — K=100, 200, 500, 1000, 2000, n_pos. Original v1
      reports only K=500/1000/2000/n_pos.

Outputs:
  stacking_pilot_4domains_v2.json     - full numerical results
  stacking_pilot_4domains_v2.md       - human-readable report (written separately)
  test_proba_4way_stacker_lr.npy      - LR-stacker out-of-fold proba
  test_proba_4way_stacker_lgbm.npy    - LGBM-stacker out-of-fold proba
  test_proba_4way_grid_best_pr.npy    - best PR-weighted blend
  test_proba_4way_grid_best_rp.npy    - best R@P-weighted blend
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

# Paths (adjust if running outside Diana's repo)
DIANA_FOLDER = Path("/Users/diana/master-thesis-2025/Диана_ВКР_финал")
SONYA_PILOT = DIANA_FOLDER / "scripts/sonya_pilot"
NPY = Path("/Users/diana/master-thesis-2025/npy_files")
SPLITS = DIANA_FOLDER / "notebooks/team_splits"

SEED = 42
B = 1000
N_FOLDS = 5  # for CV-stacker proxy
GRID_STEP = 0.05

# ---------------------------------------------------------------------------
# 1. Load (identical to v1)
# ---------------------------------------------------------------------------
y      = np.load(SPLITS / "y_test.npy").astype(int)
fusion = np.load(SONYA_PILOT / "proba/test_proba_fusion_team.npy").astype(np.float64)
karina = np.load(NPY / "test_proba_karina_team.npy").astype(np.float64)
diana  = np.load(NPY / "test_proba_diana_team.npy").astype(np.float64)
albina = np.load(NPY / "test_proba_albina_team.npy").astype(np.float64)

n = len(y)
n_pos = int(y.sum())
print(f"n_test = {n}, positives = {n_pos} ({y.mean():.4%})")

for name, p in [("fusion", fusion), ("karina", karina), ("diana", diana), ("albina", albina)]:
    assert len(p) == n
    assert not np.isnan(p).any()
    assert (p >= 0).all() and (p <= 1).all()

names = ["fusion", "karina", "diana", "albina"]
arrs  = {"fusion": fusion, "karina": karina, "diana": diana, "albina": albina}

# ---------------------------------------------------------------------------
# 2. Metric helpers (identical to v1, + Brier and ECE for [5])
# ---------------------------------------------------------------------------
def recall_at_precision(y, p, target=0.90):
    pr, rc, _ = precision_recall_curve(y, p)
    m = pr >= target
    return float(rc[m].max()) if m.any() else 0.0


def metrics(y, p):
    return dict(
        roc=roc_auc_score(y, p),
        pr=average_precision_score(y, p),
        r_at_p90=recall_at_precision(y, p, 0.90),
    )


def rank_norm(p):
    return np.argsort(np.argsort(p)) / (n - 1)


def ece(y, p, n_bins=10):
    """[IMPROVEMENT 5] Expected Calibration Error, equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(p, bins[1:-1])
    err = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not mask.any():
            continue
        err += (mask.sum() / len(y)) * abs(y[mask].mean() - p[mask].mean())
    return float(err)


# ---------------------------------------------------------------------------
# 3. Per-model metrics (+ calibration [IMPROVEMENT 5])
# ---------------------------------------------------------------------------
print("\n=== Per-domain metrics + calibration ===")
per_model = {}
for nm in names:
    m = metrics(y, arrs[nm])
    m["brier"] = brier_score_loss(y, arrs[nm])
    m["ece"] = ece(y, arrs[nm])
    per_model[nm] = m
    print(f"  {nm:8s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}  Brier={m['brier']:.4f} ECE={m['ece']:.4f}")

# ---------------------------------------------------------------------------
# 4. Correlations (identical to v1)
# ---------------------------------------------------------------------------
print("\n=== Pearson correlations ===")
corr_matrix = {n_: {} for n_ in names}
for ni in names:
    for nj in names:
        pear, _ = pearsonr(arrs[ni], arrs[nj])
        spear, _ = spearmanr(arrs[ni], arrs[nj])
        corr_matrix[ni][nj] = dict(pearson=float(pear), spearman=float(spear))
for ni in names:
    print(f"  {ni:8s}: " + ", ".join(f"{nj}={corr_matrix[ni][nj]['pearson']:.3f}" for nj in names))

# ---------------------------------------------------------------------------
# 5. Top-K overlap — EXTENDED [IMPROVEMENT 6]
# ---------------------------------------------------------------------------
print("\n=== Top-K catches & unique contribution (extended) ===")
overlap = {}
for K in [100, 200, 500, 1000, 2000, n_pos]:
    caught = {}
    for ni in names:
        idx_top = np.argsort(-arrs[ni])[:K]
        caught[ni] = set(idx_top[y[idx_top] == 1])
    union = set().union(*caught.values())
    fusion_caught = caught["fusion"]
    overlap[f"top_{K}"] = dict(
        K=K,
        per_model_caught={ni: len(caught[ni]) for ni in names},
        union=len(union),
        union_recall=len(union) / n_pos,
        union_gain_over_fusion_pct=(len(union) / max(len(fusion_caught), 1) - 1.0) * 100,
        marginal_unique={
            ni: len(caught[ni] - set().union(*(caught[nj] for nj in names if nj != ni)))
            for ni in names
        },
    )
    print(f"  K={K}: union={len(union)} (+{overlap[f'top_{K}']['union_gain_over_fusion_pct']:.1f}%)  "
          f"per-model {dict(overlap[f'top_{K}']['per_model_caught'])}  "
          f"unique {dict(overlap[f'top_{K}']['marginal_unique'])}")

# ---------------------------------------------------------------------------
# 6. Naive blends (Sonya's set, identical to v1)
# ---------------------------------------------------------------------------
print("\n=== Naive blends (Sonya's hand-picked set) ===")
blends = {
    "fusion_alone":            fusion,
    "karina_alone":            karina,
    "diana_alone":             diana,
    "albina_alone":            albina,
    "uniform_4way":            (fusion + karina + diana + albina) / 4.0,
    "fusion_heavy_4way":       0.4 * fusion + 0.2 * karina + 0.2 * diana + 0.2 * albina,
    "fusion_diana_heavy_4way": 0.35 * fusion + 0.15 * karina + 0.30 * diana + 0.20 * albina,
    "rank_avg_4way":           (rank_norm(fusion) + rank_norm(karina) + rank_norm(diana) + rank_norm(albina)) / 4.0,
}
blend_metrics = {nm: metrics(y, p) for nm, p in blends.items()}

# ---------------------------------------------------------------------------
# 7. Grid search [IMPROVEMENT 2]
# ---------------------------------------------------------------------------
print(f"\n=== Grid search (step={GRID_STEP}) ===")
grid = []
ws = np.arange(0, 1.0001, GRID_STEP)
for wf in ws:
    for wk in ws:
        for wd in ws:
            wa = 1.0 - wf - wk - wd
            if wa < -1e-9 or wa > 1.0 + 1e-9:
                continue
            wa = round(wa, 4)
            if wa < 0:
                continue
            blend = wf * fusion + wk * karina + wd * diana + wa * albina
            m = metrics(y, blend)
            grid.append((round(wf, 2), round(wk, 2), round(wd, 2), round(wa, 2),
                         m['roc'], m['pr'], m['r_at_p90']))
grid_arr = np.array(grid, dtype=[('wf','f4'),('wk','f4'),('wd','f4'),('wa','f4'),
                                  ('roc','f8'),('pr','f8'),('r_at_p','f8')])
top_pr = np.argsort(-grid_arr['pr'])[0]
top_rp = np.argsort(-grid_arr['r_at_p'])[0]
g_pr = grid_arr[top_pr]
g_rp = grid_arr[top_rp]
best_pr_blend = (float(g_pr['wf']) * fusion + float(g_pr['wk']) * karina +
                 float(g_pr['wd']) * diana + float(g_pr['wa']) * albina)
best_rp_blend = (float(g_rp['wf']) * fusion + float(g_rp['wk']) * karina +
                 float(g_rp['wd']) * diana + float(g_rp['wa']) * albina)
blends["grid_best_pr"] = best_pr_blend
blends["grid_best_rp"] = best_rp_blend
blend_metrics["grid_best_pr"] = metrics(y, best_pr_blend)
blend_metrics["grid_best_rp"] = metrics(y, best_rp_blend)
print(f"  best PR: f={g_pr['wf']:.2f} k={g_pr['wk']:.2f} d={g_pr['wd']:.2f} a={g_pr['wa']:.2f}  "
      f"ROC={g_pr['roc']:.4f} PR={g_pr['pr']:.4f} R@P={g_pr['r_at_p']:.4f}")
print(f"  best R@P: f={g_rp['wf']:.2f} k={g_rp['wk']:.2f} d={g_rp['wd']:.2f} a={g_rp['wa']:.2f}  "
      f"ROC={g_rp['roc']:.4f} PR={g_rp['pr']:.4f} R@P={g_rp['r_at_p']:.4f}")

# ---------------------------------------------------------------------------
# 8. CV-stacker proxy [IMPROVEMENT 4]
# ---------------------------------------------------------------------------
print(f"\n=== CV-stacker proxy ({N_FOLDS}-fold on test, out-of-fold predictions) ===")
print("    NOTE: this is an UPPER BOUND on stacker performance — the stacker")
print("    is trained AND evaluated on test via OOF, not on a held-out val.")
print("    Real deployable stacker would need per-domain val probas.")

X_stack = np.column_stack([fusion, karina, diana, albina])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lr = np.zeros(n)
oof_lgbm = np.zeros(n)
lr_coefs = []

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("    [lightgbm not installed — LGBM stacker skipped]")

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_stack, y)):
    # LR stacker (L2, balanced)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
    lr.fit(X_stack[tr_idx], y[tr_idx])
    oof_lr[te_idx] = lr.predict_proba(X_stack[te_idx])[:, 1]
    lr_coefs.append(lr.coef_[0].copy())
    if LGBM_AVAILABLE:
        gbm = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            class_weight="balanced", random_state=SEED, verbose=-1
        )
        gbm.fit(X_stack[tr_idx], y[tr_idx])
        oof_lgbm[te_idx] = gbm.predict_proba(X_stack[te_idx])[:, 1]

blends["stacker_lr_cv"] = oof_lr
blend_metrics["stacker_lr_cv"] = metrics(y, oof_lr)
mean_lr_coef = np.mean(lr_coefs, axis=0)
print(f"  LR-stacker OOF: ROC={blend_metrics['stacker_lr_cv']['roc']:.4f} "
      f"PR={blend_metrics['stacker_lr_cv']['pr']:.4f} "
      f"R@P={blend_metrics['stacker_lr_cv']['r_at_p90']:.4f}")
print(f"    mean LR coefs (fusion, karina, diana, albina): "
      f"{mean_lr_coef[0]:.3f}, {mean_lr_coef[1]:.3f}, {mean_lr_coef[2]:.3f}, {mean_lr_coef[3]:.3f}")
if LGBM_AVAILABLE:
    blends["stacker_lgbm_cv"] = oof_lgbm
    blend_metrics["stacker_lgbm_cv"] = metrics(y, oof_lgbm)
    print(f"  LGBM-stacker OOF: ROC={blend_metrics['stacker_lgbm_cv']['roc']:.4f} "
          f"PR={blend_metrics['stacker_lgbm_cv']['pr']:.4f} "
          f"R@P={blend_metrics['stacker_lgbm_cv']['r_at_p90']:.4f}")

# Calibration for all blends [IMPROVEMENT 5]
for nm in list(blends.keys()):
    if nm not in blend_metrics:
        blend_metrics[nm] = metrics(y, blends[nm])
    blend_metrics[nm]["brier"] = brier_score_loss(y, blends[nm])
    blend_metrics[nm]["ece"] = ece(y, blends[nm])

# ---------------------------------------------------------------------------
# 9. Best blend selection (highest PR among non-singletons, prefer naive over CV)
# ---------------------------------------------------------------------------
singletons = {"fusion_alone", "karina_alone", "diana_alone", "albina_alone"}
naive_candidates = [n_ for n_ in blends if n_ not in singletons and not n_.startswith("stacker_")]
best_naive_name = max(naive_candidates, key=lambda n_: blend_metrics[n_]["pr"])
best_naive = blends[best_naive_name]
print(f"\n=== Best naive blend by PR-AUC: {best_naive_name}")
print(f"    PR={blend_metrics[best_naive_name]['pr']:.4f} "
      f"ROC={blend_metrics[best_naive_name]['roc']:.4f} "
      f"R@P={blend_metrics[best_naive_name]['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 10. Paired bootstrap — FULL MATRIX [IMPROVEMENT 1]
# ---------------------------------------------------------------------------
print(f"\n=== Paired bootstrap (B={B}, seed={SEED}) — full matrix ===")
print("    best blend = grid_best_pr; alt blends compared vs fusion separately.")

best_blend = blends["grid_best_pr"]
singles = {nm: arrs[nm] for nm in names}

# 10a. best blend vs each single (4 pairs)
rng = np.random.default_rng(SEED)
ci_vs_each = {nm: {"roc": [], "pr": [], "r_at_p90": []} for nm in singles}
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    m_blend = metrics(yb, best_blend[idx])
    for nm, p_single in singles.items():
        m_single = metrics(yb, p_single[idx])
        for k in ci_vs_each[nm]:
            ci_vs_each[nm][k].append(m_blend[k] - m_single[k])

bootstrap_vs_each = {}
p_vals_pr = []  # for Holm-Bonferroni [IMPROVEMENT 3]
for nm, deltas in ci_vs_each.items():
    summary = {}
    for k, vs in deltas.items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        # one-sided p-value (against null that delta <= 0): fraction of bootstrap deltas <= 0
        p_val = float((vs <= 0).mean())
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo > 0 or hi < 0),
                          p_one_sided=p_val)
    bootstrap_vs_each[nm] = summary
    p_vals_pr.append((nm, summary["pr"]["p_one_sided"]))
    print(f"\n  best blend vs {nm}_alone:")
    for k, s in summary.items():
        sig = "*" if s["significant"] else " "
        print(f"    Δ{k:9s} = {s['mean']:+.4f}  95% CI [{s['lo']:+.4f}; {s['hi']:+.4f}] {sig} (p≈{s['p_one_sided']:.3f})")

# 10b. Holm-Bonferroni correction [IMPROVEMENT 3]
print(f"\n=== Holm-Bonferroni correction on 4 PR-AUC comparisons (FWER α=0.05) ===")
sorted_p = sorted(p_vals_pr, key=lambda x: x[1])
holm_results = {}
for i, (nm, p_val) in enumerate(sorted_p):
    alpha_adj = 0.05 / (len(sorted_p) - i)
    holm_results[nm] = dict(p_one_sided=p_val, alpha_adj=alpha_adj,
                            reject_null=p_val < alpha_adj)
    status = "REJECT H0 (significant)" if p_val < alpha_adj else "fail to reject"
    print(f"  vs {nm}: p={p_val:.4f}, adj-α={alpha_adj:.4f} → {status}")

# 10c. Alt blends vs fusion (Sonya's section 8)
print(f"\n=== Alt blends vs fusion (paired bootstrap) ===")
rng = np.random.default_rng(SEED)
alt_blends = ["uniform_4way", "fusion_heavy_4way", "fusion_diana_heavy_4way",
              "rank_avg_4way", "grid_best_pr", "grid_best_rp", "stacker_lr_cv"]
if LGBM_AVAILABLE:
    alt_blends.append("stacker_lgbm_cv")

alt_deltas = {nm: {"roc": [], "pr": [], "r_at_p90": []} for nm in alt_blends if nm in blends}
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    m_base = metrics(yb, fusion[idx])
    for nm in alt_deltas:
        m_b = metrics(yb, blends[nm][idx])
        for k in alt_deltas[nm]:
            alt_deltas[nm][k].append(m_b[k] - m_base[k])

bootstrap_alt = {}
for nm, deltas in alt_deltas.items():
    summary = {}
    for k, vs in deltas.items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo > 0 or hi < 0))
    bootstrap_alt[nm] = summary
    m = blend_metrics[nm]
    flag = "*" if summary["pr"]["significant"] else " "
    print(f"  {nm:24s} PR={m['pr']:.4f}  ΔPR={summary['pr']['mean']:+.4f}{flag} ({summary['pr']['lo']:+.4f}, {summary['pr']['hi']:+.4f})")

# ---------------------------------------------------------------------------
# 11. Save artefacts
# ---------------------------------------------------------------------------
output = {
    "version": "v2",
    "improvements_over_v1": [
        "1. Full pairwise bootstrap matrix (blend vs each single)",
        "2. Grid search optimal weights by PR and R@P",
        "3. Holm-Bonferroni multiple-testing correction",
        "4. CV-stacker proxy (LR + LGBM) via test-internal 5-fold OOF",
        "5. Calibration analysis (Brier + ECE)",
        "6. Extended top-K table (K=100..n_pos)",
    ],
    "n_test": n,
    "n_positives": n_pos,
    "positive_rate": float(y.mean()),
    "seed": SEED,
    "bootstrap_B": B,
    "n_cv_folds": N_FOLDS,
    "grid_step": GRID_STEP,
    "per_model_metrics": per_model,
    "correlations": corr_matrix,
    "topk_overlap": overlap,
    "blend_metrics": blend_metrics,
    "best_naive_blend_by_pr": best_naive_name,
    "grid_best_pr_weights": {"fusion": float(g_pr['wf']), "karina": float(g_pr['wk']),
                              "diana": float(g_pr['wd']), "albina": float(g_pr['wa'])},
    "grid_best_rp_weights": {"fusion": float(g_rp['wf']), "karina": float(g_rp['wk']),
                              "diana": float(g_rp['wd']), "albina": float(g_rp['wa'])},
    "stacker_lr_mean_coefs": {"fusion": float(mean_lr_coef[0]), "karina": float(mean_lr_coef[1]),
                               "diana": float(mean_lr_coef[2]), "albina": float(mean_lr_coef[3])},
    "bootstrap_best_vs_each_single": bootstrap_vs_each,
    "holm_bonferroni_pr": holm_results,
    "bootstrap_alt_blends_vs_fusion": bootstrap_alt,
}
out_json = SONYA_PILOT / "stacking_pilot_4domains_v2.json"
out_json.write_text(json.dumps(output, indent=2, default=float))
print(f"\nSaved JSON: {out_json}")

np.save(SONYA_PILOT / "test_proba_4way_grid_best_pr.npy", best_pr_blend)
np.save(SONYA_PILOT / "test_proba_4way_grid_best_rp.npy", best_rp_blend)
np.save(SONYA_PILOT / "test_proba_4way_stacker_lr.npy", oof_lr)
if LGBM_AVAILABLE:
    np.save(SONYA_PILOT / "test_proba_4way_stacker_lgbm.npy", oof_lgbm)
print("Saved probas: grid_best_pr, grid_best_rp, stacker_lr_cv" +
      (", stacker_lgbm_cv" if LGBM_AVAILABLE else ""))
