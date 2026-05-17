"""
2-domain cross-domain stacking pilot.

Inputs (all on team_test_idx.npy, n=58,410):
  team_runs/splits/y_test.npy
  team_runs/proba/test_proba_fusion_team.npy   - Sonya, Fusion (CLIP+e5+tabular)
  team_runs/proba/test_proba_e5_team.npy       - Sonya, e5 text-only
  team_runs/proba/test_proba_e7_team.npy       - Sonya, e7 text+PCA
  team_runs/proba/test_proba_e0_team_clean.npy - Sonya, tabular-only baseline
  team_runs/proba/test_proba_karina_team.npy   - Karina, CatBoost+CLIP+TF-IDF+typosquat

Outputs:
  team_runs/results/stacking_pilot_2domains.json - all numeric results
  team_runs/results/stacking_pilot_2domains.md   - human-readable writeup

What this script does:
  1. Sanity-checks the inputs (shapes, ranges, NaNs)
  2. Reports per-model metrics on team_test (ROC-AUC, PR-AUC, Recall@P>=0.9)
  3. Computes Pearson and Spearman correlations between all 5 probas
  4. Runs top-K counterfeit-overlap analysis between Sonya-Fusion and Karina
  5. Evaluates four naive blends of Sonya-Fusion + Karina
  6. Computes paired bootstrap 95% CI on (blend - fusion_alone), B=1000

What this script does NOT do:
  A trained meta-classifier (LR/LGBM) is intentionally omitted because we have
  only test-set probas from Karina. Training a meta-classifier on the test set
  is leakage. The meta-classifier requires per-domain probas on team_val_idx
  (n=69,335), which were not delivered before the deadline.

Reproducibility:
  Single seed = 42. Run with:  python 05_stacking_pilot_2domains.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]  # team_runs/
SEED = 42
B = 1000

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
y      = np.load(ROOT / "splits/y_test.npy").astype(int)
fusion = np.load(ROOT / "proba/test_proba_fusion_team.npy").astype(np.float64)
e5     = np.load(ROOT / "proba/test_proba_e5_team.npy").astype(np.float64)
e7     = np.load(ROOT / "proba/test_proba_e7_team.npy").astype(np.float64)
e0     = np.load(ROOT / "proba/test_proba_e0_team_clean.npy").astype(np.float64)
karina = np.load(ROOT / "proba/test_proba_karina_team.npy").astype(np.float64)

n = len(y)
n_pos = int(y.sum())
print(f"n_test = {n}, positives = {n_pos} ({y.mean():.4%})")

# Sanity checks
for name, p in [("fusion", fusion), ("e5", e5), ("e7", e7), ("e0", e0), ("karina", karina)]:
    assert len(p) == n, f"{name}: length {len(p)} != {n}"
    assert not np.isnan(p).any(), f"{name}: contains NaN"
    assert (p >= 0).all() and (p <= 1).all(), f"{name}: outside [0, 1]"

# ---------------------------------------------------------------------------
# 2. Metric helpers
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


# ---------------------------------------------------------------------------
# 3. Per-model point estimates
# ---------------------------------------------------------------------------
print("\n=== Per-domain metrics on team test ===")
per_model = {}
for name, p in [("fusion (Sonya)", fusion), ("e5 (Sonya)", e5), ("e7 (Sonya)", e7),
                ("e0 (Sonya)", e0), ("karina", karina)]:
    m = metrics(y, p)
    per_model[name] = m
    print(f"  {name:18s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
print("\n=== Pairwise correlations (Pearson / Spearman) ===")
names = ["fusion", "e5", "e7", "e0", "karina"]
arrs  = [fusion, e5, e7, e0, karina]
corr_matrix = {n_: {} for n_ in names}
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        pear, _ = pearsonr(arrs[i], arrs[j])
        spear, _ = spearmanr(arrs[i], arrs[j])
        corr_matrix[ni][nj] = dict(pearson=float(pear), spearman=float(spear))
print("              " + "  ".join(f"{n:>14s}" for n in names))
for ni in names:
    row = [f"P={corr_matrix[ni][nj]['pearson']:.3f}/S={corr_matrix[ni][nj]['spearman']:.3f}"
           for nj in names]
    print(f"  {ni:10s}  " + "  ".join(f"{c:>14s}" for c in row))

# ---------------------------------------------------------------------------
# 5. Top-K counterfeit overlap (Sonya-Fusion vs Karina)
# ---------------------------------------------------------------------------
print("\n=== Top-K counterfeit overlap (fusion vs karina) ===")
overlap = {}
for K in [500, 1000, 2000, n_pos]:
    f_top = np.argsort(-fusion)[:K]
    k_top = np.argsort(-karina)[:K]
    f_caught = set(f_top[y[f_top] == 1])
    k_caught = set(k_top[y[k_top] == 1])
    both = f_caught & k_caught
    only_f = f_caught - k_caught
    only_k = k_caught - f_caught
    union = f_caught | k_caught
    overlap[f"top_{K}"] = dict(
        K=K,
        sonya_caught=len(f_caught),
        karina_caught=len(k_caught),
        both=len(both),
        only_sonya=len(only_f),
        only_karina=len(only_k),
        union=len(union),
        union_recall=len(union) / n_pos,
        sonya_precision=len(f_caught) / K,
        karina_precision=len(k_caught) / K,
        union_gain_pct=(len(union) / max(len(f_caught), 1) - 1.0) * 100,
    )
    print(f"  K={K}: sonya={len(f_caught)}, karina={len(k_caught)}, both={len(both)}, "
          f"only_sonya={len(only_f)}, only_karina={len(only_k)}, union={len(union)} "
          f"(+{(len(union)/max(len(f_caught),1)-1)*100:.1f}% over sonya)")

# ---------------------------------------------------------------------------
# 6. Naive blends: point estimates
# ---------------------------------------------------------------------------
print("\n=== Naive blends (point estimates) ===")
blends = {
    "fusion_alone":  fusion,
    "blend_07f_03k": 0.7 * fusion + 0.3 * karina,
    "blend_06f_04k": 0.6 * fusion + 0.4 * karina,
    "blend_05f_05k": 0.5 * fusion + 0.5 * karina,
    "rank_avg":      0.5 * (np.argsort(np.argsort(fusion)) / n
                          + np.argsort(np.argsort(karina)) / n),
}
blend_metrics = {name: metrics(y, p) for name, p in blends.items()}
for name, m in blend_metrics.items():
    print(f"  {name:15s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 7. Paired bootstrap 95% CI (blend - fusion_alone)
# ---------------------------------------------------------------------------
print(f"\n=== Paired bootstrap (B={B}, seed={SEED}) ===")
rng = np.random.default_rng(SEED)
bootstrap = {}
for name, p in blends.items():
    if name == "fusion_alone":
        continue
    deltas = {"roc": [], "pr": [], "r_at_p90": []}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        m_blend = metrics(yb, p[idx])
        m_base  = metrics(yb, fusion[idx])
        for k in deltas:
            deltas[k].append(m_blend[k] - m_base[k])
    summary = {}
    for k, vs in deltas.items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo > 0 or hi < 0))
    bootstrap[name] = summary
    print(f"\n  {name} vs fusion_alone:")
    for k, s in summary.items():
        sig = "*" if s["significant"] else " "
        print(f"    Δ{k:9s} = {s['mean']:+.4f}  95% CI [{s['lo']:+.4f}; {s['hi']:+.4f}] {sig}")

# ---------------------------------------------------------------------------
# 8. Save numeric results for thesis
# ---------------------------------------------------------------------------
output = {
    "n_test": n,
    "n_positives": n_pos,
    "positive_rate": float(y.mean()),
    "seed": SEED,
    "bootstrap_B": B,
    "per_model_metrics": per_model,
    "correlations": corr_matrix,
    "topk_overlap": overlap,
    "blend_metrics": blend_metrics,
    "bootstrap_deltas_vs_fusion": bootstrap,
}
out_path = ROOT / "results/stacking_pilot_2domains.json"
out_path.write_text(json.dumps(output, indent=2, default=float))
print(f"\nSaved: {out_path}")
