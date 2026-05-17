"""
3-domain cross-domain stacking pilot (Sonya + Karina + Diana).

Adds Diana's M2 Multimodal CatBoost (real-estate domain) to the 2-domain pilot.
Diana's reported metrics: ROC=0.9561, PR=0.7175, R@P>=0.9=0.0099 — verified on
our team_test (numerically identical, MD5 differs due to resave but content
matches: see notes in stacking_pilot_3domains.md).

Inputs (all on team_test_idx.npy, n=58,410):
  team_runs/splits/y_test.npy
  team_runs/proba/test_proba_fusion_team.npy   - Sonya, Fusion (CLIP+e5+tabular)
  team_runs/proba/test_proba_karina_team.npy   - Karina, CatBoost+CLIP+TFIDF+typosquat
  team_runs/proba/test_proba_diana_team.npy    - Diana, M2 Multimodal CatBoost
                                                  (CLIP-512 + TF-IDF SVD-50 +
                                                   38 tabular + K-means structural)

Outputs:
  team_runs/results/stacking_pilot_3domains.json
  team_runs/results/stacking_pilot_3domains.md (written by hand from JSON)

What this script does:
  1. Sanity-checks the inputs (shapes, ranges, NaNs)
  2. Reports per-model metrics on team_test (ROC-AUC, PR-AUC, Recall@P>=0.9)
  3. Computes Pearson and Spearman correlations between all 3 probas
  4. Runs top-K counterfeit-overlap analysis pairwise and 3-way
  5. Evaluates naive blends: all pairs (50/50) + 3-way uniform + weighted variants
  6. Computes paired bootstrap 95% CI for each blend vs Sonya-fusion baseline

What this script does NOT do (intentional, documented as future work):
  A trained meta-classifier (LR/LGBM over probas) requires per-domain probas
  on team_val_idx (n=69,335) to avoid test-set leakage. Val-probas have not
  been delivered; the team decided to ship naive linear blending with bootstrap
  CI as the final method.

Reproducibility:
  Single seed = 42. Run with:  python 06_stacking_pilot_3domains.py
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
karina = np.load(ROOT / "proba/test_proba_karina_team.npy").astype(np.float64)
diana  = np.load(ROOT / "proba/test_proba_diana_team.npy").astype(np.float64)

n = len(y)
n_pos = int(y.sum())
print(f"n_test = {n}, positives = {n_pos} ({y.mean():.4%})")

# Sanity checks
for name, p in [("fusion", fusion), ("karina", karina), ("diana", diana)]:
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
for name, p in [("fusion (Sonya)", fusion),
                ("karina (mobile/ads)", karina),
                ("diana (real estate)", diana)]:
    m = metrics(y, p)
    per_model[name] = m
    print(f"  {name:22s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
print("\n=== Pairwise correlations (Pearson / Spearman) ===")
names = ["fusion", "karina", "diana"]
arrs  = [fusion, karina, diana]
corr_matrix = {n_: {} for n_ in names}
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        pear, _ = pearsonr(arrs[i], arrs[j])
        spear, _ = spearmanr(arrs[i], arrs[j])
        corr_matrix[ni][nj] = dict(pearson=float(pear), spearman=float(spear))
print("            " + "  ".join(f"{n:>16s}" for n in names))
for ni in names:
    row = [f"P={corr_matrix[ni][nj]['pearson']:.3f}/S={corr_matrix[ni][nj]['spearman']:.3f}"
           for nj in names]
    print(f"  {ni:8s}  " + "  ".join(f"{c:>16s}" for c in row))

# ---------------------------------------------------------------------------
# 5. Top-K counterfeit overlap
# ---------------------------------------------------------------------------
print("\n=== Top-K counterfeit overlap (pairwise + 3-way) ===")
overlap = {}
for K in [500, 1000, 2000, n_pos]:
    f_top = np.argsort(-fusion)[:K]
    k_top = np.argsort(-karina)[:K]
    d_top = np.argsort(-diana)[:K]
    f_caught = set(f_top[y[f_top] == 1])
    k_caught = set(k_top[y[k_top] == 1])
    d_caught = set(d_top[y[d_top] == 1])
    pair_fk = f_caught & k_caught
    pair_fd = f_caught & d_caught
    pair_kd = k_caught & d_caught
    triple  = f_caught & k_caught & d_caught
    union   = f_caught | k_caught | d_caught
    only_f  = f_caught - k_caught - d_caught
    only_k  = k_caught - f_caught - d_caught
    only_d  = d_caught - f_caught - k_caught
    overlap[f"top_{K}"] = dict(
        K=K,
        fusion_caught=len(f_caught),
        karina_caught=len(k_caught),
        diana_caught=len(d_caught),
        all_three=len(triple),
        fusion_karina=len(pair_fk),
        fusion_diana=len(pair_fd),
        karina_diana=len(pair_kd),
        only_fusion=len(only_f),
        only_karina=len(only_k),
        only_diana=len(only_d),
        union=len(union),
        union_recall=len(union) / n_pos,
        union_gain_over_fusion_pct=(len(union) / max(len(f_caught), 1) - 1.0) * 100,
    )
    print(f"  K={K}: f={len(f_caught)}, k={len(k_caught)}, d={len(d_caught)}, "
          f"all3={len(triple)}, union={len(union)} "
          f"(+{(len(union)/max(len(f_caught),1)-1)*100:.1f}% over fusion alone)")

# ---------------------------------------------------------------------------
# 6. Naive blends: point estimates
# ---------------------------------------------------------------------------
print("\n=== Naive blends (point estimates) ===")

# Helper: rank-average
def rank_norm(p):
    return np.argsort(np.argsort(p)) / (n - 1)

blends = {
    # baselines
    "fusion_alone":          fusion,
    "karina_alone":          karina,
    "diana_alone":           diana,
    # pairs (50/50)
    "pair_fusion_karina":    0.5 * fusion + 0.5 * karina,
    "pair_fusion_diana":     0.5 * fusion + 0.5 * diana,
    "pair_karina_diana":     0.5 * karina + 0.5 * diana,
    # best 2-domain blend from previous pilot
    "blend_07f_03k":         0.7 * fusion + 0.3 * karina,
    # 3-way uniform
    "uniform_3way":          (fusion + karina + diana) / 3.0,
    # weighted variants (fusion-heavy is best in 2-domain pilot)
    "w_05f_025k_025d":       0.5 * fusion + 0.25 * karina + 0.25 * diana,
    "w_04f_03k_03d":         0.4 * fusion + 0.3 * karina + 0.3 * diana,
    "w_06f_02k_02d":         0.6 * fusion + 0.2 * karina + 0.2 * diana,
    # rank-average (scale-free, robust to calibration mismatch)
    "rank_avg_3way":         (rank_norm(fusion) + rank_norm(karina) + rank_norm(diana)) / 3.0,
}
blend_metrics = {name: metrics(y, p) for name, p in blends.items()}
for name, m in blend_metrics.items():
    print(f"  {name:24s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 7. Paired bootstrap 95% CI vs fusion_alone baseline
# ---------------------------------------------------------------------------
print(f"\n=== Paired bootstrap (B={B}, seed={SEED}), baseline = fusion_alone ===")
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
# 8. Pick best blend (highest PR-AUC point estimate, excluding singletons)
# ---------------------------------------------------------------------------
candidate_names = [n_ for n_ in blends if n_ not in ("fusion_alone", "karina_alone", "diana_alone")]
best_name = max(candidate_names, key=lambda n_: blend_metrics[n_]["pr"])
print(f"\n=== Best blend by PR-AUC: {best_name} ===")
print(f"  PR-AUC = {blend_metrics[best_name]['pr']:.4f}")
print(f"  ROC    = {blend_metrics[best_name]['roc']:.4f}")
print(f"  R@P90  = {blend_metrics[best_name]['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 9. Save numeric results
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
    "best_blend_by_pr": best_name,
}
out_path = ROOT / "results/stacking_pilot_3domains.json"
out_path.write_text(json.dumps(output, indent=2, default=float))
print(f"\nSaved: {out_path}")
