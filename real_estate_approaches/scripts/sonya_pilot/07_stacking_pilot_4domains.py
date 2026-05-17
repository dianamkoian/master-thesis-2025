"""
4-domain cross-domain stacking pilot (Sonya + Karina + Diana + Albina).

Final ensemble: adds Albina's headline model on top of the 3-domain pilot.

Inputs (all on team_test_idx.npy, n=58,410):
  team_runs/splits/y_test.npy
  team_runs/proba/test_proba_fusion_team.npy   - Sonya, Fusion (fintech)
  team_runs/proba/test_proba_karina_team.npy   - Karina (mobile/ads)
  team_runs/proba/test_proba_diana_team.npy    - Diana (real estate)
  team_runs/proba/test_proba_albina_team.npy   - Albina (uploaded 2026-05-11)

Outputs:
  team_runs/results/stacking_pilot_4domains.json
  team_runs/results/stacking_pilot_4domains.md (written separately)

Same methodology as the 2-domain and 3-domain pilots: sanity checks,
per-model metrics, pairwise correlations, top-K overlap (now 4-way),
naive linear blends, paired bootstrap 95% CI (B=1000, seed=42).

Reproducibility:
  python 07_stacking_pilot_4domains.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
B = 1000

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
y      = np.load(ROOT / "splits/y_test.npy").astype(int)
fusion = np.load(ROOT / "proba/test_proba_fusion_team.npy").astype(np.float64)
karina = np.load(ROOT / "proba/test_proba_karina_team.npy").astype(np.float64)
diana  = np.load(ROOT / "proba/test_proba_diana_team.npy").astype(np.float64)
albina = np.load(ROOT / "proba/test_proba_albina_team.npy").astype(np.float64)

n = len(y)
n_pos = int(y.sum())
print(f"n_test = {n}, positives = {n_pos} ({y.mean():.4%})")

for name, p in [("fusion", fusion), ("karina", karina), ("diana", diana), ("albina", albina)]:
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


def rank_norm(p):
    return np.argsort(np.argsort(p)) / (n - 1)


# ---------------------------------------------------------------------------
# 3. Per-model point estimates
# ---------------------------------------------------------------------------
print("\n=== Per-domain metrics on team test ===")
per_model = {}
models = [
    ("fusion (Sonya, fintech)",    fusion),
    ("karina (mobile/ads)",        karina),
    ("diana (real estate)",        diana),
    ("albina",                     albina),
]
for name, p in models:
    m = metrics(y, p)
    per_model[name] = m
    print(f"  {name:28s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------
print("\n=== Pairwise correlations (Pearson / Spearman) ===")
names = ["fusion", "karina", "diana", "albina"]
arrs  = {"fusion": fusion, "karina": karina, "diana": diana, "albina": albina}
corr_matrix = {n_: {} for n_ in names}
for ni in names:
    for nj in names:
        pear, _ = pearsonr(arrs[ni], arrs[nj])
        spear, _ = spearmanr(arrs[ni], arrs[nj])
        corr_matrix[ni][nj] = dict(pearson=float(pear), spearman=float(spear))
print("            " + "  ".join(f"{n:>16s}" for n in names))
for ni in names:
    row = [f"P={corr_matrix[ni][nj]['pearson']:.3f}/S={corr_matrix[ni][nj]['spearman']:.3f}"
           for nj in names]
    print(f"  {ni:8s}  " + "  ".join(f"{c:>16s}" for c in row))

# ---------------------------------------------------------------------------
# 5. Top-K coverage
# ---------------------------------------------------------------------------
print("\n=== Top-K counterfeit catches & union coverage ===")
overlap = {}
for K in [500, 1000, 2000, n_pos]:
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
        # Marginal contribution of each model: items they uniquely add to union of the OTHER three
        marginal_unique={
            ni: len(caught[ni] - set().union(*(caught[nj] for nj in names if nj != ni)))
            for ni in names
        },
    )
    print(f"  K={K}: union={len(union)} (+{(len(union)/max(len(fusion_caught),1)-1)*100:.1f}% over fusion)")
    print(f"    per-model catches: " + ", ".join(f"{ni}={len(caught[ni])}" for ni in names))
    print(f"    marginal unique:   " + ", ".join(f"{ni}={overlap[f'top_{K}']['marginal_unique'][ni]}" for ni in names))

# ---------------------------------------------------------------------------
# 6. Naive blends
# ---------------------------------------------------------------------------
print("\n=== Naive blends (point estimates) ===")
blends = {
    # singletons
    "fusion_alone":               fusion,
    "karina_alone":               karina,
    "diana_alone":                diana,
    "albina_alone":               albina,
    # baseline blends from previous pilots
    "blend_07f_03k":              0.7 * fusion + 0.3 * karina,
    "blend_06f_02k_02d":          0.6 * fusion + 0.2 * karina + 0.2 * diana,
    "uniform_3way_fkd":           (fusion + karina + diana) / 3.0,
    # 4-way
    "uniform_4way":               (fusion + karina + diana + albina) / 4.0,
    "fusion_heavy_4way":          0.4 * fusion + 0.2 * karina + 0.2 * diana + 0.2 * albina,
    "fusion_diana_heavy_4way":    0.35 * fusion + 0.15 * karina + 0.30 * diana + 0.20 * albina,
    "rank_avg_4way":              (rank_norm(fusion) + rank_norm(karina) + rank_norm(diana) + rank_norm(albina)) / 4.0,
    # pair / triple with Albina to check her marginal contribution
    "pair_fusion_albina":         0.5 * fusion + 0.5 * albina,
    "triple_fkd":                 (fusion + karina + diana) / 3.0,
    "triple_fka":                 (fusion + karina + albina) / 3.0,
    "triple_fda":                 (fusion + diana  + albina) / 3.0,
    "triple_kda":                 (karina + diana + albina) / 3.0,
}
blend_metrics = {name: metrics(y, p) for name, p in blends.items()}
for name, m in blend_metrics.items():
    print(f"  {name:28s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 7. Paired bootstrap 95% CI vs fusion baseline
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
# 8. Best blend
# ---------------------------------------------------------------------------
singletons = {"fusion_alone", "karina_alone", "diana_alone", "albina_alone"}
candidates = [n_ for n_ in blends if n_ not in singletons]
best_name = max(candidates, key=lambda n_: blend_metrics[n_]["pr"])
print(f"\n=== Best blend by PR-AUC: {best_name} ===")
print(f"  PR-AUC = {blend_metrics[best_name]['pr']:.4f}")
print(f"  ROC    = {blend_metrics[best_name]['roc']:.4f}")
print(f"  R@P90  = {blend_metrics[best_name]['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 9. Save
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
out_path = ROOT / "results/stacking_pilot_4domains.json"
out_path.write_text(json.dumps(output, indent=2, default=float))
print(f"\nSaved: {out_path}")
