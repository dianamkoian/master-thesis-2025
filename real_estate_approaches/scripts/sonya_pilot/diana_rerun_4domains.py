"""
4-domain cross-domain stacking pilot — RERUN with Diana's NEW M2-FE+ proba.

Replaces Diana's old proba (M2 mainline clean, PR=0.7175, R@P=0.0099)
with the new M2-FE+ standalone proba (PR=0.7375, R@P=0.1154) from
/Users/diana/master-thesis-2025/npy_files/test_proba_diana_team.npy.

Reproduces Sonya's pilot methodology exactly, but extends the paired bootstrap
to compare the best blend against EVERY single model (not just fusion).

Outputs:
  diana_rerun_4domains.json  - full numerical results
  diana_rerun_4domains.md    - human-readable report (written separately)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_auc_score)

# Paths
DIANA_FOLDER = Path("/Users/diana/master-thesis-2025/Диана_ВКР_финал")
SONYA_PILOT = DIANA_FOLDER / "scripts/sonya_pilot"
NPY = Path("/Users/diana/master-thesis-2025/npy_files")
SPLITS = DIANA_FOLDER / "notebooks/team_splits"

SEED = 42
B = 1000

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
y      = np.load(SPLITS / "y_test.npy").astype(int)
fusion = np.load(SONYA_PILOT / "proba/test_proba_fusion_team.npy").astype(np.float64)
karina = np.load(NPY / "test_proba_karina_team.npy").astype(np.float64)
diana  = np.load(NPY / "test_proba_diana_team.npy").astype(np.float64)  # M2-FE+
albina = np.load(NPY / "test_proba_albina_team.npy").astype(np.float64)

n = len(y)
n_pos = int(y.sum())
print(f"n_test = {n}, positives = {n_pos} ({y.mean():.4%})")

for name, p in [("fusion", fusion), ("karina", karina), ("diana", diana), ("albina", albina)]:
    assert len(p) == n
    assert not np.isnan(p).any()
    assert (p >= 0).all() and (p <= 1).all()

# ---------------------------------------------------------------------------
# 2. Metrics
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
# 3. Per-model metrics
# ---------------------------------------------------------------------------
print("\n=== Per-domain metrics on team test (NEW Diana proba: M2-FE+) ===")
per_model = {}
models = [
    ("fusion (Sonya, fintech)",       fusion),
    ("karina (mobile/ads)",           karina),
    ("diana (real estate, M2-FE+)",   diana),
    ("albina",                        albina),
]
for name, p in models:
    m = metrics(y, p)
    per_model[name] = m
    print(f"  {name:34s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

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
for ni in names:
    print(f"  {ni:8s} | " + "  ".join(f"{nj}: {corr_matrix[ni][nj]['pearson']:.3f}" for nj in names))

# ---------------------------------------------------------------------------
# 5. Top-K overlap
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
        marginal_unique={
            ni: len(caught[ni] - set().union(*(caught[nj] for nj in names if nj != ni)))
            for ni in names
        },
    )
    print(f"  K={K}: union={len(union)} (+{(len(union)/max(len(fusion_caught),1)-1)*100:.1f}% over fusion)")
    print(f"    per-model: " + ", ".join(f"{ni}={len(caught[ni])}" for ni in names))
    print(f"    marginal unique: " + ", ".join(f"{ni}={overlap[f'top_{K}']['marginal_unique'][ni]}" for ni in names))

# ---------------------------------------------------------------------------
# 6. Naive blends (same set as Sonya's pilot)
# ---------------------------------------------------------------------------
print("\n=== Naive blends (point estimates) ===")
blends = {
    # singletons
    "fusion_alone":            fusion,
    "karina_alone":            karina,
    "diana_alone":             diana,
    "albina_alone":            albina,
    # baselines from previous pilots
    "blend_07f_03k":           0.7 * fusion + 0.3 * karina,
    "blend_06f_02k_02d":       0.6 * fusion + 0.2 * karina + 0.2 * diana,
    "uniform_3way_fkd":        (fusion + karina + diana) / 3.0,
    # 4-way blends (Sonya's set)
    "uniform_4way":            (fusion + karina + diana + albina) / 4.0,
    "fusion_heavy_4way":       0.4 * fusion + 0.2 * karina + 0.2 * diana + 0.2 * albina,
    "fusion_diana_heavy_4way": 0.35 * fusion + 0.15 * karina + 0.30 * diana + 0.20 * albina,
    "rank_avg_4way":           (rank_norm(fusion) + rank_norm(karina) + rank_norm(diana) + rank_norm(albina)) / 4.0,
    # pair / triple
    "pair_fusion_albina":      0.5 * fusion + 0.5 * albina,
    "triple_fkd":              (fusion + karina + diana) / 3.0,
    "triple_fka":              (fusion + karina + albina) / 3.0,
    "triple_fda":              (fusion + diana  + albina) / 3.0,
    "triple_kda":              (karina + diana + albina) / 3.0,
}
blend_metrics = {name: metrics(y, p) for name, p in blends.items()}
for name in sorted(blend_metrics, key=lambda n_: -blend_metrics[n_]["pr"]):
    m = blend_metrics[name]
    print(f"  {name:30s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

# ---------------------------------------------------------------------------
# 7. Mini grid search for new optimal weights (PR-AUC and R@P targets)
# ---------------------------------------------------------------------------
print("\n=== Grid search 4-way (step=0.05, sum=1.0) ===")
grid = []
step = 0.05
ws = np.arange(0, 1.0001, step)
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
# Top-5 by PR-AUC
top_pr = np.argsort(-grid_arr['pr'])[:10]
print("  Top-10 by PR-AUC:")
for i in top_pr:
    g = grid_arr[i]
    print(f"    f={g['wf']:.2f} k={g['wk']:.2f} d={g['wd']:.2f} a={g['wa']:.2f}  "
          f"ROC={g['roc']:.4f} PR={g['pr']:.4f} R@P={g['r_at_p']:.4f}")
top_rp = np.argsort(-grid_arr['r_at_p'])[:10]
print("  Top-10 by R@P>=0.9:")
for i in top_rp:
    g = grid_arr[i]
    print(f"    f={g['wf']:.2f} k={g['wk']:.2f} d={g['wd']:.2f} a={g['wa']:.2f}  "
          f"ROC={g['roc']:.4f} PR={g['pr']:.4f} R@P={g['r_at_p']:.4f}")

best_pr_w = grid_arr[top_pr[0]]
best_rp_w = grid_arr[top_rp[0]]
best_pr_blend = (float(best_pr_w['wf']) * fusion + float(best_pr_w['wk']) * karina +
                 float(best_pr_w['wd']) * diana + float(best_pr_w['wa']) * albina)
blends["grid_best_pr"] = best_pr_blend
blend_metrics["grid_best_pr"] = metrics(y, best_pr_blend)

best_rp_blend = (float(best_rp_w['wf']) * fusion + float(best_rp_w['wk']) * karina +
                 float(best_rp_w['wd']) * diana + float(best_rp_w['wa']) * albina)
blends["grid_best_rp"] = best_rp_blend
blend_metrics["grid_best_rp"] = metrics(y, best_rp_blend)

# ---------------------------------------------------------------------------
# 8. Best blend by PR-AUC (from Sonya's hand-picked set, exclude singletons)
# ---------------------------------------------------------------------------
singletons = {"fusion_alone", "karina_alone", "diana_alone", "albina_alone"}
candidates = [n_ for n_ in blends if n_ not in singletons]
best_name = max(candidates, key=lambda n_: blend_metrics[n_]["pr"])
print(f"\n=== Best blend by PR-AUC: {best_name} ===")
print(f"  PR-AUC = {blend_metrics[best_name]['pr']:.4f}")
print(f"  ROC    = {blend_metrics[best_name]['roc']:.4f}")
print(f"  R@P90  = {blend_metrics[best_name]['r_at_p90']:.4f}")

best_blend = blends[best_name]

# ---------------------------------------------------------------------------
# 9. Paired bootstrap: best blend vs EACH single model (4 pairs)
# ---------------------------------------------------------------------------
print(f"\n=== Paired bootstrap (B={B}, seed={SEED}), best blend = {best_name} ===")
rng = np.random.default_rng(SEED)
singles = {"fusion": fusion, "karina": karina, "diana": diana, "albina": albina}

# Pre-compute bootstrap deltas vs each single
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
for nm, deltas in ci_vs_each.items():
    summary = {}
    for k, vs in deltas.items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo > 0 or hi < 0))
    bootstrap_vs_each[nm] = summary
    print(f"\n  best blend vs {nm}_alone:")
    for k, s in summary.items():
        sig = "*" if s["significant"] else " "
        print(f"    Δ{k:9s} = {s['mean']:+.4f}  95% CI [{s['lo']:+.4f}; {s['hi']:+.4f}] {sig}")

# Also: alternative blends vs fusion (like Sonya's section 8)
print(f"\n=== Alt blends vs fusion_alone (paired bootstrap) ===")
rng = np.random.default_rng(SEED)
alt_blends = ["uniform_4way", "fusion_heavy_4way", "fusion_diana_heavy_4way",
              "rank_avg_4way", "pair_fusion_albina", "grid_best_pr", "grid_best_rp"]
alt_deltas = {nm: {"roc": [], "pr": [], "r_at_p90": []} for nm in alt_blends}
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    m_base = metrics(yb, fusion[idx])
    for nm in alt_blends:
        m_b = metrics(yb, blends[nm][idx])
        for k in alt_deltas[nm]:
            alt_deltas[nm][k].append(m_b[k] - m_base[k])

bootstrap_alt_vs_fusion = {}
for nm, deltas in alt_deltas.items():
    summary = {}
    for k, vs in deltas.items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo > 0 or hi < 0))
    bootstrap_alt_vs_fusion[nm] = summary
    m = blend_metrics[nm]
    print(f"  {nm:30s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P={m['r_at_p90']:.4f}  | "
          f"ΔPR={summary['pr']['mean']:+.4f}* " if summary['pr']['significant']
          else f"  {nm:30s} ΔPR={summary['pr']['mean']:+.4f} ")

# ---------------------------------------------------------------------------
# 10. Save artefacts
# ---------------------------------------------------------------------------
output = {
    "n_test": n,
    "n_positives": n_pos,
    "positive_rate": float(y.mean()),
    "seed": SEED,
    "bootstrap_B": B,
    "diana_source": "M2-FE+ standalone (npy_files/test_proba_diana_team.npy, 2026-05-11 17:33)",
    "per_model_metrics": per_model,
    "correlations": corr_matrix,
    "topk_overlap": overlap,
    "blend_metrics": blend_metrics,
    "best_blend_by_pr": best_name,
    "grid_best_pr_weights": {"fusion": float(best_pr_w['wf']), "karina": float(best_pr_w['wk']),
                              "diana": float(best_pr_w['wd']), "albina": float(best_pr_w['wa'])},
    "grid_best_rp_weights": {"fusion": float(best_rp_w['wf']), "karina": float(best_rp_w['wk']),
                              "diana": float(best_rp_w['wd']), "albina": float(best_rp_w['wa'])},
    "bootstrap_best_vs_each_single": bootstrap_vs_each,
    "bootstrap_alt_blends_vs_fusion": bootstrap_alt_vs_fusion,
}
out_path = SONYA_PILOT / "diana_rerun_4domains.json"
out_path.write_text(json.dumps(output, indent=2, default=float))
print(f"\nSaved JSON: {out_path}")

# Also save the best blend as a npy for Sonya
np.save(SONYA_PILOT / "test_proba_4way_best_diana_rerun.npy", best_blend)
print(f"Saved best blend proba: {SONYA_PILOT / 'test_proba_4way_best_diana_rerun.npy'}")
