"""
Direct A/B: Sonya's old 4-way blend (with old Diana) vs Sonya's same blend (with new Diana),
and vs new grid-optimal blend (with new Diana).

Computes paired bootstrap CIs for the upgrade effect alone.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_auc_score)

DIANA_FOLDER = Path("/Users/diana/master-thesis-2025/Диана_ВКР_финал")
SONYA_PILOT = DIANA_FOLDER / "scripts/sonya_pilot"
NPY = Path("/Users/diana/master-thesis-2025/npy_files")
SPLITS = DIANA_FOLDER / "notebooks/team_splits"

SEED = 42
B = 1000

y      = np.load(SPLITS / "y_test.npy").astype(int)
fusion = np.load(SONYA_PILOT / "proba/test_proba_fusion_team.npy").astype(np.float64)
karina = np.load(NPY / "test_proba_karina_team.npy").astype(np.float64)
albina = np.load(NPY / "test_proba_albina_team.npy").astype(np.float64)

diana_old = np.load(SONYA_PILOT / "proba/test_proba_diana_team.npy").astype(np.float64)  # M2 mainline clean
diana_new = np.load(NPY / "test_proba_diana_team.npy").astype(np.float64)  # M2-FE+ standalone

n = len(y)

def recall_at_precision(y, p, target=0.90):
    pr, rc, _ = precision_recall_curve(y, p)
    m = pr >= target
    return float(rc[m].max()) if m.any() else 0.0

def metrics(y, p):
    return dict(roc=roc_auc_score(y,p), pr=average_precision_score(y,p), r_at_p90=recall_at_precision(y,p,0.90))

# Sonya's headline blend (same weights)
def blend_sonya(d):
    return 0.35*fusion + 0.15*karina + 0.30*d + 0.20*albina

old_blend = blend_sonya(diana_old)
new_blend_same_w = blend_sonya(diana_new)

# Grid best from rerun (f=0.25, k=0.00, d=0.50, a=0.25)
grid_best = 0.25*fusion + 0.00*karina + 0.50*diana_new + 0.25*albina

print("\n=== Singletons ===")
for nm, p in [("diana_OLD (M2 mainline)", diana_old), ("diana_NEW (M2-FE+)", diana_new)]:
    m = metrics(y,p); print(f"  {nm:30s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

print("\n=== Blends ===")
for nm, p in [
    ("Sonya old (with diana_old)", old_blend),
    ("Sonya weights (with diana_new)", new_blend_same_w),
    ("Grid best (f=.25 d=.50 a=.25, diana_new)", grid_best),
]:
    m = metrics(y,p); print(f"  {nm:42s} ROC={m['roc']:.4f} PR={m['pr']:.4f} R@P90={m['r_at_p90']:.4f}")

print(f"\n=== Paired bootstrap B={B} ===")
rng = np.random.default_rng(SEED)
pairs = [
    ("new_same_w_vs_old", new_blend_same_w, old_blend),
    ("grid_best_vs_old",  grid_best,        old_blend),
    ("grid_best_vs_new_same_w", grid_best,  new_blend_same_w),
]
deltas = {nm: {"roc":[], "pr":[], "r_at_p90":[]} for nm, _, _ in pairs}
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    yb = y[idx]
    if yb.sum()==0 or yb.sum()==n:
        continue
    for nm, A, Bx in pairs:
        mA = metrics(yb, A[idx]); mB = metrics(yb, Bx[idx])
        for k in deltas[nm]:
            deltas[nm][k].append(mA[k] - mB[k])

out = {}
for nm in deltas:
    summary = {}
    for k, vs in deltas[nm].items():
        vs = np.array(vs)
        lo, hi = np.quantile(vs, [0.025, 0.975])
        summary[k] = dict(mean=float(vs.mean()), lo=float(lo), hi=float(hi),
                          significant=bool(lo>0 or hi<0))
    out[nm] = summary
    print(f"\n  {nm}:")
    for k, s in summary.items():
        sig = "*" if s["significant"] else " "
        print(f"    Δ{k:9s} = {s['mean']:+.4f}  95% CI [{s['lo']:+.4f}; {s['hi']:+.4f}] {sig}")

(SONYA_PILOT / "old_vs_new_diana_diff.json").write_text(json.dumps({
    "singletons": {
        "diana_old": metrics(y, diana_old),
        "diana_new_m2_fe_plus": metrics(y, diana_new),
    },
    "blends": {
        "sonya_old_with_diana_old":   metrics(y, old_blend),
        "sonya_weights_with_diana_new": metrics(y, new_blend_same_w),
        "grid_best_diana_new":        metrics(y, grid_best),
    },
    "bootstrap_pairs": out,
}, indent=2, default=float))
print(f"\nSaved: {SONYA_PILOT / 'old_vs_new_diana_diff.json'}")
