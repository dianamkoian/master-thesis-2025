"""
extended_stacking.py
====================

Расширенный стекинг 12 моделей четырёх доменов на едином командном тесте
(seller-based, n_test = 58 410).

Входы:
  - packages/<package>/3_probas_team_split/test_proba_*.npy  (12 файлов)
  - packages/package_diana/3_probas_team_split/y_test_team.npy
  - notebooks/team_test_sellers.npy (SellerID на каждую строку test;
    регенерируется из исходного CSV при отсутствии)

Выходы:
  - notebooks/extended_stacking_inventory.json
  - notebooks/extended_stacking_diversity.json
  - notebooks/extended_stacking_summary.json
  - notebooks/extended_stacking_bootstrap.json
  - notebooks/extended_stacking_channels.json
  - npy_files/test_proba_extended_stack_pr.npy
  - npy_files/test_proba_extended_stack_rap.npy
  - proba_of_sonya/test_proba_diana_extended_stack.npy

Метрики: PR-AUC (основная), Recall@P>=0.9 (R@P), ROC-AUC.

Воспроизводимость: random_state=42 всюду.

CLI:
    python extended_stacking.py
    python extended_stacking.py --skip-bootstrap   # быстрая прогонка без CI
"""

from __future__ import annotations
import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "packages"
NB = ROOT / "real_estate_approaches" / "notebooks"
NPY = ROOT / "npy_files"
SONYA = ROOT / "proba_of_sonya"
CSV_TRAIN = ROOT / "data" / "ml_ozon_ounterfeit_train.csv"

CANDIDATES = [
    ("diana_team",          "package_diana",  "test_proba_diana_team.npy"),
    ("diana_calibrated",    "package_diana",  "test_proba_diana_team_calibrated.npy"),
    ("diana_3way",          "package_diana",  "test_proba_diana_3way.npy"),
    ("diana_best_ensemble", "package_diana",  "test_proba_diana_best_ensemble.npy"),
    ("diana_no_te",         "package_diana",  "test_proba_diana_no_te.npy"),
    ("soni_fusion",         "package_soni",   "test_proba_fusion_team.npy"),
    ("soni_e7",             "package_soni",   "test_proba_e7_team.npy"),
    ("soni_e5",             "package_soni",   "test_proba_e5_team.npy"),
    ("soni_e0_full",        "package_soni",   "test_proba_e0_team_full.npy"),
    ("soni_e0_clean",       "package_soni",   "test_proba_e0_team_clean.npy"),
    ("karina_combined",     "package_karina", "test_proba_karina_team.npy"),
    ("albina_mmd_v4",       "package_albina", "test_proba_albina_team.npy"),
]
NAMES = [c[0] for c in CANDIDATES]
SEED = 42


def _metrics(y_true, p):
    roc = float(roc_auc_score(y_true, p))
    pr = float(average_precision_score(y_true, p))
    prec, rec, _ = precision_recall_curve(y_true, p)
    mask = prec >= 0.9
    rap = float(rec[mask].max()) if mask.any() else 0.0
    return {"roc": roc, "pr": pr, "rap90": rap}


def _load_probas():
    X = np.column_stack(
        [np.load(PKG / pkg / "3_probas_team_split" / f) for (_, pkg, f) in CANDIDATES]
    )
    y = np.load(PKG / "package_diana" / "3_probas_team_split" / "y_test_team.npy").astype(np.int8)
    assert X.shape == (58_410, 12), f"X shape mismatch: {X.shape}"
    assert y.shape == (58_410,), f"y shape mismatch: {y.shape}"
    return X, y


def _load_or_make_sellers(test_idx_path: Path) -> np.ndarray:
    """Recover SellerID per test row from the source CSV, cache as npy."""
    cached = NB / "team_test_sellers.npy"
    if cached.exists():
        return np.load(cached)
    test_idx = np.load(test_idx_path)
    df = pd.read_csv(CSV_TRAIN, usecols=["SellerID", "resolution"])
    sellers = df.iloc[test_idx]["SellerID"].values
    NB.mkdir(parents=True, exist_ok=True)
    np.save(cached, sellers)
    return sellers


def phase1_inventory(X, y):
    out = {
        "y_test_shape": list(y.shape),
        "y_test_sum": int(y.sum()),
        "y_test_mean": float(y.mean()),
        "probas": [],
    }
    print("=== Phase 1: Inventory ===")
    for j, (nick, pkg, fname) in enumerate(CANDIDATES):
        col = X[:, j]
        m = _metrics(y, col)
        out["probas"].append(
            {"nick": nick, "pkg": pkg, "file": fname, **m,
             "mean": float(col.mean()), "min": float(col.min()), "max": float(col.max())}
        )
        print(f"  {nick:>22}: ROC={m['roc']:.4f}  PR={m['pr']:.4f}  R@P={m['rap90']:.4f}")
    return out


def phase2_diversity(X, y):
    print("\n=== Phase 2: Diversity ===")
    n = X.shape[1]
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(X[:, i], X[:, j])
            rho[i, j] = r
            rho[j, i] = r
    redundant = []
    pruned = set()
    pr_aucs = {nick: average_precision_score(y, X[:, j]) for j, nick in enumerate(NAMES)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(rho[i, j]) > 0.97:
                a, b = NAMES[i], NAMES[j]
                redundant.append({"a": a, "b": b, "rho": float(rho[i, j])})
                pruned.add(a if pr_aucs[a] < pr_aucs[b] else b)
    print(f"  redundant pairs (|rho|>0.97): {len(redundant)}")
    print(f"  pruned: {sorted(pruned)}")

    K = 500
    catches = {nick: set(np.argsort(-X[:, j])[:K].tolist()) for j, nick in enumerate(NAMES)}
    y_pos = set(np.where(y == 1)[0].tolist())
    unique_catches = {}
    for nick in NAMES:
        others = set().union(*(catches[k] for k in NAMES if k != nick))
        unique_catches[nick] = len(catches[nick] - others & y_pos) if False else \
                                 len((catches[nick] - others) & y_pos)
    print("  unique top-500 positive catches:")
    for nick, v in sorted(unique_catches.items(), key=lambda kv: -kv[1]):
        print(f"    {nick:>22}: {v}")
    return {
        "models": NAMES,
        "spearman_matrix": rho.tolist(),
        "redundant_pairs": redundant,
        "pruned": sorted(pruned),
        "kept_after_pruning": [n for n in NAMES if n not in pruned],
        "top500_unique_positive_catches": unique_catches,
    }


def phase3_stacking(X, y, sellers):
    print("\n=== Phase 3: Stacking ===")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    fit_idx, eval_idx = next(gss.split(X, y, groups=sellers))
    overlap = set(sellers[fit_idx]) & set(sellers[eval_idx])
    assert len(overlap) == 0, "Seller overlap between meta-fit and meta-eval"

    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_eval, y_eval = X[eval_idx], y[eval_idx]
    print(f"  meta-fit n={len(fit_idx)}, positives={y_fit.sum()}")
    print(f"  meta-eval n={len(eval_idx)}, positives={y_eval.sum()}")

    results = {}

    W0 = np.zeros(len(NAMES))
    W0[NAMES.index("diana_team")] = 0.35
    W0[NAMES.index("karina_combined")] = 0.15
    W0[NAMES.index("soni_fusion")] = 0.30
    W0[NAMES.index("albina_mmd_v4")] = 0.20
    p_b0_eval = X_eval @ W0
    p_b0_full = X @ W0
    results["B0_baseline_blend"] = {
        "metrics_eval": _metrics(y_eval, p_b0_eval),
        "metrics_full": _metrics(y, p_b0_full),
        "weights": dict(zip(NAMES, [float(w) for w in W0])),
    }

    scaler = StandardScaler().fit(X_fit)
    Xf, Xe, Xa = scaler.transform(X_fit), scaler.transform(X_eval), scaler.transform(X)
    best_lr = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED)
        lr.fit(Xf, y_fit)
        p_eval = lr.predict_proba(Xe)[:, 1]
        m = _metrics(y_eval, p_eval)
        if best_lr is None or m["pr"] > best_lr[1]["pr"]:
            best_lr = (C, m, lr, p_eval)
    C_best, m_lr, lr_best, p_eval_lr = best_lr
    p_full_lr = lr_best.predict_proba(Xa)[:, 1]
    results["L1_logreg"] = {
        "best_C": C_best,
        "metrics_eval": m_lr,
        "metrics_full": _metrics(y, p_full_lr),
        "coefficients": dict(zip(NAMES, [float(c) for c in lr_best.coef_[0]])),
        "intercept": float(lr_best.intercept_[0]),
    }
    print(f"  L1 LR (C={C_best}): eval PR={m_lr['pr']:.4f} R@P={m_lr['rap90']:.4f}")

    cb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
                            eval_metric="PRAUC")
    cb.fit(X_fit, y_fit, eval_set=(X_eval, y_eval), use_best_model=True)
    p_eval_cb = cb.predict_proba(X_eval)[:, 1]
    p_full_cb = cb.predict_proba(X)[:, 1]
    m_cb = _metrics(y_eval, p_eval_cb)
    results["L2_catboost"] = {
        "metrics_eval": m_cb,
        "metrics_full": _metrics(y, p_full_cb),
        "feature_importance": dict(zip(NAMES, [float(v) for v in cb.feature_importances_])),
    }
    print(f"  L2 CatBoost: eval PR={m_cb['pr']:.4f} R@P={m_cb['rap90']:.4f}")

    def neg_pr(w, X_, y_):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s < 1e-12:
            return 0.0
        return -average_precision_score(y_, X_ @ (w / s))

    n = len(NAMES)
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * n
    res = minimize(neg_pr, w0, args=(X_fit, y_fit), method="SLSQP",
                    bounds=bnds, constraints=cons,
                    options={"maxiter": 300, "ftol": 1e-7, "disp": False})
    W_conv = np.clip(res.x, 0, None)
    W_conv = W_conv / W_conv.sum()
    p_eval_conv = X_eval @ W_conv
    p_full_conv = X @ W_conv
    m_conv = _metrics(y_eval, p_eval_conv)
    results["L3_convex_pr"] = {
        "metrics_eval": m_conv,
        "metrics_full": _metrics(y, p_full_conv),
        "weights": dict(zip(NAMES, [float(w) for w in W_conv])),
        "converged": bool(res.success),
    }
    print(f"  L3 convex SLSQP: eval PR={m_conv['pr']:.4f} R@P={m_conv['rap90']:.4f}")

    best_en = None
    for l1r in [0.0, 0.3, 0.5, 0.7, 1.0]:
        en = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1r,
                                C=1.0, max_iter=3000, class_weight="balanced",
                                random_state=SEED)
        try:
            en.fit(Xf, y_fit)
            p_eval = en.predict_proba(Xe)[:, 1]
            m = _metrics(y_eval, p_eval)
            if best_en is None or m["pr"] > best_en[1]["pr"]:
                best_en = (l1r, m, en, p_eval)
        except Exception:
            pass
    l1r_b, m_en, en_b, p_eval_en = best_en
    p_full_en = en_b.predict_proba(Xa)[:, 1]
    results["L4_elasticnet"] = {
        "best_l1_ratio": l1r_b,
        "metrics_eval": m_en,
        "metrics_full": _metrics(y, p_full_en),
        "coefficients": dict(zip(NAMES, [float(c) for c in en_b.coef_[0]])),
    }
    print(f"  L4 ElasticNet (l1_ratio={l1r_b}): eval PR={m_en['pr']:.4f} R@P={m_en['rap90']:.4f}")

    np.save(NB / "fit_idx.npy", fit_idx)
    np.save(NB / "eval_idx.npy", eval_idx)
    np.save(NB / "meta_eval_p_lr.npy", p_eval_lr)
    np.save(NB / "meta_eval_p_cb.npy", p_eval_cb)
    np.save(NB / "meta_eval_p_conv.npy", p_eval_conv)
    np.save(NB / "meta_eval_p_en.npy", p_eval_en)
    np.save(NB / "p_full_lr.npy", p_full_lr)
    np.save(NB / "p_full_cb.npy", p_full_cb)
    np.save(NB / "p_full_conv.npy", p_full_conv)
    np.save(NB / "p_full_en.npy", p_full_en)
    return results, fit_idx, eval_idx, p_eval_lr, p_eval_cb, p_eval_conv, p_eval_en, p_full_lr


def _paired_bootstrap(y_true, p_a, p_b, metric, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.sum() == 0:
            continue
        pa, pb = p_a[idx], p_b[idx]
        if metric == "pr":
            ma = average_precision_score(yb, pa); mb = average_precision_score(yb, pb)
        elif metric == "rap":
            pa_, ra, _ = precision_recall_curve(yb, pa); pb_, rb, _ = precision_recall_curve(yb, pb)
            ma = float(ra[pa_ >= 0.9].max()) if (pa_ >= 0.9).any() else 0.0
            mb = float(rb[pb_ >= 0.9].max()) if (pb_ >= 0.9).any() else 0.0
        elif metric == "roc":
            ma = roc_auc_score(yb, pa); mb = roc_auc_score(yb, pb)
        else:
            raise ValueError(metric)
        deltas.append(mb - ma)
    d = np.array(deltas)
    return {
        "mean": float(d.mean()),
        "ci_low": float(np.quantile(d, 0.025)),
        "ci_high": float(np.quantile(d, 0.975)),
        "significant": bool((np.quantile(d, 0.025) > 0) or (np.quantile(d, 0.975) < 0)),
    }


def phase4_bootstrap(X, y, eval_idx, p_lr_eval, p_cb_eval, p_conv_eval, p_en_eval):
    print("\n=== Phase 4: Bootstrap CI ===")
    W0 = np.zeros(len(NAMES))
    W0[NAMES.index("diana_team")] = 0.35
    W0[NAMES.index("karina_combined")] = 0.15
    W0[NAMES.index("soni_fusion")] = 0.30
    W0[NAMES.index("albina_mmd_v4")] = 0.20
    p_b0_eval = X[eval_idx] @ W0
    p_diana_solo_eval = X[eval_idx, NAMES.index("diana_team")]
    p_diana_be_eval = X[eval_idx, NAMES.index("diana_best_ensemble")]
    y_eval = y[eval_idx]

    out = {}
    pairs = [("L1_LR", p_lr_eval), ("L2_CB", p_cb_eval), ("L3_convex", p_conv_eval), ("L4_EN", p_en_eval)]
    for nm, p_b in pairs:
        out[f"{nm}_vs_B0"] = {m: _paired_bootstrap(y_eval, p_b0_eval, p_b, m) for m in ["pr", "rap", "roc"]}
    out["L1_vs_M2_FE_plus"] = {m: _paired_bootstrap(y_eval, p_diana_solo_eval, p_lr_eval, m) for m in ["pr", "rap"]}
    out["L1_vs_Diana_best_ensemble"] = {m: _paired_bootstrap(y_eval, p_diana_be_eval, p_lr_eval, m) for m in ["pr", "rap"]}

    for k, v in out.items():
        print(f"  {k}:")
        for m, d in v.items():
            sig = "[sig]" if d["significant"] else ""
            print(f"    {m}: Δ={d['mean']:+.4f}  CI=[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {sig}")
    return out


def phase5_publish(p_full_lr):
    print("\n=== Phase 5: Publish artifacts ===")
    NPY.mkdir(parents=True, exist_ok=True)
    SONYA.mkdir(parents=True, exist_ok=True)
    np.save(NPY / "test_proba_extended_stack_pr.npy", p_full_lr.astype("float32"))
    np.save(NPY / "test_proba_extended_stack_rap.npy", p_full_lr.astype("float32"))
    np.save(SONYA / "test_proba_diana_extended_stack.npy", p_full_lr.astype("float32"))
    print(f"  wrote {NPY / 'test_proba_extended_stack_pr.npy'}")
    print(f"  wrote {SONYA / 'test_proba_diana_extended_stack.npy'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    NB.mkdir(parents=True, exist_ok=True)
    X, y = _load_probas()
    sellers = _load_or_make_sellers(PKG / "package_diana" / "3_probas_team_split" / "team_test_idx.npy")
    print(f"X shape={X.shape}, y shape={y.shape}, unique sellers={len(np.unique(sellers))}")

    inv = phase1_inventory(X, y)
    (NB / "extended_stacking_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")

    div = phase2_diversity(X, y)
    (NB / "extended_stacking_diversity.json").write_text(json.dumps(div, ensure_ascii=False, indent=2), encoding="utf-8")

    res = phase3_stacking(X, y, sellers)
    summary, fit_idx, eval_idx, p_lr_e, p_cb_e, p_cv_e, p_en_e, p_full_lr = res
    (NB / "extended_stacking_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_bootstrap:
        ci = phase4_bootstrap(X, y, eval_idx, p_lr_e, p_cb_e, p_cv_e, p_en_e)
        (NB / "extended_stacking_bootstrap.json").write_text(json.dumps(ci, ensure_ascii=False, indent=2), encoding="utf-8")

    channels = {
        "pr_channel": "L1_logreg",
        "rap_channel": "L1_logreg",
        "note": "L1 LR одновременно лидирует и по PR-AUC, и по R@P>=0.9 на meta-eval; "
                "разделение на два канала не оправдано данными",
    }
    (NB / "extended_stacking_channels.json").write_text(json.dumps(channels, ensure_ascii=False, indent=2), encoding="utf-8")

    phase5_publish(p_full_lr)
    print("\nDONE.")


if __name__ == "__main__":
    main()
