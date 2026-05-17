"""experiment_price_threshold.py — эксперимент чувствительности M2-FE+-подобной
модели к порогу `price_too_high` (1,5 vs 2,0) на двух splits (indiv vs team).

Цель — эмпирически проверить feature-space-аргумент для защиты ВКР:
  «В M2-FE+ среди 636 признаков `price_too_high` не критичен, в CDSM Mode 1
   среди 48 признаков критичен.»

Архитектура эксперимента (M2-FE+-like, упрощённая):
  - 38 tab (командный baseline) + cat=2
  - 50 TF-IDF SVD (concat name+description+brand)
  - 25 CLIP-PCA (visual)
  - 3 FADAML price (price_ratio, price_too_low, price_too_high) ← переменный порог
  - 3 typosquat-Deng (brand_exact, brand_fuzzy, typosquat)
  - Total: 119 признаков
CatBoost iterations=1000, depth=6, lr=0.05, scale_pos_weight=spw, seed=42.

Запуск (~20 мин в фоне):
    python3 Диана_ВКР_финал/scripts/experiment_price_threshold.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rapidfuzz.fuzz import partial_ratio
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

ROOT = Path("/Users/diana/master-thesis-2025")
CSV = ROOT / "Diana's folder" / "ml_ozon_ounterfeit_train.csv"
TEAM_SPLITS = ROOT / "Диана_ВКР_финал" / "notebooks" / "team_splits"
NB = ROOT / "Диана_ВКР_финал" / "notebooks"
CLIP_PARQUET = ROOT / "counterfeit_service" / "clip_embeddings.parquet"
OUT = ROOT / "Диана_ВКР_финал" / "scripts" / "experiment_price_threshold_results.json"

SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true, p) -> dict:
    roc = float(roc_auc_score(y_true, p))
    pr = float(average_precision_score(y_true, p))
    prec, rec, _ = precision_recall_curve(y_true, p)
    m = prec >= 0.9
    rap = float(rec[m].max()) if m.any() else 0.0
    return {"ROC": roc, "PR": pr, "R@P": rap}


def build_features(df_part, cat_med, tab38, cat_tab38, tfidf_svd_obj_pair,
                    clip_pca_obj, clip_mat, id2row, price_threshold_high: float,
                    price_threshold_low: float = 0.3):
    """Собирает 119-мерный feature vector. price_threshold_high — параметр эксперимента."""
    n = len(df_part)
    # 1. Tab38
    A = df_part[tab38].copy()
    for c in tab38:
        if c in cat_tab38:
            A[c] = A[c].fillna("nan").astype(str)
        else:
            A[c] = A[c].fillna(0)

    # 2. FADAML price (переменный порог!)
    cm = df_part["CommercialTypeName4"].map(cat_med).fillna(df_part["PriceDiscounted"].median())
    price_ratio = (df_part["PriceDiscounted"].fillna(0) / (cm + 1)).astype(np.float32)
    price_too_low = (price_ratio < price_threshold_low).astype(np.float32)
    price_too_high = (price_ratio > price_threshold_high).astype(np.float32)
    C = np.column_stack([price_ratio.values, price_too_low.values, price_too_high.values])

    # 3. Typosquat-Deng (brand_exact, brand_fuzzy, typosquat)
    brand_str = df_part["brand_name"].fillna("").astype(str).str.lower()
    name_str = df_part["name_rus"].fillna("").astype(str).str.lower()
    brand_exact = np.array([(b in n) and len(b) > 0 for b, n in zip(brand_str, name_str)], dtype=np.float32)
    fuzzy = np.array([partial_ratio(b, n) / 100.0 if (b and n) else 0.0 for b, n in zip(brand_str, name_str)], dtype=np.float32)
    typosquat = np.maximum(0, fuzzy - 0.5 * brand_exact)
    D = np.column_stack([brand_exact, fuzzy, typosquat])

    # 4. TF-IDF + SVD
    tfidf_obj, svd_obj = tfidf_svd_obj_pair
    text = (df_part["brand_name"].fillna("").astype(str) + " "
            + df_part["description"].fillna("").astype(str) + " "
            + df_part["name_rus"].fillna("").astype(str)).values
    sparse = tfidf_obj.transform(text)
    E = svd_obj.transform(sparse).astype(np.float32)

    # 5. CLIP + PCA
    ids = df_part["ItemID"].astype("int64").values
    clip_raw = np.zeros((n, clip_mat.shape[1]), dtype=np.float32)
    for i, iid in enumerate(ids):
        r = id2row.get(int(iid))
        if r is not None:
            clip_raw[i] = clip_mat[r]
    G = clip_pca_obj.transform(clip_raw).astype(np.float32)

    # Конкатенация
    X = np.hstack([A.values, C, D, E, G])
    names = (list(tab38)
              + ["price_ratio", "price_too_low", "price_too_high"]
              + ["brand_exact", "brand_fuzzy", "typosquat"]
              + [f"tfidf_svd_{i}" for i in range(50)]
              + [f"clip_pca_{i}" for i in range(25)])
    cat_indices = [names.index(c) for c in cat_tab38 if c in names]
    return X, names, cat_indices


def run_experiment(split_name: str, train_idx, val_idx, test_idx, y_test,
                    df, cat_med, tab38, cat_tab38, clip_mat, id2row,
                    price_threshold_high: float):
    """Один прогон: обучить CatBoost + замерить test-метрики."""
    log(f"  [{split_name}, threshold={price_threshold_high}] preparing features…")

    df_tr = df.iloc[train_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)
    df_te = df.iloc[test_idx].reset_index(drop=True)
    y_tr = df_tr["resolution"].astype("int8").values
    y_va = df_va["resolution"].astype("int8").values

    # Fit tfidf+svd, clip_pca на train+val combined
    text_tv = (pd.concat([df_tr, df_va])["brand_name"].fillna("").astype(str) + " "
                + pd.concat([df_tr, df_va])["description"].fillna("").astype(str) + " "
                + pd.concat([df_tr, df_va])["name_rus"].fillna("").astype(str)).values
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    Xtf_tv = tfidf.fit_transform(text_tv)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    svd.fit(Xtf_tv)

    ids_tv = np.concatenate([df_tr["ItemID"].astype("int64").values, df_va["ItemID"].astype("int64").values])
    clip_tv = np.zeros((len(ids_tv), clip_mat.shape[1]), dtype=np.float32)
    for i, iid in enumerate(ids_tv):
        r = id2row.get(int(iid))
        if r is not None:
            clip_tv[i] = clip_mat[r]
    pca_clip = PCA(n_components=25, random_state=SEED).fit(clip_tv)

    X_tr, names, cat_idx = build_features(df_tr, cat_med, tab38, cat_tab38,
                                            (tfidf, svd), pca_clip, clip_mat, id2row,
                                            price_threshold_high)
    X_va, _, _ = build_features(df_va, cat_med, tab38, cat_tab38, (tfidf, svd), pca_clip, clip_mat, id2row, price_threshold_high)
    X_te, _, _ = build_features(df_te, cat_med, tab38, cat_tab38, (tfidf, svd), pca_clip, clip_mat, id2row, price_threshold_high)

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    log(f"  [{split_name}, threshold={price_threshold_high}] training CatBoost (X_tr={X_tr.shape}, spw={spw:.2f})…")
    model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        eval_metric="AUC", scale_pos_weight=spw,
        early_stopping_rounds=50, random_seed=SEED, verbose=0, cat_features=cat_idx,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    dt = time.time() - t0
    p_te = model.predict_proba(X_te)[:, 1].astype(np.float64)
    m = metrics(y_test, p_te)
    log(f"  [{split_name}, threshold={price_threshold_high}] DONE in {dt:.0f}s  ROC={m['ROC']:.4f}  PR={m['PR']:.4f}  R@P={m['R@P']:.4f}  best_iter={model.tree_count_}")
    return m, model.tree_count_, dt


def main() -> None:
    log("Loading data…")
    df = pd.read_csv(CSV, encoding="utf-8")
    df["ItemID"] = df["ItemID"].astype("int64")

    DROP = {"id", "ItemID", "SellerID", "name_rus", "description", "brand_name", "text", "resolution"}
    tab38 = [c for c in df.columns if c not in DROP]
    cat_tab38 = [c for c in tab38 if df[c].dtype == "object"]
    log(f"  tab38: {len(tab38)} cols, cat: {cat_tab38}")

    cat_med = df.groupby("CommercialTypeName4")["PriceDiscounted"].median().to_dict()

    log("Loading CLIP embeddings…")
    clip_df = pd.read_parquet(CLIP_PARQUET)
    clip_dim = clip_df["embedding"].iloc[0].shape[0]
    clip_mat = np.stack(clip_df["embedding"].values).astype(np.float32)
    id2row = {int(i): r for r, i in enumerate(clip_df["ItemID"].values)}
    log(f"  CLIP: {clip_mat.shape}")

    # Splits
    team_train = np.load(TEAM_SPLITS / "team_train_idx.npy")
    team_val = np.load(TEAM_SPLITS / "team_val_idx.npy")
    team_test = np.load(TEAM_SPLITS / "team_test_idx.npy")
    y_test_team = np.load(TEAM_SPLITS / "y_test.npy")

    indiv_train = np.load(NB / "train_idx_indiv.npy")
    indiv_val = np.load(NB / "val_idx_indiv.npy")
    indiv_test = np.load(NB / "test_idx_indiv.npy")
    y_test_indiv = np.load(NB / "y_test_indiv.npy")

    log(f"Team: train={len(team_train)}, val={len(team_val)}, test={len(team_test)}, pos={y_test_team.mean():.4f}")
    log(f"Indiv: train={len(indiv_train)}, val={len(indiv_val)}, test={len(indiv_test)}, pos={y_test_indiv.mean():.4f}")

    results = {}
    for split_name, tr, va, te, yte in [
        ("team", team_train, team_val, team_test, y_test_team),
        ("indiv", indiv_train, indiv_val, indiv_test, y_test_indiv),
    ]:
        for thr in [2.0, 1.5]:
            log(f"\n=== RUN: split={split_name}, price_too_high > {thr} ===")
            m, n_iter, dt = run_experiment(split_name, tr, va, te, yte, df, cat_med, tab38, cat_tab38, clip_mat, id2row, thr)
            results[f"{split_name}__thr_{thr}"] = {
                "split": split_name,
                "price_too_high_threshold": thr,
                "metrics_test": m,
                "best_iter": int(n_iter),
                "training_time_sec": round(dt, 1),
            }
            # Save after each run (так если упадёт — частичные данные сохранятся)
            with OUT.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"{'config':<30}  {'ROC':>8}  {'PR':>8}  {'R@P':>8}")
    for k, v in results.items():
        m = v["metrics_test"]
        log(f"{k:<30}  {m['ROC']:>8.4f}  {m['PR']:>8.4f}  {m['R@P']:>8.4f}")
    log("")
    log(f"Сохранено: {OUT}")


if __name__ == "__main__":
    main()
