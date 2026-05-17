"""build_coauthor_proxy_artifacts.py — reproduction-обучение 4 артефактов соавторов
в Дианиной зоне для разблокировки live-CDSM v3 4-channel ensemble.

Что обучается (метки `_reproduction` явные в metadata):

    1. e5_pca.pkl       — sklearn PCA(n_components=25) на multilingual-e5-small
                           эмбеддингах train+val (138 788 × 384).
    2. clip_pca.pkl     — sklearn PCA(n_components=25) на CLIP ViT-B/32
                           эмбеддингах train+val (138 788 × 512).
    3. ftmff_catboost.cbm — reproduction Сониной Fusion (FT-MFF канал § 4.2 ВКР).
                             Архитектура из `packages/package_soni/2_notebooks/
                             03_multimodal_colab.ipynb` cell 8: 42 MY_FEATURES
                             (все df.columns кроме resolution/ItemID/SellerID) +
                             25 clip_pca + 25 t_pca = 92 features.
                             Гиперпараметры: iter=1000, depth=6, lr=0.05,
                             scale_pos_weight=spw, early_stop=50, random_seed=42.
    4. amm_thinker.pkl   — reproduction Альбининой MMD-Thinker v4 (AMM канал
                            § 4.3 ВКР). Архитектура: 3-mode CatBoost (Mode 1 numeric
                            only / Mode 2 +svd_text / Mode 3 +clip_emb) + meta-LR
                            на 9-мерном представлении [p1,p2,p3, |Δ|×3, ×3] +
                            sample_weighting через uncertainty Mode 1.

Все артефакты помечены полем `_reproduction_metadata` с указанием:
  - что это reproduction Дианой, не canonical from соавторов
  - какие гиперпараметры использовались
  - на каких данных обучено
  - какие метрики ожидаются (Fusion canonical: PR=0.7284, R@P=0.1077;
                              AMM canonical:    PR=0.7084, R@P=0.0276)

Запуск (~30-40 минут CPU):
    python3 Диана_ВКР_финал/scripts/build_coauthor_proxy_artifacts.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

ROOT = Path("/Users/diana/master-thesis-2025")
PKG = ROOT / "packages"
SPLITS = ROOT / "Диана_ВКР_финал" / "notebooks" / "team_splits"
NB_CDSM = ROOT / "Диана_ВКР_финал" / "notebooks" / "cdsm"
ART = ROOT / "Диана_ВКР_финал" / "counterfeit_service" / "artifacts" / "cdsm_v3"
CSV = ROOT / "Diana's folder" / "ml_ozon_ounterfeit_train.csv"
CLIP_PARQUET = ROOT / "counterfeit_service" / "clip_embeddings.parquet"
TEXT_E5 = ROOT / "text_e5_small.parquet"

SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true: np.ndarray, p: np.ndarray, name: str = "") -> dict:
    roc = float(roc_auc_score(y_true, p))
    pr = float(average_precision_score(y_true, p))
    prec, rec, _ = precision_recall_curve(y_true, p)
    m = prec >= 0.9
    rap = float(rec[m].max()) if m.any() else 0.0
    if name:
        log(f"  {name}: ROC={roc:.4f}  PR={pr:.4f}  R@P={rap:.4f}")
    return {"ROC": roc, "PR": pr, "R@P": rap}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # 1. Load splits, target, df
    # ─────────────────────────────────────────────────────────────────────
    log("Loading data and splits…")
    train_idx = np.load(SPLITS / "team_train_idx.npy")
    val_idx = np.load(SPLITS / "team_val_idx.npy")
    test_idx = np.load(SPLITS / "team_test_idx.npy")
    y_test = np.load(SPLITS / "y_test.npy").astype(np.int8)
    df = pd.read_csv(CSV, encoding="utf-8")
    df["ItemID"] = df["ItemID"].astype("int64")
    y_train = df.iloc[train_idx]["resolution"].astype("int8").values
    y_val = df.iloc[val_idx]["resolution"].astype("int8").values
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    log(f"  sizes train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}, spw={spw:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # 2. PCA-25 on e5 + CLIP — fit на train+val, transform на test
    # ─────────────────────────────────────────────────────────────────────
    log("[STEP 1/4] Loading raw embeddings…")
    e5_all = np.load(NB_CDSM / "e5_small_embeddings.npy")  # (197198, 384)
    log(f"  e5 embeddings: {e5_all.shape}")

    # CLIP — берём через ItemID merge
    clip_df = pd.read_parquet(CLIP_PARQUET)
    clip_dim = clip_df["embedding"].iloc[0].shape[0]
    clip_mat = np.stack(clip_df["embedding"].values).astype(np.float32)
    id2row = {int(i): r for r, i in enumerate(clip_df["ItemID"].values)}

    def merge_clip(idx: np.ndarray) -> np.ndarray:
        ids = df.iloc[idx]["ItemID"].astype("int64").values
        out = np.zeros((len(ids), clip_dim), dtype=np.float32)
        for i, iid in enumerate(ids):
            r = id2row.get(int(iid))
            if r is not None:
                out[i] = clip_mat[r]
        return out

    clip_train = merge_clip(train_idx)
    clip_val = merge_clip(val_idx)
    clip_test = merge_clip(test_idx)
    log(f"  CLIP merged: train {clip_train.shape}, val {clip_val.shape}, test {clip_test.shape}")

    # e5 — индексы (e5_all уже по строкам исходного df.iloc)
    e5_train = e5_all[train_idx]
    e5_val = e5_all[val_idx]
    e5_test = e5_all[test_idx]

    log("[STEP 1/4] Fitting PCA-25 on CLIP and e5 (fit on train+val)…")
    pca_clip = PCA(n_components=25, random_state=SEED)
    clip_tv = np.vstack([clip_train, clip_val])
    pca_clip.fit(clip_tv)
    clip_pca_train = pca_clip.transform(clip_train).astype(np.float32)
    clip_pca_val = pca_clip.transform(clip_val).astype(np.float32)
    clip_pca_test = pca_clip.transform(clip_test).astype(np.float32)
    log(f"  clip_pca explained_variance: {pca_clip.explained_variance_ratio_.sum():.3f}")
    joblib.dump(pca_clip, ART / "clip_pca.pkl")
    log(f"  ✓ Saved clip_pca.pkl ({(ART / 'clip_pca.pkl').stat().st_size:,} bytes)")

    pca_e5 = PCA(n_components=25, random_state=SEED)
    e5_tv = np.vstack([e5_train, e5_val])
    pca_e5.fit(e5_tv)
    e5_pca_train = pca_e5.transform(e5_train).astype(np.float32)
    e5_pca_val = pca_e5.transform(e5_val).astype(np.float32)
    e5_pca_test = pca_e5.transform(e5_test).astype(np.float32)
    log(f"  e5_pca explained_variance: {pca_e5.explained_variance_ratio_.sum():.3f}")
    joblib.dump(pca_e5, ART / "e5_pca.pkl")
    log(f"  ✓ Saved e5_pca.pkl ({(ART / 'e5_pca.pkl').stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────────────
    # 3. Sonya Fusion (FT-MFF channel) — reproduction
    # ─────────────────────────────────────────────────────────────────────
    log("[STEP 2/4] Reproducing Sonya Fusion (FT-MFF)…")
    MY_FEATURES = [c for c in df.columns if c not in ["resolution", "ItemID", "SellerID"]]
    MY_CATS = df[MY_FEATURES].select_dtypes(include=["object"]).columns.tolist()
    MY_NUMS = [c for c in MY_FEATURES if c not in MY_CATS]
    log(f"  MY_FEATURES = {len(MY_FEATURES)} (numeric={len(MY_NUMS)}, cat={len(MY_CATS)})")

    def prep_tab(idx: np.ndarray) -> pd.DataFrame:
        X = df.iloc[idx][MY_FEATURES].copy().reset_index(drop=True)
        if MY_NUMS:
            X[MY_NUMS] = X[MY_NUMS].fillna(0)
        for c in MY_CATS:
            X[c] = X[c].fillna("nan").astype(str)
        return X

    Xt = prep_tab(train_idx)
    Xv = prep_tab(val_idx)
    Xs = prep_tab(test_idx)
    clip_cols = [f"clip_pca_{i}" for i in range(25)]
    text_cols = [f"t_pca_{i}" for i in range(25)]
    Xt = pd.concat([Xt, pd.DataFrame(clip_pca_train, columns=clip_cols),
                    pd.DataFrame(e5_pca_train, columns=text_cols)], axis=1)
    Xv = pd.concat([Xv, pd.DataFrame(clip_pca_val, columns=clip_cols),
                    pd.DataFrame(e5_pca_val, columns=text_cols)], axis=1)
    Xs = pd.concat([Xs, pd.DataFrame(clip_pca_test, columns=clip_cols),
                    pd.DataFrame(e5_pca_test, columns=text_cols)], axis=1)
    log(f"  Xt shape (Fusion): {Xt.shape}")

    fusion_params = dict(
        iterations=1000, depth=6, learning_rate=0.05,
        eval_metric="AUC", scale_pos_weight=spw,
        early_stopping_rounds=50, random_seed=SEED, verbose=200,
    )
    fusion_model = CatBoostClassifier(**fusion_params)
    log(f"  Training Fusion CatBoost...")
    t0 = time.time()
    fusion_model.fit(Xt, y_train, cat_features=MY_CATS, eval_set=(Xv, y_val), use_best_model=True)
    log(f"  Trained in {time.time()-t0:.0f}s, best_iter={fusion_model.tree_count_}")
    proba_fusion = fusion_model.predict_proba(Xs)[:, 1].astype("float64")
    m_fusion = metrics(y_test, proba_fusion, "Fusion reproduction (test=58410)")
    log(f"  Canonical (Sonya original): PR=0.7284  R@P=0.1077  ROC=0.9522")
    fusion_model.save_model(str(ART / "ftmff_catboost.cbm"))
    log(f"  ✓ Saved ftmff_catboost.cbm ({(ART / 'ftmff_catboost.cbm').stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────────────
    # 4. Albina MMD-Thinker v4 (AMM channel) — reproduction
    # ─────────────────────────────────────────────────────────────────────
    log("[STEP 3/4] Reproducing Albina MMD-Thinker v4 (AMM)…")
    # DROP per Альбинин ноут cell 18
    DROP_AMM = {"id", "ItemID", "SellerID", "name_rus", "description",
                "brand_name", "CommercialTypeName4", "resolution"}
    NUM_COLS = [c for c in df.columns if c not in DROP_AMM
                and df[c].dtype in ["float64", "int64", "float32"]]
    log(f"  AMM numeric columns (Mode 1 base): {len(NUM_COLS)}")

    # SVD-50 на text — обучим простой sklearn TruncatedSVD на TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    df["text_full"] = (df["name_rus"].fillna("").astype(str) + " "
                        + df["description"].fillna("").astype(str) + " "
                        + df["brand_name"].fillna("").astype(str)).str.slice(0, 1024)
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    text_tv = df.iloc[np.concatenate([train_idx, val_idx])]["text_full"].values
    text_te = df.iloc[test_idx]["text_full"].values
    Xtf_tv = tfidf.fit_transform(text_tv)
    Xtf_te = tfidf.transform(text_te)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    svd_tv = svd.fit_transform(Xtf_tv).astype(np.float32)
    svd_te = svd.transform(Xtf_te).astype(np.float32)
    n_tr = len(train_idx)
    svd_train = svd_tv[:n_tr]
    svd_val = svd_tv[n_tr:]
    svd_test = svd_te
    log(f"  SVD-50 fitted: train {svd_train.shape}, val {svd_val.shape}, test {svd_test.shape}")

    def build_mode(idx, with_svd: bool, with_clip: bool, name_suffix: str) -> pd.DataFrame:
        X = df.iloc[idx][NUM_COLS].fillna(0).reset_index(drop=True)
        if with_svd:
            svd_df = svd_train if name_suffix == "train" else (svd_val if name_suffix == "val" else svd_test)
            X = pd.concat([X, pd.DataFrame(svd_df, columns=[f"svd_{i}" for i in range(50)])], axis=1)
        if with_clip:
            clip_df_ = clip_pca_train if name_suffix == "train" else (clip_pca_val if name_suffix == "val" else clip_pca_test)
            X = pd.concat([X, pd.DataFrame(clip_df_, columns=[f"clip_pca_{i}" for i in range(25)])], axis=1)
        return X

    amm_params = dict(
        iterations=600, depth=6, learning_rate=0.05,
        eval_metric="AUC", scale_pos_weight=spw,
        early_stopping_rounds=40, random_seed=SEED, verbose=0,
    )
    amm_modes = {}
    amm_test_probas = {}
    for mode_name, with_svd, with_clip in [
        ("mode1", False, False), ("mode2", True, False), ("mode3", True, True),
    ]:
        log(f"  Training AMM {mode_name}...")
        Xt_m = build_mode(train_idx, with_svd, with_clip, "train")
        Xv_m = build_mode(val_idx, with_svd, with_clip, "val")
        Xs_m = build_mode(test_idx, with_svd, with_clip, "test")
        m = CatBoostClassifier(**amm_params)
        m.fit(Xt_m, y_train, eval_set=(Xv_m, y_val), use_best_model=True)
        amm_modes[f"{mode_name}_catboost"] = m
        amm_test_probas[mode_name] = m.predict_proba(Xs_m)[:, 1]
        # OOF на train+val для meta-LR
        # Для простоты: predict_proba на train для построения meta-features (это leak, но Альбинин ноут делает похоже)
        # Альтернатива — отдельный StratifiedKFold OOF, но это удвоит время. Reproduction-цель — close но не bit-identical.
        log(f"    {mode_name}: trained, best_iter={m.tree_count_}, "
            f"test PR={average_precision_score(y_test, amm_test_probas[mode_name]):.4f}")

    # Meta-LR на 9-dim meta-vector
    # Используем val (vs train) для построения meta — это honest split-protocol.
    log("  Training AMM meta-LR on val-OOF probas…")
    val_probas = {}
    for mode_name, with_svd, with_clip in [("mode1", False, False), ("mode2", True, False), ("mode3", True, True)]:
        Xv_m = build_mode(val_idx, with_svd, with_clip, "val")
        val_probas[mode_name] = amm_modes[f"{mode_name}_catboost"].predict_proba(Xv_m)[:, 1]
    p1v, p2v, p3v = val_probas["mode1"], val_probas["mode2"], val_probas["mode3"]
    meta_train = np.column_stack([
        p1v, p2v, p3v,
        np.abs(p1v - p2v), np.abs(p2v - p3v), np.abs(p1v - p3v),
        p1v * p2v, p1v * p3v, p2v * p3v,
    ])
    sw = 1.0 + (1.0 - 2.0 * np.abs(p1v - 0.5))
    meta_lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=SEED)
    meta_lr.fit(meta_train, y_val, sample_weight=sw)

    # Apply meta-LR on test
    p1t, p2t, p3t = amm_test_probas["mode1"], amm_test_probas["mode2"], amm_test_probas["mode3"]
    meta_test = np.column_stack([
        p1t, p2t, p3t,
        np.abs(p1t - p2t), np.abs(p2t - p3t), np.abs(p1t - p3t),
        p1t * p2t, p1t * p3t, p2t * p3t,
    ])
    proba_amm = meta_lr.predict_proba(meta_test)[:, 1]
    m_amm = metrics(y_test, proba_amm, "AMM-Thinker v4 reproduction (test=58410)")
    log(f"  Canonical (Albina original): PR=0.7084  R@P=0.0276  ROC=0.9545")

    amm_pkl = {
        "mode1_catboost": amm_modes["mode1_catboost"],
        "mode2_catboost": amm_modes["mode2_catboost"],
        "mode3_catboost": amm_modes["mode3_catboost"],
        "meta_lr": meta_lr,
        "feature_cols": {
            "mode1": NUM_COLS,
            "mode2": NUM_COLS + [f"svd_{i}" for i in range(50)],
            "mode3": NUM_COLS + [f"svd_{i}" for i in range(50)] + [f"clip_pca_{i}" for i in range(25)],
        },
        "sample_weight_uncertainty_threshold": 0.5,
        "tfidf_vectorizer": tfidf,
        "svd": svd,
        "_reproduction_metadata": {
            "by": "Diana Mkoian, 2026-05-12",
            "canonical_source": "packages/package_albina/2_notebooks/safe_ozon_v4_mmd__1_.ipynb",
            "note": "Reproduction-обучение в Дианиной зоне для разблокировки live-CDSM v3. "
                    "Не bit-точное воспроизведение Альбининой модели; ожидается близость метрик "
                    "PR≈0.7084, R@P≈0.0276 (canonical Альбинин).",
        },
    }
    joblib.dump(amm_pkl, ART / "amm_thinker.pkl")
    log(f"  ✓ Saved amm_thinker.pkl ({(ART / 'amm_thinker.pkl').stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────────────
    # 5. Verification summary
    # ─────────────────────────────────────────────────────────────────────
    log("[STEP 4/4] Final verification summary")
    summary = {
        "rebuilt_at": "2026-05-12",
        "by": "Diana Mkoian — reproduction-обучение для разблокировки live-CDSM v3 4-channel ensemble",
        "artifacts": {
            "clip_pca.pkl": {
                "type": "sklearn.PCA(n_components=25)",
                "trained_on": "CLIP train+val 138 788 × 512",
                "explained_variance": float(pca_clip.explained_variance_ratio_.sum()),
                "canonical_source": "packages/package_soni/2_notebooks/03_multimodal_colab.ipynb cell 4",
            },
            "e5_pca.pkl": {
                "type": "sklearn.PCA(n_components=25)",
                "trained_on": "multilingual-e5-small train+val 138 788 × 384",
                "explained_variance": float(pca_e5.explained_variance_ratio_.sum()),
                "canonical_source": "packages/package_soni/2_notebooks/03_multimodal_colab.ipynb cell 4",
            },
            "ftmff_catboost.cbm": {
                "type": "CatBoostClassifier",
                "features": "92-dim (42 MY_FEATURES + 25 clip_pca + 25 t_pca)",
                "hyperparams": fusion_params,
                "metrics_test": m_fusion,
                "canonical_metrics_Sonya": {"PR": 0.7284, "RaP": 0.1077, "ROC": 0.9522},
                "canonical_source": "packages/package_soni/2_notebooks/03_multimodal_colab.ipynb cell 8",
            },
            "amm_thinker.pkl": {
                "type": "dict with 3 CatBoostClassifier + meta-LR + sample_weights",
                "modes": "Mode 1 numeric (DROP_AMM) / Mode 2 +svd / Mode 3 +clip_pca",
                "hyperparams": amm_params,
                "metrics_test": m_amm,
                "canonical_metrics_Albina": {"PR": 0.7084, "RaP": 0.0276, "ROC": 0.9545},
                "canonical_source": "packages/package_albina/2_notebooks/safe_ozon_v4_mmd__1_.ipynb cell 18+",
            },
        },
        "_disclaimer": (
            "Эти 4 артефакта — Дианина reproduction для production-deploy. Они НЕ bit-точно "
            "воспроизводят канонические probas из packages/*/3_probas_team_split/*.npy. "
            "Канонические артефакты соавторов финализируются в индивидуальных ВКР § 4.2 (С. Красовская) "
            "и § 4.3 (А. Бахтиарова). Reproduction-метрики ожидаются в пределах ±0.02 PR от canonical."
        ),
    }
    with (ART / "coauthor_reproduction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    log(f"  ✓ Saved coauthor_reproduction_summary.json")
    log("")
    log("DONE. All 4 reproduction artifacts saved to:")
    log(f"  {ART}")


if __name__ == "__main__":
    main()
