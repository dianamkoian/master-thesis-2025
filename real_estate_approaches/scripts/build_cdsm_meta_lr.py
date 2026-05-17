"""build_cdsm_meta_lr.py — standalone-восстановление cdsm_meta_lr.pkl.

Воспроизводит cell 20 `train_meta(...)` из ноута 08_cross_domain_synthesis_v2.ipynb
без перезапуска самого ноута. Использует уже сохранённые OOF-probas Mode 1/2/3.

Корень потребности: cell 30 крашнулся на `NameError: lr is not defined`
(реальное имя переменной в ноуте — `lr_meta`), поэтому при папермилл-прогоне
производственный артефакт meta-LR не сохранился.

Сохраняемый объект — `sklearn.pipeline.Pipeline` со StandardScaler и LogReg
внутри, чтобы CDSMPipeline.predict_proba_one в `_cdsm_loaders.py` мог
применить scaler автоматически (Pipeline.predict_proba делает scaler + lr
за один вызов).

Запуск (из корня репо):
    python3 real_estate_approaches/scripts/build_cdsm_meta_lr.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PKG_DIANA = ROOT / "packages" / "package_diana" / "3_probas_team_split"
NB_CDSM = ROOT / "real_estate_approaches" / "notebooks" / "cdsm"
ARTIFACTS = ROOT / "counterfeit_service" / "artifacts" / "cdsm_v3"
CSV_PATH = ROOT / "data" / "ml_ozon_ounterfeit_train.csv"

SEED = 42


def meta_features(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """9-мерное расширенное представление (canonical из cell 20 ноута 08).

    Порядок диффов: |p1-p2|, |p2-p3|, |p1-p3| (НЕ |p1-p3|, |p2-p3|).
    Согласован с `_cdsm_loaders.py:206-208` после фикса.
    """
    return np.column_stack([
        p1, p2, p3,
        np.abs(p1 - p2), np.abs(p2 - p3), np.abs(p1 - p3),
        p1 * p2, p1 * p3, p2 * p3,
    ])


def rap_at_p90(y_true: np.ndarray, p: np.ndarray) -> float:
    prec, rec, _ = precision_recall_curve(y_true, p)
    mask = prec >= 0.9
    return float(rec[mask].max()) if mask.any() else 0.0


def main() -> None:
    print("=== build_cdsm_meta_lr — standalone recovery после crash cell 30 ===\n")

    # 1) OOF-probas (train+val, n=138 788)
    oof_m1 = np.load(NB_CDSM / "oof_mode1.npy").astype(np.float64)
    oof_m2 = np.load(NB_CDSM / "oof_mode2.npy").astype(np.float64)
    oof_m3 = np.load(NB_CDSM / "oof_mode3.npy").astype(np.float64)
    test_m1 = np.load(NB_CDSM / "test_mode1.npy").astype(np.float64)
    test_m2 = np.load(NB_CDSM / "test_mode2.npy").astype(np.float64)
    test_m3 = np.load(NB_CDSM / "test_mode3.npy").astype(np.float64)
    print(f"OOF probas (train+val): shapes ({oof_m1.shape}, {oof_m2.shape}, {oof_m3.shape})")
    print(f"Test probas:            shapes ({test_m1.shape}, {test_m2.shape}, {test_m3.shape})")

    # 2) y_tv — из csv по train_idx + val_idx (cell 4 ноута 08)
    train_idx = np.load(PKG_DIANA / "team_train_idx.npy")
    val_idx = np.load(PKG_DIANA / "team_val_idx.npy")
    y_test = np.load(PKG_DIANA / "y_test_team.npy").astype(np.int8)
    df = pd.read_csv(CSV_PATH, encoding="utf-8", usecols=["resolution"])
    y_train = df.iloc[train_idx]["resolution"].astype("int8").values
    y_val = df.iloc[val_idx]["resolution"].astype("int8").values
    y_tv = np.concatenate([y_train, y_val])
    assert len(y_tv) == len(oof_m1), f"y_tv {len(y_tv)} vs OOF {len(oof_m1)}"
    print(f"y_tv (для обучения meta-LR): {y_tv.shape}, positives={y_tv.sum()} ({y_tv.mean():.4f})")
    print(f"y_test (для sanity-check):   {y_test.shape}, positives={y_test.sum()} ({y_test.mean():.4f})\n")

    # 3) Meta-vectors + sample_weight (формула Wu & Fu MMD-Thinker, 2025)
    m_tv = meta_features(oof_m1, oof_m2, oof_m3)
    m_te = meta_features(test_m1, test_m2, test_m3)
    sw = 1.0 + (1.0 - 2.0 * np.abs(oof_m1 - 0.5))  # больший вес на uncertain Mode 1

    # 4) Pipeline(scaler, lr) — обучение
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=SEED,
        )),
    ])
    pipe.fit(m_tv, y_tv, lr__sample_weight=sw)

    # 5) Sanity-метрики на test (с known headline из cdsm_v3_summary.json::cdsm)
    p_te = pipe.predict_proba(m_te)[:, 1]
    roc = roc_auc_score(y_test, p_te)
    pr = average_precision_score(y_test, p_te)
    rap = rap_at_p90(y_test, p_te)
    print(f"CDSM standalone (test): ROC={roc:.4f}  PR={pr:.4f}  R@P={rap:.4f}")
    print(f"Заявлено в cdsm_v3_summary.json::cdsm: ROC=0.9530  PR=0.7136  R@P=0.1271")
    delta_pr = pr - 0.7136
    delta_rap = rap - 0.1271
    print(f"Δ vs headline: PR={delta_pr:+.4f}  R@P={delta_rap:+.4f}\n")

    # 6) Лог coefs (отладка)
    feat_names = ["p1", "p2", "p3", "|p1-p2|", "|p2-p3|", "|p1-p3|", "p1*p2", "p1*p3", "p2*p3"]
    coefs = dict(zip(feat_names, pipe.named_steps["lr"].coef_[0].round(3).tolist()))
    print(f"Meta-LR coefs (после Pipeline-fit): {coefs}")
    print(f"Meta-LR intercept: {pipe.named_steps['lr'].intercept_[0]:.4f}\n")

    # 7) Сохранение
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "cdsm_meta_lr.pkl"
    joblib.dump(pipe, out_path)
    print(f"✓ Сохранён {out_path}  ({out_path.stat().st_size} bytes)")
    print()
    print("Production-сервис: при загрузке `cdsm_v3_4channel`")
    print("CDSMPipeline.predict_proba_one теперь будет вызывать `pipe.predict_proba(meta)`,")
    print("где Pipeline сам применит StandardScaler перед LR.")


if __name__ == "__main__":
    main()
