"""cdsm_v3_feature_importances.py — feature importance CDSM v3.

Standalone-скрипт: загружает 15 fold-моделей CDSM Mode 1/2/3 из artifacts,
вычисляет `CatBoostClassifier.get_feature_importance(type='PredictionValuesChange')`,
усредняет по 5 fold-моделям внутри каждого Mode и присваивает семантические
имена признакам, реконструируя композицию `assemble_features()` из ноута 08
(cells 5–15: A+B+C+D для Mode 1, +E+F для Mode 2, +G+H+I для Mode 3).

Запуск (из корня репо):
    python3 Диана_ВКР_финал/scripts/cdsm_v3_feature_importances.py

Выход:
    counterfeit_service/artifacts/cdsm_v3/feature_importances.json
    (3 mode'а × {top15, all_importances, n_features})

Может использоваться в § 5.5 ВКР для замены устаревшей Таблицы 5.6 (которая
описывает stacking_cb, а не финальный CDSM v3 4-channel ensemble).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path("/Users/diana/master-thesis-2025")
ART = ROOT / "Диана_ВКР_финал" / "counterfeit_service" / "artifacts" / "cdsm_v3"
TEAM_META = ROOT / "Диана_ВКР_финал" / "notebooks" / "team_splits" / "team_split_meta.json"

# Восстановление имён признаков по группам (cell 5–13 ноута 08)
with TEAM_META.open() as f:
    meta = json.load(f)
TAB38 = meta["team_feature_cols"]  # 38 командных tab columns

B_NAMES = ["brand_freq", "brand_fake_rate", "brand_mean_rating5", "brand_mean_gmv30"]
C_NAMES = ["price_ratio", "price_too_low", "price_too_high"]
D_NAMES = ["brand_exact", "brand_fuzzy", "typosquat"]
E_NAMES = [f"tfidf_svd_{i}" for i in range(50)]
F_NAMES = [f"e5_pca_{i}" for i in range(25)]
G_NAMES = [f"clip_pca_{i}" for i in range(25)]
H_NAMES = ["clip_dist_centroid", "clip_norm", "clip_dist_centered", "clip_dist_norm"]
I_NAMES = ["kl_text_tab"]

GROUP_NAMES = {
    "A": TAB38, "B": B_NAMES, "C": C_NAMES, "D": D_NAMES,
    "E": E_NAMES, "F": F_NAMES, "G": G_NAMES, "H": H_NAMES, "I": I_NAMES,
}

MODE_GROUPS = {
    "mode1": ["A", "B", "C", "D"],                          # 48 features
    "mode2": ["A", "B", "C", "D", "E", "F"],                # 123 features
    "mode3": ["A", "B", "C", "D", "E", "F", "G", "H", "I"], # 153 features
}


def build_feature_names(groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        out.extend(GROUP_NAMES[g])
    return out


def main() -> None:
    print("=== CDSM v3 Feature Importance (mean across 5 fold-моделей per mode) ===\n")
    summary: dict = {}

    for mode_name, groups in MODE_GROUPS.items():
        names = build_feature_names(groups)
        fold_imps: list[np.ndarray] = []
        for fold in range(5):
            cb = CatBoostClassifier()
            cb.load_model(str(ART / f"cdsm_{mode_name}_fold{fold}.cbm"))
            imp = cb.get_feature_importance(type="PredictionValuesChange")
            fold_imps.append(imp)

        mean_imp = np.mean(fold_imps, axis=0)
        std_imp = np.std(fold_imps, axis=0)
        n_features = len(mean_imp)

        if n_features != len(names):
            print(f"⚠ {mode_name}: model has {n_features} features, names list has {len(names)} — using positional indices instead")
            names = [f"feat_{i}" for i in range(n_features)]

        # Top-15
        order = np.argsort(mean_imp)[::-1]
        top15 = [(names[i], float(mean_imp[i]), float(std_imp[i])) for i in order[:15]]

        print(f"\n── {mode_name.upper()} ({n_features} features, fold-mean ± fold-std) ──")
        print(f"  {'#':>3}  {'признак':<30}  {'mean':>8}  {'std':>8}  {'группа':<6}")
        # Группа для каждого признака
        feat_group_map: dict[str, str] = {}
        for g in groups:
            for n in GROUP_NAMES[g]:
                feat_group_map[n] = g
        for rank, (n, mean_v, std_v) in enumerate(top15, 1):
            g_letter = feat_group_map.get(n, "?")
            print(f"  {rank:>3}. {n:<30}  {mean_v:>8.3f}  {std_v:>8.3f}  {g_letter:<6}")

        summary[mode_name] = {
            "n_features": n_features,
            "groups": groups,
            "top15": [{"name": n, "mean": m, "std": s} for n, m, s in top15],
            "all_importances": {n: float(v) for n, v in zip(names, mean_imp.tolist())},
            "all_std": {n: float(v) for n, v in zip(names, std_imp.tolist())},
        }

    out_path = ART / "feature_importances.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Сохранён {out_path}")
    print(f"  Размер: {out_path.stat().st_size:,} bytes")

    # Опционально CSV для удобства вставки в Word
    rows = []
    for mode_name, info in summary.items():
        for rank, item in enumerate(info["top15"], 1):
            g = "?"
            for grp_letter in info["groups"]:
                if item["name"] in GROUP_NAMES[grp_letter]:
                    g = grp_letter
                    break
            rows.append({
                "mode": mode_name, "rank": rank,
                "feature": item["name"], "mean_importance": round(item["mean"], 3),
                "std": round(item["std"], 3), "group": g,
            })
    df = pd.DataFrame(rows)
    csv_path = ART / "feature_importances_top15.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV сохранён {csv_path}  ({csv_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
