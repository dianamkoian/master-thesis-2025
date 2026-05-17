"""
Дипломный эксперимент § 4.4.9.7: verdict-conditioned reasoning от
Qwen2.5-1.5B-Instruct на 50 borderline-объектах командного теста.

Вердикт берётся от 4-way ансамбля `grid_best_pr` = 0,25·fusion + 0,50·M2-FE+
+ 0,25·albina (PR-AUC = 0,7556 на командном тесте, см.
`scripts/sonya_pilot/stacking_pilot_4domains_v2.json`).

Выходы:
  artifacts_lmm/verdict_conditioned_reasoning_50.csv  — полная таблица
  artifacts_lmm/verdict_conditioned_reasoning_50.md   — для вставки в § 4.4.9.7

Запуск:
  cd Диана_ВКР_финал
  python scripts/llm_reasoning_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/diana/master-thesis-2025")
DIPLOMA = ROOT / "Диана_ВКР_финал"
SCRIPTS = DIPLOMA / "scripts"
NOTEBOOKS = DIPLOMA / "notebooks"
SERVICE_APP = DIPLOMA / "counterfeit_service" / "app"
OUT_DIR = NOTEBOOKS / "artifacts_lmm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Импортируем LLMExplainer из сервисного пакета — единая prompt-логика.
sys.path.insert(0, str(SERVICE_APP.parent))  # counterfeit_service/
from app.llm_explainer import LLMExplainer  # noqa: E402

import os as _os_env
SEED = 42
N_DEMO = int(_os_env.getenv("DEMO_N", "200"))
BORDERLINE_LO = float(_os_env.getenv("DEMO_BORDER_LO", "0.30"))
BORDERLINE_HI = float(_os_env.getenv("DEMO_BORDER_HI", "0.70"))

print("=" * 70)
print("Verdict-conditioned reasoning demo (§ 4.4.9.7)")
print("=" * 70)

# 1. Загружаем 4-way ансамбль и индексы.
proba = np.load(SCRIPTS / "sonya_pilot" / "test_proba_4way_grid_best_pr.npy")
y_test = np.load(NOTEBOOKS / "team_splits" / "y_test.npy")
test_idx = np.load(NOTEBOOKS / "team_splits" / "team_test_idx.npy")
print(f"ensemble proba: n={len(proba)}, mean={proba.mean():.4f}, "
      f"borderline ({BORDERLINE_LO},{BORDERLINE_HI}): "
      f"{int(((proba > BORDERLINE_LO) & (proba < BORDERLINE_HI)).sum())} objects")

# 2. Отбираем 50 borderline разнообразно по proba.
mask = (proba > BORDERLINE_LO) & (proba < BORDERLINE_HI)
candidate_positions = np.where(mask)[0]
rng = np.random.default_rng(SEED)
# stratified-like: бины по proba, выбираем равномерно по бинам
n_bins = 5
bin_edges = np.linspace(BORDERLINE_LO, BORDERLINE_HI, n_bins + 1)
selected_positions = []
per_bin = N_DEMO // n_bins
for b in range(n_bins):
    in_bin = candidate_positions[
        (proba[candidate_positions] >= bin_edges[b])
        & (proba[candidate_positions] < bin_edges[b + 1])
    ]
    take = rng.choice(in_bin, size=min(per_bin, len(in_bin)), replace=False)
    selected_positions.extend(take.tolist())
# дополним до 50 случайными borderline, если что-то выпало
extra_needed = N_DEMO - len(selected_positions)
if extra_needed > 0:
    pool = np.setdiff1d(candidate_positions, np.array(selected_positions))
    selected_positions.extend(rng.choice(pool, size=extra_needed, replace=False).tolist())
selected_positions = np.array(sorted(selected_positions))[:N_DEMO]
print(f"selected {len(selected_positions)} borderline objects across {n_bins} proba-bins")

# 3. Достаём фичи карточек из train CSV.
DATA_CSV = ROOT / "Diana's folder" / "ml_ozon_ounterfeit_train.csv"
df = pd.read_csv(DATA_CSV, encoding="utf-8")
COLS_NEEDED = [
    "ItemID", "name_rus", "description", "brand_name", "CommercialTypeName4",
    "PriceDiscounted", "item_time_alive", "seller_time_alive",
    "item_count_sales30", "item_count_returns30",
]
df_sub = df.iloc[test_idx[selected_positions]][COLS_NEEDED].reset_index(drop=True)
df_sub["true_label"] = y_test[selected_positions]
df_sub["ensemble_proba"] = proba[selected_positions]
print(f"loaded card features for {len(df_sub)} objects")

# 4. Загружаем LLM и генерируем reasoning.
print(f"\nloading Qwen2.5-1.5B-Instruct ...")
llm = LLMExplainer()
t0 = time.time()
reasonings: list[str] = []
for i, row in df_sub.iterrows():
    card = {
        "name": row["name_rus"],
        "description": row["description"] if pd.notna(row["description"]) else "",
        "brand": row["brand_name"] if pd.notna(row["brand_name"]) else "",
        "CommercialTypeName4": row["CommercialTypeName4"],
        "PriceDiscounted": row["PriceDiscounted"],
        "item_time_alive": row["item_time_alive"],
        "seller_time_alive": row["seller_time_alive"],
        "item_count_sales30": row["item_count_sales30"],
        "item_count_returns30": row["item_count_returns30"],
    }
    try:
        r = llm.explain(card, float(row["ensemble_proba"]))
    except Exception as e:  # noqa: BLE001
        r = f"[ERROR] {type(e).__name__}: {e}"
    reasonings.append(r)
    if (i + 1) % 5 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(df_sub) - i - 1) / max(rate, 1e-6)
        print(f"  [{i+1:2d}/{len(df_sub)}]  elapsed={elapsed:5.1f}s  "
              f"ETA={eta:5.1f}s  proba={row['ensemble_proba']:.3f}")

df_sub["reasoning"] = reasonings

# 5. Сохраняем.
out_csv = OUT_DIR / f"verdict_conditioned_reasoning_{N_DEMO}.csv"
df_sub.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nsaved csv: {out_csv}")

# Агрегированная статистика по бинам — для дипломной таблицы § 4.4.9.7.
stats_lines = ["# Статистика по бинам proba 4-way ансамбля", ""]
stats_lines.append(f"Всего объектов: {len(df_sub)}; диапазон proba: "
                   f"[{df_sub['ensemble_proba'].min():.3f}; {df_sub['ensemble_proba'].max():.3f}]; "
                   f"доля контрафакта (true_label=1): "
                   f"{(df_sub['true_label']==1).mean()*100:.1f}%.")
stats_lines.append("")
stats_lines.append("| Бин proba | n | true_counterfeit | true_original | avg len reasoning |")
stats_lines.append("|---|---|---|---|---|")
bins = np.linspace(BORDERLINE_LO, BORDERLINE_HI, 6)  # 5 интервалов
for i in range(len(bins) - 1):
    lo_, hi_ = bins[i], bins[i+1]
    mask_b = (df_sub["ensemble_proba"] >= lo_) & (df_sub["ensemble_proba"] < hi_ + (1e-9 if i == len(bins)-2 else 0))
    sub = df_sub[mask_b]
    if len(sub) == 0:
        continue
    n_c = int((sub["true_label"] == 1).sum())
    n_o = int((sub["true_label"] == 0).sum())
    avg_len = int(sub["reasoning"].astype(str).str.len().mean())
    stats_lines.append(f"| [{lo_:.2f}; {hi_:.2f}) | {len(sub)} | {n_c} | {n_o} | {avg_len} |")
out_stats = OUT_DIR / f"verdict_conditioned_reasoning_{N_DEMO}_stats.md"
out_stats.write_text("\n".join(stats_lines), encoding="utf-8")
print(f"saved stats: {out_stats}")

out_md = OUT_DIR / f"verdict_conditioned_reasoning_{N_DEMO}.md"
md_lines = [
    f"# Verdict-conditioned reasoning: {N_DEMO} borderline-примеров (§ 4.4.9.7)",
    "",
    f"Вердикт: 4-way ансамбль `grid_best_pr` (PR-AUC = 0,7556 на командном тесте, "
    f"n={len(proba)}). Borderline-зона ({BORDERLINE_LO}, {BORDERLINE_HI}). "
    f"Объектов отобрано: {len(df_sub)}. "
    f"Reasoning: Qwen2.5-1.5B-Instruct, zero-shot, MPS, max_new_tokens=200.",
    "",
]
for i, row in df_sub.iterrows():
    md_lines.append(f"## Пример {i+1}. ItemID={row['ItemID']}")
    md_lines.append("")
    md_lines.append(f"- **Название:** {row['name_rus']}")
    md_lines.append(f"- **Бренд:** {row['brand_name']}")
    md_lines.append(f"- **Категория:** {row['CommercialTypeName4']}")
    md_lines.append(f"- **Цена:** {row['PriceDiscounted']}")
    md_lines.append(f"- **Возраст товара/продавца:** "
                    f"{row['item_time_alive']} / {row['seller_time_alive']} дн.")
    md_lines.append(f"- **Прогноз ансамбля:** {row['ensemble_proba']:.3f}")
    md_lines.append(f"- **Истинная метка:** "
                    f"{'контрафакт' if row['true_label']==1 else 'оригинал'}")
    md_lines.append("")
    md_lines.append(f"**Reasoning (Qwen2.5):** {row['reasoning']}")
    md_lines.append("")
out_md.write_text("\n".join(md_lines), encoding="utf-8")
print(f"saved md:  {out_md}")
print(f"total runtime: {time.time()-t0:.1f}s")
