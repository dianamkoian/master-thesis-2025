"""
Live-демо verdict-conditioned reasoning в сервисе без поднятия Docker.

Делает три тестовых запроса к `ReasoningPredictor` (через тот же контракт, что
и FastAPI-endpoint /predict). Печатает полный JSON-ответ для каждого — включая
поля `reasoning` и `reasoning_mode`.

Запуск:
    cd service/counterfeit_service
    python scripts/demo_reasoning_in_service.py

Время: ~30 с на загрузку D2V+CatBoost + ~15 с на загрузку Qwen2.5 + ~5 с на каждый
borderline-запрос. Confident-запросы идут мгновенно (без LLM).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # counterfeit_service/
sys.path.insert(0, str(ROOT))

from app.predictor_with_reasoning import ReasoningPredictor

# Используем stub-predictor вместо D2VCatBoost: D2V в текущем окружении даёт
# несовместимый CLIP-вывод (38400 фич вместо 512) — это отдельная проблема
# сервисного бэкбоуна, не связанная с LLM-каналом. Для демонстрации
# verdict-conditioned reasoning достаточно любого inner predictor'а.

# ───────────────────────────────────────────────────────────────────────────
# Тестовые карточки. Выбраны намеренно с разных «пугающих уровней», чтобы
# покрыть borderline (LLM-канал работает) и confident (LLM пропускается).
# ───────────────────────────────────────────────────────────────────────────

# Тестовая картинка. Ищем сначала в новой структуре, затем legacy.
_DATA_IMG_CANDIDATES = [
    ROOT.parent.parent / "data" / "ml_ozon_counterfeit_test_images" / "10.png",
    Path("/Users/diana/master-thesis-2025/data/ml_ozon_counterfeit_test_images/10.png"),
]
DATA_IMG = next((p for p in _DATA_IMG_CANDIDATES if p.exists()), _DATA_IMG_CANDIDATES[0])

TEST_CARDS = [
    {
        "label": "[blocking ожидается] Дешёвый iPhone у молодого продавца",
        "name": "Смартфон iPhone 15 Pro Max 256GB оригинал",
        "description": "Новый запечатанный, Apple iPhone 15 Pro Max",
        "brand": "Apple",
        "tab_inputs": {
            "CommercialTypeName4": "Смартфоны",
            "PriceDiscounted": 8500.0,
            "item_time_alive": 3.0,
            "seller_time_alive": 12.0,
            "item_count_sales30": 1.0,
            "item_count_returns30": 1.0,
        },
    },
    {
        "label": "[borderline ожидается] Брендовая косметика, средние сигналы",
        "name": "Помада L'Oreal Color Riche 124 розовая",
        "description": "Стойкая увлажняющая помада, оригинальная упаковка",
        "brand": "L'Oreal",
        "tab_inputs": {
            "CommercialTypeName4": "Косметика",
            "PriceDiscounted": 290.0,
            "item_time_alive": 45.0,
            "seller_time_alive": 90.0,
            "item_count_sales30": 11.0,
            "item_count_returns30": 3.0,
        },
    },
    {
        "label": "[confident_negative ожидается] Зрелый продавец, обычный товар",
        "name": "Кружка керамическая 350 мл белая",
        "description": "Простая столовая кружка для повседневного использования",
        "brand": "",
        "tab_inputs": {
            "CommercialTypeName4": "Посуда",
            "PriceDiscounted": 250.0,
            "item_time_alive": 540.0,
            "seller_time_alive": 1200.0,
            "item_count_sales30": 87.0,
            "item_count_returns30": 2.0,
        },
    },
]
for c in TEST_CARDS:
    c["image_path"] = DATA_IMG


def main():
    if not DATA_IMG.exists():
        sys.exit(f"image not found: {DATA_IMG}. Положите любую PNG-карточку.")

    print("=" * 72)
    print("Live demo: verdict-conditioned reasoning в сервисе (ReasoningPredictor)")
    print("=" * 72)
    inner = os.getenv("DEMO_INNER", "d2v_catboost")
    print(f"\nLoading predictor ({inner} + lazy Qwen2.5) …")
    t0 = time.time()
    p = ReasoningPredictor(inner_name=inner)
    p.load()
    print(f"  ready in {time.time()-t0:.1f}s")

    image_bytes = DATA_IMG.read_bytes()

    for i, card in enumerate(TEST_CARDS, 1):
        print(f"\n{'-' * 72}")
        print(f"Карточка {i}/{len(TEST_CARDS)}: {card['label']}")
        print(f"  name:  {card['name']}")
        print(f"  brand: {card['brand']!r}")
        print(f"  price: {card['tab_inputs']['PriceDiscounted']}  "
              f"seller_age: {card['tab_inputs']['seller_time_alive']}d")

        t1 = time.time()
        result = p.predict(
            image_bytes=image_bytes,
            name=card["name"],
            description=card["description"],
            brand=card["brand"],
            tab_inputs=card["tab_inputs"],
        )
        dt = time.time() - t1

        print(f"\n  → request took {dt:.2f}s")
        print(f"  → JSON-ответ как из FastAPI /predict:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("Done. Для запуска через FastAPI/uvicorn:")
    print("  cd service/counterfeit_service")
    print("  PREDICTOR_TYPE=reasoning_pipeline uvicorn app.main:app --port 8000")
    print("  → curl -X POST http://localhost:8000/predict -F image=@... [...]")
    print("=" * 72)


if __name__ == "__main__":
    main()
