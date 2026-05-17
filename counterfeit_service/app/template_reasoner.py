"""
Rule-based explainer для verdict-conditioned reasoning (§ 4.4.9.7 + § 7.4).

Архитектурное обоснование. Эмпирически зафиксировано (§ 4.4.9.3, § 4.4.9.4,
§ 4.4.9.6), что foundation-LLM в zero-shot режиме на маркетплейс-borderline
демонстрируют отрицательный перенос как по proba (Qwen2.5/CLIP/Qwen2-VL
ROC ≈ 0,5), так и по практическому качеству генерации (Qwen2.5-1.5B/0.5B
склонны к echo-input на коротких промптах с малым контекстом). В связи с
этим в production-конфигурации сервиса используется детерминированный
rule-based explainer, формирующий объяснения на основе правил-эвристик,
выведенных из SHAP top-15 признаков финальной модели (§ 4.4.7) и
counterfactual-анализа (§ 4.4.7.2).

LLM-канал реализован в `llm_explainer.py` и сохранён как опциональный
feature flag (`USE_LLM_REASONING=1`); полностью документирован в § 4.4.9.7
ВКР с эмпирическим разбором 50 borderline-объектов командного теста.
"""
from __future__ import annotations

from typing import Optional

# Маркеры в названии, явно указывающие на нелицензионный товар.
_REPLICA_MARKERS = (
    "реплика", "копия", "копию", "lookalike", "look-alike", "replica",
    "оригинал" + "ная" * 0,  # 'оригинальная подделка' встречается в name_rus
    "под видом", "1:1", "первая копия", "люкс копия", "ААА копия",
)

# Подмножество известных премиальных брендов: цена ниже порога ставит
# карточку под подозрение. Список намеренно короткий и расширяемый — это
# не полный реестр, а триггеры для бытовой electronics-категории Ozon.
_PREMIUM_BRANDS_MIN_PRICE = {
    "apple":   30_000,
    "samsung": 15_000,
    "sony":    10_000,
    "bose":    8_000,
    "dyson":   15_000,
    "nintendo": 15_000,
    "lego":    3_000,
    "lancome": 2_500,
    "chanel":  3_000,
    "ysl":     2_500,
    "dior":    3_000,
    "gucci":   5_000,
    "nike":    3_500,
    "adidas":  3_500,
    "lacoste": 3_000,
}


def _is_replica(text: Optional[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in _REPLICA_MARKERS)


def _is_premium_underpriced(brand: Optional[str], price: float) -> tuple[bool, str]:
    """True + читаемый бренд, если premium-бренд и цена ниже минимального."""
    if not brand or price <= 0:
        return False, ""
    low = brand.strip().lower()
    threshold = _PREMIUM_BRANDS_MIN_PRICE.get(low)
    if threshold and price < threshold:
        return True, brand.strip()
    return False, ""


def _collect_factors(card: dict) -> list[str]:
    """Собрать читаемые причины-фразы из карточки.

    Каждая фраза — короткая «человеческая» формулировка причины. Порядок —
    по убыванию значимости (взвешено через SHAP top-15 в § 4.4.7).
    """
    factors: list[str] = []

    name = (card.get("name") or "").strip()
    description = (card.get("description") or "").strip()
    brand = (card.get("brand") or "").strip()
    price = float(card.get("PriceDiscounted") or card.get("price") or 0.0)
    item_age = float(card.get("item_time_alive") or 0.0)
    seller_age = float(card.get("seller_time_alive") or 0.0)
    sales30 = float(card.get("item_count_sales30") or 0.0)
    returns30 = float(card.get("item_count_returns30") or 0.0)

    # 1. Прямые признаки контрафакта в названии.
    if _is_replica(name) or _is_replica(description):
        factors.append("в названии или описании указано, что это реплика или копия")

    # 2. Premium-бренд + подозрительно низкая цена.
    is_underpriced, brand_label = _is_premium_underpriced(brand, price)
    if is_underpriced:
        factors.append(f"цена сильно ниже обычной розничной для бренда {brand_label} — типичный признак подделки")
    elif price > 0 and price < 100:
        factors.append("цена нерыночно низкая — товар отдают практически даром")

    # 3. Молодой продавец без истории.
    if seller_age > 0 and seller_age < 30:
        factors.append(f"продавец зарегистрирован совсем недавно — около {int(seller_age)} дн. назад")
    elif seller_age == 0 and (price > 0 or name):
        factors.append("история продавца ещё не сформирована, нет данных о его поведении")

    # 4. Молодая карточка / нет продаж.
    if item_age > 0 and item_age < 7:
        factors.append(f"карточка опубликована меньше недели назад ({int(item_age)} дн.)")
    if sales30 == 0 and seller_age > 0 and item_age > 7:
        factors.append("за последний месяц по карточке не было продаж — нетипично для зрелого товара")

    # 5. Возвраты при низких продажах.
    if returns30 > 0 and sales30 < max(1.0, returns30 * 3):
        factors.append(f"за последние 30 дней были возвраты ({int(returns30)}) при минимальных продажах ({int(sales30)})")

    # 6. Несоблюдение требований к заполнению карточки.
    if not name:
        factors.append("в карточке не заполнено название")
    if not description and brand:
        factors.append("бренд указан, но описание товара отсутствует")
    if not brand and name:
        factors.append("название заполнено, а бренд — нет")
    if price == 0:
        factors.append("цена не указана")

    return factors


def _upper_first(s: str) -> str:
    """Сделать первую букву заглавной, не ломая casing остального (Apple/Dyson)."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def _humanize_join(factors: list[str], limit: int, connector: str = "а ещё") -> str:
    """Склеить причины человеческим текстом."""
    chosen = factors[:limit]
    if not chosen:
        return ""
    if len(chosen) == 1:
        return _upper_first(chosen[0])
    if len(chosen) == 2:
        return f"{_upper_first(chosen[0])}, {connector} {chosen[1]}"
    head = ", ".join(chosen[:-1])
    return f"{_upper_first(head)}, {connector} {chosen[-1]}"


# Тёплые шаблоны: 1-е предложение — что произошло; 2-е — почему; 3-е — что делать.
# Заметь: проценты, «модель», «сигналы» — никогда не упоминаются.

_TEMPLATES = {
    "blocking_explanation": {
        "headline": "Карточка автоматически снята с продажи.",
        "reason_intro": "Основная причина: {reasons}.",
        "advice": (
            "Если считаете решение ошибочным — отредактируйте карточку "
            "(название, бренд, цену) и обратитесь в поддержку для повторной проверки."
        ),
        "fallback_reason": (
            "сочетание поведенческих признаков продавца и параметров карточки "
            "совпадает с паттерном, типичным для контрафактных листингов"
        ),
        "connector": "и",
        "max_factors": 2,
    },
    "borderline_explanation": {
        "headline": "Карточка отправлена модератору на ручную проверку.",
        "reason_intro": "Что насторожило: {reasons}.",
        "advice": (
            "Модератор посмотрит карточку в ближайшее время — "
            "пока что её видимость для покупателей не меняется."
        ),
        "fallback_reason": (
            "несколько параметров карточки и продавца одновременно отклоняются "
            "от обычного паттерна для этой категории"
        ),
        "connector": "а ещё",
        "max_factors": 3,
    },
    "confident_positive_no_block": {
        "headline": "Карточка добавлена в очередь контроля качества.",
        "reason_intro": "На что обратить внимание: {reasons}.",
        "advice": (
            "Карточка остаётся в продаже, но рекомендуется выборочная "
            "проверка для подтверждения соответствия требованиям маркетплейса."
        ),
        "fallback_reason": (
            "сочетание параметров одновременно повышает риск, "
            "хотя ни один из них в отдельности не критичен"
        ),
        "connector": "и",
        "max_factors": 2,
    },
}


def generate_reasoning(
    card: dict,
    prediction: float,
    mode: str = "borderline_explanation",
) -> str:
    """Сформировать тёплый, человекочитаемый reasoning-текст.

    Результат — 2-3 предложения структуры «что произошло → почему → что делать».
    Гарантированно не упоминает «модель», «вероятность», «сигналы»; не
    выдумывает фактов вне карточки.
    """
    tmpl = _TEMPLATES.get(mode, _TEMPLATES["borderline_explanation"])
    factors = _collect_factors(card)
    if factors:
        reasons = _humanize_join(factors, limit=tmpl["max_factors"], connector=tmpl["connector"])
    else:
        reasons = tmpl["fallback_reason"]
    text = f"{tmpl['headline']} {tmpl['reason_intro'].format(reasons=reasons)} {tmpl['advice']}"
    # Защита от двойной точки и лишних пробелов.
    text = text.replace(" .", ".").replace("..", ".").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text
