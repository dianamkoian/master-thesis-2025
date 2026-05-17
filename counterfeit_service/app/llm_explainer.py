"""
LLM-канал каскадной архитектуры (§ 4.4.9.7 ВКР).

Verdict-conditioned reasoning: получая прогноз от primary predictor + признаки
карточки, генерирует текстовое объяснение через Qwen2.5-1.5B-Instruct в
zero-shot режиме. Используется как explanation engine, дополняющий вердикт-канал.

Архитектурное обоснование: эмпирически (§ 4.4.9.4) Qwen2.5 в роли Stage 2
предиктора даёт отрицательный перенос по proba (ROC ≈ 0,52), однако
качественный reasoning сохраняется. Логичное разделение каналов — proba от
основного классификатора, reasoning от LLM — даёт интерпретируемость без
ухудшения метрик ранжирования.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_REASONING_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_DEVICE = os.getenv("LLM_REASONING_DEVICE", "")  # "mps"/"cpu"/"cuda"; "" = auto
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LLM_REASONING_MAX_TOKENS", "260"))
MAX_SENTENCES = int(os.getenv("LLM_REASONING_MAX_SENTENCES", "4"))

# Шаблонные преамбулы, которые Qwen2.5-1.5B-Instruct склонна вставлять.
# Если ответ начинается с одной из них, отрезаем до первой осмысленной фразы.
_PREAMBLE_PREFIXES = (
    "модель оценила",
    "модель оценивает",
    "модель посчитала",
    "модель предсказывает",
    "модель основана",
    "модель сделала",
    "представительский образец",
    "представительство",
    "при анализе карточки",
    "при оценке вероятности",
    "вероятность контрафакта",
    "анализируя признаки",
    "анализ карточки",
    "этот вопрос требует",
    "это подразумевает",
    "сигналы модели",
    "согласно сигналам",
    "по сигналам",
)

# Запретные подстроки в каждом предложении ответа — оно вычищается,
# если LLM проигнорировала инструкцию и вставила технические данные.
_FORBIDDEN_SUBSTRINGS = (
    "multimodal_score", "image_signal", "text_signal",
    "сигнал", "сигналы", "процент", "вероятност",
    "модель оцен", "модель счит", "модель посчит", "модель пред",
)
import re as _re_filter  # local alias for filter pass


def _select_device(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_prompt(
    card: dict,
    prediction: float,
    signals: Optional[dict] = None,
    mode: str = "borderline_explanation",
) -> list[dict]:
    """Сформировать verdict-conditioned ChatML-сообщения для Qwen2.5-Instruct.

    Возвращает список [{"role": "system", ...}, {"role": "user", ...}],
    который применяется через tokenizer.apply_chat_template() в LLMExplainer.

    Args:
        card: словарь с полями карточки. Распознаются: name, description,
              brand, PriceDiscounted/price, item_time_alive, seller_time_alive,
              CommercialTypeName4/category, item_count_sales30, item_count_returns30.
        prediction: probability контрафакта ∈ [0, 1] от primary predictor.
        signals: опциональные сигналы модели (НЕ используются в промпте,
                 чтобы LLM не вставляла их обратно в ответ; принимаются для
                 обратной совместимости интерфейса с предыдущими версиями).
        mode: режим, определяет тональность объяснения. Поддерживается:
              `blocking_explanation` (для уведомления селлера / тикета поддержки),
              `borderline_explanation` (для модератора, ручная проверка),
              `confident_positive_no_block` (контроль качества).
    """
    def g(*keys, default="не указано"):
        for k in keys:
            v = card.get(k)
            if v is not None and v != "":
                return v
        return default

    # Внутренняя метка для подсказки тональности (low/mid/high риск),
    # без явных процентов в промпте — иначе слабая модель просто их повторяет.
    if prediction >= 0.85:
        risk = "очень высокий"
    elif prediction >= 0.5:
        risk = "повышенный"
    elif prediction >= 0.25:
        risk = "умеренный"
    else:
        risk = "низкий"

    # Возвращаем СПИСОК ChatML-сообщений; LLMExplainer.explain применит
    # apply_chat_template() корректно для Qwen2.5-Instruct. Plain-text
    # промпт для instruct-моделей даёт echo вместо ответа.
    system = (
        "Ты — аналитик маркетплейса. Твоя задача — кратко и понятно объяснить "
        "селлеру или модератору, почему карточка попала на дополнительную проверку. "
        "Пиши деловым нейтральным тоном, СПЛОШНЫМ ТЕКСТОМ в 2–3 предложениях. "
        "ЗАПРЕЩЕНО использовать: нумерованные и маркированные списки (1./2./-./•), "
        "заголовки, markdown-разметку (**, ##), переносы строк внутри ответа. "
        "Опирайся ТОЛЬКО на признаки карточки. Запрещено упоминать слова «модель», "
        "«сигналы», «вероятность», «прогноз», «multimodal_score», «image_signal», "
        "«text_signal», а также числовые проценты и score'ы. "
        "Не выдумывай факты, которых нет в карточке."
    )
    user = (
        f"{_MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS['borderline_explanation'])}\n\n"
        "Карточка:\n"
        f"• Название: {g('name', 'name_rus')}\n"
        f"• Описание: {g('description')}\n"
        f"• Бренд: {g('brand', 'brand_name')}\n"
        f"• Категория: {g('CommercialTypeName4', 'category')}\n"
        f"• Цена: {g('PriceDiscounted', 'price')} ₽\n"
        f"• Возраст карточки на площадке: {g('item_time_alive')} дн.\n"
        f"• Возраст продавца: {g('seller_time_alive')} дн.\n"
        f"• Продажи за 30 дней: {g('item_count_sales30')}\n"
        f"• Возвраты за 30 дней: {g('item_count_returns30')}\n"
        f"• Внутренняя оценка риска: {risk}.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


_MODE_INSTRUCTIONS = {
    "blocking_explanation": (
        "Контекст: автоматическое уведомление селлеру о блокировке товара. "
        "Тон — официальный, нейтральный, без агрессии. "
        "Начни ответ с фразы «Карточка заблокирована: » или «Товар снят с продажи: » "
        "и в двух-трёх предложениях объясни, какие конкретные особенности карточки "
        "(цена, бренд, возраст продавца, история продаж/возвратов) привели к блокировке."
    ),
    "borderline_explanation": (
        "Контекст: пояснение для модератора, который вручную проверит карточку. "
        "Тон — спокойный, аналитический. Опиши в двух-трёх предложениях, какие "
        "особенности карточки (цена, бренд, возраст продавца, история продаж) "
        "вызывают сомнения, и куда стоит посмотреть модератору в первую очередь."
    ),
    "confident_positive_no_block": (
        "Контекст: пояснение для аналитика контроля качества. Тон — деловой. "
        "В одном-двух предложениях укажи ключевые особенности карточки, "
        "поддерживающие подозрение на контрафакт."
    ),
}


class LLMExplainer:
    """Lazy-load Qwen2.5-1.5B-Instruct wrapper для verdict-conditioned reasoning."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.model_path = model_path
        self.device = _select_device(device)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        logger.info("LLMExplainer initialised (model=%s, device=%s)", self.model_path, self.device)

    def load(self) -> None:
        """Загрузка модели и токенизатора. Идемпотентна."""
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import — heavy
        import torch
        # CPU inference optimization: задаём число потоков по числу
        # доступных ядер (по умолчанию torch берёт 1 или половину).
        # На macOS-Docker это даёт ×2–4 ускорение для Qwen2.5-0.5B/1.5B.
        if self.device == "cpu":
            try:
                ncpu = os.cpu_count() or 4
                torch.set_num_threads(ncpu)
                torch.set_num_interop_threads(max(1, ncpu // 2))
                logger.info("torch CPU threads: intraop=%d interop=%d", ncpu, ncpu // 2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch thread tuning failed: %s", exc)
        logger.info("Loading LLM weights from %s ...", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # float32 на CPU работает быстрее float16 (без специальных bf16-инструкций);
        # на MPS/CUDA — auto подберёт оптимальный тип.
        dtype = torch.float32 if self.device == "cpu" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        logger.info("LLM ready on %s (max_new_tokens=%d)", self.device, self.max_new_tokens)

    def explain(
        self,
        card: dict,
        prediction: float,
        signals: Optional[dict] = None,
        mode: str = "borderline_explanation",
    ) -> str:
        """Zero-shot inference через ChatML-template + post-process.

        Использует tokenizer.apply_chat_template() — это критично для
        Qwen2.5-Instruct: без него модель работает как plain LM-continuation
        и echo'ит промпт вместо генерации ответа.
        """
        if self.model is None:
            self.load()
        messages = build_prompt(card, prediction, signals, mode=mode)
        # apply_chat_template вернёт строку с правильной разметкой <|im_start|>...
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Отрезать echoed prompt из output (input_ids → токены ответа).
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _postprocess(answer, MAX_SENTENCES)


# ---------------------------------------------------------------------------
# Post-processing: убрать шаблонные преамбулы, обрезать до N предложений.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402 — module-level utility


def _postprocess(text: str, max_sentences: int) -> str:
    """Очистить ответ LLM:
    1) сжать в одну строку, убрать markdown и списочные маркеры;
    2) убрать шаблонные преамбулы и оборванные list-пункты;
    3) выкинуть предложения с запрещёнными терминами и числами;
    4) оставить первые N законченных предложений с осмысленным окончанием."""
    s = text.strip()
    # 1. Свести многострочный текст в один абзац, убрать markdown.
    s = _re.sub(r"\n+", " ", s)                     # все переводы строк → пробел
    s = _re.sub(r"\*+([^*]+?)\*+", r"\1", s)        # **жирный** → жирный
    s = _re.sub(r"`+([^`]+?)`+", r"\1", s)          # `моноширинный` → обычный
    s = _re.sub(r"^\s*#{1,6}\s+", "", s, flags=_re.M)  # заголовки # / ##
    # Убрать list-маркеры в начале и середине строки: "1. ", "2. ", "- ", "• "
    s = _re.sub(r"(?:^|\s)(?:\d+\.\s*|[-•]\s+)", " ", s)
    # Сжать множественные пробелы.
    s = _re.sub(r"\s+", " ", s).strip()
    # 2. Если первое предложение — шаблонная преамбула, отрезать до следующего.
    lower = s.lower()
    for prefix in _PREAMBLE_PREFIXES:
        if lower.startswith(prefix):
            cut = s.find(".")
            if 0 < cut < len(s) - 1:
                s = s[cut + 1:].strip()
                break
    # 3. Разбить на предложения и отфильтровать «грязные» и слишком короткие.
    raw_sentences = _re.split(r"(?<=[\.\!\?])\s+", s)
    clean = []
    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 8:               # фрагменты типа "2." или "А."
            continue
        low = sent.lower()
        # Убираем технические предложения с процентами / сигналами / «моделями».
        if any(bad in low for bad in _FORBIDDEN_SUBSTRINGS):
            continue
        # Убираем предложения, которые на 30%+ состоят из цифр (probability-dump).
        digits = sum(c.isdigit() for c in sent)
        if len(sent) > 4 and digits / len(sent) > 0.30:
            continue
        clean.append(sent)
    # 4. Откинуть оборванное последнее предложение без финального знака.
    if clean and not _re.search(r"[\.\!\?]\s*$", clean[-1]):
        if len(clean) > 1:
            clean = clean[:-1]
    s = " ".join(clean[:max_sentences]).strip()
    # 5. Снять trailing markdown «**», «:», итд.
    s = _re.sub(r"\s*[\*_:]+\s*$", "", s).strip()
    # 6. Fallback: если все предложения отфильтрованы — отдать generic-формулировку.
    if not s:
        return ("Карточка вызывает сомнения по совокупности факторов: "
                "цена, история продавца и продаж требуют дополнительной проверки модератором.")
    return s
