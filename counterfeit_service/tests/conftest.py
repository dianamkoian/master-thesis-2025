"""
Pytest fixtures для функционального тестирования сервиса (§ 7.5.2 ВКР).

Тесты не требуют RabbitMQ или PostgreSQL — все внешние зависимости подменены:
  - PREDICTOR_TYPE=stub_borderline → детерминированный predictor без ML-артефактов;
  - DATABASE_URL → in-memory SQLite (`sqlite+aiosqlite:///:memory:`);
  - AUDIT_IMAGE_STORAGE_ENABLE=0 → изображения не сохраняются на диск;
  - LLMExplainer.explain пропатчен через monkeypatch → возвращает фиксированную
    строку (так как загрузка Qwen2.5-1.5B-Instruct требует ~30 секунд и 3 ГБ диска
    в тестовом окружении).
"""
from __future__ import annotations

import io
import os

import pytest


# Подмена env ДО импорта приложения — критично, потому что `predictor = get_predictor()`
# выполняется на module-load и читает PREDICTOR_TYPE один раз.
# По умолчанию используем reasoning_pipeline поверх stub_borderline — это
# даёт полный контракт response без зависимости от M2-FE+ артефактов и LLM.
os.environ.setdefault("PREDICTOR_TYPE", "reasoning_pipeline")
os.environ.setdefault("PREDICTOR_INNER", "stub_borderline")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_audit.db")
os.environ.setdefault("AUDIT_IMAGE_STORAGE_ENABLE", "0")
# Невалидный hostname — lifespan быстро упадёт на DNS и продолжит без RabbitMQ.
os.environ.setdefault("RABBITMQ_URL", "amqp://test:test@rabbitmq-not-here.invalid:5672/")


@pytest.fixture(scope="session")
def stub_image_bytes() -> bytes:
    """Минимальное валидное PNG-изображение 1×1 (для multipart upload)."""
    # 67-byte PNG header for a 1×1 transparent pixel — самое короткое
    # допустимое PNG, генерируется детерминированно через Pillow.
    from PIL import Image
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    """Заменить LLM на быстрый stub, чтобы тесты не грузили Qwen2.5 (~30 с)."""
    from app import llm_explainer

    def fake_explain(self, card, prediction, signals=None, mode="borderline_explanation"):
        if mode == "blocking_explanation":
            return "Карточка заблокирована: тестовый стаб LLM-канала."
        if mode == "borderline_explanation":
            return "Тестовый стаб LLM-канала, режим borderline."
        return "Тестовый стаб LLM-канала."

    def fake_load(self):
        # Помечаем модель как «загруженную», чтобы explain не пытался её грузить.
        self.model = object()
        self.tokenizer = object()

    monkeypatch.setattr(llm_explainer.LLMExplainer, "explain", fake_explain)
    monkeypatch.setattr(llm_explainer.LLMExplainer, "load", fake_load)


@pytest.fixture
def client():
    """FastAPI TestClient. Lifespan запустится автоматически (без RabbitMQ)."""
    # Импорт внутри fixture — env-переменные подменены выше.
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
