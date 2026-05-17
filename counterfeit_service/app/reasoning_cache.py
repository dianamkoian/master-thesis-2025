"""
In-memory кеш для deferred LLM-объяснений (§ 4.4.9.7 + § 7.4).

Архитектурное обоснование. Pattern progressive disclosure: фронтенд получает
быстрый verdict (~1 c) и опрашивает reasoning отдельным GET-запросом
(~3–7 c на LLM-инференс). Этот модуль реализует короткоживущий thread-safe
кеш для хранения промежуточных результатов LLM-генерации между
synchronous `/predict?defer_reasoning=true` и polling-запросами клиента к
`GET /predict/{task_id}/reasoning`.

Limitation: in-memory cache валиден только в рамках одного worker'а. Для
горизонтально-масштабируемого деплоя (несколько uvicorn-replic'ов за
балансировщиком) необходимо заменить на Redis с тем же контрактом — модель
данных и TTL остаются прежними.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

# По умолчанию объяснения живут 10 минут — достаточно для polling-цикла
# фронтенда (~5 запросов с интервалом 1.5 с) и для отладочного просмотра
# через /predict/{task_id}/reasoning, недостаточно для накопления памяти.
DEFAULT_TTL_SECONDS = 600
# Грубый предохранитель: при превышении лимита кеш чистится самым старым
# entry. В production-сценарии Redis с TTL делает это сам.
MAX_ENTRIES = 10_000


@dataclass
class CachedReasoning:
    status: str                   # "pending" | "ready" | "error"
    reasoning: Optional[str]
    reasoning_mode: Optional[str]
    created_at: float
    error: Optional[str] = None


class ReasoningCache:
    """Thread-safe key→value кеш с TTL."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_entries: int = MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._store: dict[str, CachedReasoning] = {}

    def mark_pending(self, task_id: str) -> None:
        with self._lock:
            self._evict_locked()
            self._store[task_id] = CachedReasoning(
                status="pending", reasoning=None, reasoning_mode=None,
                created_at=time.time(),
            )

    def mark_ready(self, task_id: str, reasoning: str, reasoning_mode: str) -> None:
        with self._lock:
            self._store[task_id] = CachedReasoning(
                status="ready", reasoning=reasoning, reasoning_mode=reasoning_mode,
                created_at=time.time(),
            )

    def mark_error(self, task_id: str, error: str) -> None:
        with self._lock:
            self._store[task_id] = CachedReasoning(
                status="error", reasoning=None, reasoning_mode="llm_error",
                created_at=time.time(), error=error,
            )

    def get(self, task_id: str) -> Optional[CachedReasoning]:
        with self._lock:
            entry = self._store.get(task_id)
            if entry is None:
                return None
            if time.time() - entry.created_at > self._ttl:
                self._store.pop(task_id, None)
                return None
            return entry

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def _evict_locked(self) -> None:
        if len(self._store) < self._max:
            return
        # Самые старые записи удаляем первыми. Это O(n) на eviction, но
        # eviction случается редко (только при достижении MAX_ENTRIES).
        oldest = sorted(self._store.items(), key=lambda kv: kv[1].created_at)
        for task_id, _ in oldest[: self._max // 4]:
            self._store.pop(task_id, None)


# Глобальный синглтон, импортируется в main.py.
reasoning_cache = ReasoningCache()
