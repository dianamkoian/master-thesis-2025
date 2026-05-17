"""
Нагрузочное тестирование сервиса (§ 7.5.3 ВКР).

Параметризуемый async-load-tester: запускает N concurrent клиентов,
каждый шлёт R запросов, измеряет latency (p50/p95/p99), throughput и
error rate для sync /predict и async /predict-async.

Запуск (примеры):
  python scripts/loadtest_predict.py --mode sync  --concurrency 5  --requests 100
  python scripts/loadtest_predict.py --mode async --concurrency 10 --requests 200
  python scripts/loadtest_predict.py --suite                   # полный sweep
                                                                 # 1/5/10/50 conc.

Output:
  artifacts_loadtest/loadtest_{mode}_c{N}_r{R}.csv  — raw per-request метрики
  artifacts_loadtest/loadtest_summary.md            — сводная таблица
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx


HERE = Path(__file__).resolve().parent.parent  # counterfeit_service/
OUT = HERE / "artifacts_loadtest"
OUT.mkdir(parents=True, exist_ok=True)

# Тестовая картинка. Ищем сначала в новой структуре (master-thesis-2026/data/), затем legacy.
_IMG_CANDIDATES = [
    HERE.parent.parent / "data" / "ml_ozon_counterfeit_test_images" / "10.png",
    Path("/Users/diana/master-thesis-2025/data/ml_ozon_counterfeit_test_images/10.png"),
]
IMG = next((p for p in _IMG_CANDIDATES if p.exists()), _IMG_CANDIDATES[0])
URL = "http://localhost:8000"

CARDS = [
    {"name": "Смартфон Samsung Galaxy S24", "brand": "Samsung", "category": "Смартфоны",
     "price": "65000", "item_time_alive": "120", "seller_time_alive": "850",
     "item_count_sales30": "45", "item_count_returns30": "2"},
    {"name": "Кружка керамическая 350мл", "brand": "", "category": "Посуда",
     "price": "250", "item_time_alive": "540", "seller_time_alive": "1200",
     "item_count_sales30": "87", "item_count_returns30": "2"},
    {"name": "iPhone 15 Pro Max 256GB оригинал", "brand": "Apple", "category": "Смартфоны",
     "price": "8500", "item_time_alive": "3", "seller_time_alive": "12",
     "item_count_sales30": "1", "item_count_returns30": "1"},
    {"name": "Dyson фен реплика", "brand": "Dyson", "category": "Фен",
     "price": "2500", "item_time_alive": "2", "seller_time_alive": "8",
     "item_count_sales30": "0", "item_count_returns30": "0"},
]


async def one_request_sync(client: httpx.AsyncClient, image_bytes: bytes, card: dict) -> tuple[bool, float, int]:
    t0 = time.perf_counter()
    files = {"image": ("test.png", image_bytes, "image/png")}
    try:
        r = await client.post(f"{URL}/predict?defer_reasoning=true", files=files, data=card, timeout=30.0)
        return r.status_code == 200, (time.perf_counter() - t0) * 1000, r.status_code
    except Exception:
        return False, (time.perf_counter() - t0) * 1000, 0


async def one_request_async(client: httpx.AsyncClient, image_bytes: bytes, card: dict) -> tuple[bool, float, int]:
    """Запрос /predict-async + poll до status='done' (засекаем total)."""
    t0 = time.perf_counter()
    files = {"image": ("test.png", image_bytes, "image/png")}
    try:
        r = await client.post(f"{URL}/predict-async", files=files, data=card, timeout=10.0)
        if r.status_code != 200:
            return False, (time.perf_counter() - t0) * 1000, r.status_code
        task_id = r.json()["task_id"]
        # Polling
        for _ in range(60):
            await asyncio.sleep(0.5)
            rr = await client.get(f"{URL}/result/{task_id}", timeout=5.0)
            if rr.status_code == 200 and rr.json().get("status") == "done":
                return True, (time.perf_counter() - t0) * 1000, 200
        return False, (time.perf_counter() - t0) * 1000, 408  # timeout
    except Exception:
        return False, (time.perf_counter() - t0) * 1000, 0


async def worker_loop(client, image_bytes, card_pool, mode, n_requests, results):
    fn = one_request_sync if mode == "sync" else one_request_async
    for i in range(n_requests):
        ok, ms, code = await fn(client, image_bytes, card_pool[i % len(card_pool)])
        results.append((ok, ms, code))


async def one_request_error_scenario(client: httpx.AsyncClient, scenario: str) -> tuple[bool, float, int]:
    """Запрос «ожидаемой ошибки» — проверяет, что сервис корректно валидирует
    некорректный вход и не падает (expected: HTTP 400 / 404 / 422)."""
    t0 = time.perf_counter()
    try:
        if scenario == "empty_image":
            r = await client.post(f"{URL}/predict",
                                  files={"image": ("empty.png", b"", "image/png")},
                                  data={"name": "test"}, timeout=10.0)
            ok = r.status_code == 400
        elif scenario == "non_image_mime":
            r = await client.post(f"{URL}/predict",
                                  files={"image": ("test.txt", b"not an image", "text/plain")},
                                  data={"name": "test"}, timeout=10.0)
            ok = r.status_code == 400
        elif scenario == "no_file":
            r = await client.post(f"{URL}/predict",
                                  data={"name": "test"}, timeout=10.0)
            ok = r.status_code in (400, 422)  # FastAPI вернёт 422 при отсутствии required File
        elif scenario == "unknown_task":
            r = await client.get(f"{URL}/predict/00000000-0000-0000-0000-000000000000/reasoning",
                                 timeout=5.0)
            ok = r.status_code == 404
        elif scenario == "unknown_result":
            r = await client.get(f"{URL}/result/no-such-task",
                                 timeout=5.0)
            ok = r.status_code == 404
        else:
            ok, code = False, 0
            return ok, (time.perf_counter() - t0) * 1000, code
        return ok, (time.perf_counter() - t0) * 1000, r.status_code
    except Exception:
        return False, (time.perf_counter() - t0) * 1000, 0


async def run_error_suite(per_scenario: int = 20) -> list:
    """Прогнать каждый error-сценарий `per_scenario` раз. Сервис должен
    стабильно возвращать ожидаемый 4xx-код без timeout'ов и падений."""
    scenarios = ["empty_image", "non_image_mime", "no_file",
                 "unknown_task", "unknown_result"]
    out = []
    async with httpx.AsyncClient() as client:
        for sc in scenarios:
            results = []
            t_start = time.perf_counter()
            tasks = [one_request_error_scenario(client, sc) for _ in range(per_scenario)]
            res = await asyncio.gather(*tasks)
            wall = time.perf_counter() - t_start
            ok = sum(1 for r in res if r[0])
            latencies = [r[1] for r in res]
            codes = sorted({r[2] for r in res})
            summary = {
                "scenario": sc, "requests": per_scenario,
                "expected_handled": ok,
                "http_codes": codes,
                "latency_ms_p50": round(sorted(latencies)[len(latencies) // 2], 1),
                "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
                "throughput_rps": round(per_scenario / wall, 1),
            }
            out.append(summary)
            print(f"  {sc:18s}: {ok}/{per_scenario} handled correctly, "
                  f"codes={codes}, p50={summary['latency_ms_p50']} ms")
    return out


async def run(mode: str, concurrency: int, requests_per_worker: int) -> dict:
    image_bytes = IMG.read_bytes()
    results: list = []
    async with httpx.AsyncClient() as client:
        t_start = time.perf_counter()
        await asyncio.gather(*[
            worker_loop(client, image_bytes, CARDS, mode, requests_per_worker, results)
            for _ in range(concurrency)
        ])
        wall = time.perf_counter() - t_start

    latencies = [r[1] for r in results if r[0]]
    n_total = len(results)
    n_ok = sum(1 for r in results if r[0])
    n_err = n_total - n_ok

    def pct(p):
        if not latencies:
            return 0
        s = sorted(latencies)
        k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
        return s[k]

    summary = {
        "mode": mode, "concurrency": concurrency, "requests": n_total,
        "ok": n_ok, "errors": n_err,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(n_ok / wall, 2) if wall > 0 else 0,
        "latency_ms_p50": round(pct(50), 1),
        "latency_ms_p95": round(pct(95), 1),
        "latency_ms_p99": round(pct(99), 1),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_ms_max": round(max(latencies), 1) if latencies else 0,
    }
    # Сохранить raw CSV.
    csv_path = OUT / f"loadtest_{mode}_c{concurrency}_r{requests_per_worker}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ok", "latency_ms", "http_code"])
        for r in results:
            w.writerow([int(r[0]), f"{r[1]:.2f}", r[2]])
    summary["csv"] = str(csv_path.relative_to(HERE))
    return summary


def fmt_row(s: dict) -> str:
    return (
        f"| {s['mode']} | {s['concurrency']} | {s['requests']} | "
        f"{s['ok']}/{s['requests']} | {s['throughput_rps']} | "
        f"{s['latency_ms_p50']} | {s['latency_ms_p95']} | "
        f"{s['latency_ms_p99']} | {s['latency_ms_max']} |"
    )


async def suite():
    print(f"Loadtest suite (image={IMG.name}, target={URL})\n")
    summaries = []
    # Sync: разные concurrency, по 25 запросов на worker.
    for c in (1, 5, 10, 25):
        print(f"  Running sync, concurrency={c}, requests/worker=25 ...")
        s = await run("sync", c, 25)
        summaries.append(s)
        print(f"    → throughput={s['throughput_rps']} RPS  "
              f"p50={s['latency_ms_p50']} ms  p95={s['latency_ms_p95']} ms  "
              f"errors={s['errors']}")
    # Async: меньше нагрузка, потому что polling каждые 500 мс.
    for c in (1, 5, 10):
        print(f"  Running async, concurrency={c}, requests/worker=10 ...")
        s = await run("async", c, 10)
        summaries.append(s)
        print(f"    → throughput={s['throughput_rps']} RPS  "
              f"p50={s['latency_ms_p50']} ms  p95={s['latency_ms_p95']} ms  "
              f"errors={s['errors']}")

    # Markdown summary.
    md = ["# Loadtest summary — sync /predict + async /predict-async (§ 7.5.3)",
          "",
          f"Target: `{URL}` (Docker-compose, M-series host, CPU inference). "
          "Predictor: `reasoning_pipeline` поверх `d2v_catboost`, "
          "reasoning channel: rule-based template (default).",
          "",
          "| mode | concurrency | requests | ok/total | throughput, RPS | "
          "p50, ms | p95, ms | p99, ms | max, ms |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        md.append(fmt_row(s))
    md.append("")
    md.append(f"_Сгенерировано {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"скриптом `scripts/loadtest_predict.py`._")
    (OUT / "loadtest_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved summary: {OUT / 'loadtest_summary.md'}")
    return summaries


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--requests", type=int, default=25, help="requests per worker")
    ap.add_argument("--suite", action="store_true", help="run full sweep")
    ap.add_argument("--errors", action="store_true", help="run error-handling suite")
    args = ap.parse_args()
    if args.errors:
        print("Error-handling suite — 5 scenarios × 20 requests each\n")
        results = asyncio.run(run_error_suite(per_scenario=20))
        # Записываем в md рядом со standard summary.
        md = ["", "## Error-handling load-test (§ 7.5.3, robustness)",
              "",
              "Проверка корректности обработки невалидных входов на сервисе под "
              "concurrent-нагрузкой. Каждый сценарий — 20 параллельных запросов; "
              "успех = сервис вернул ожидаемый 4xx-код в каждом из них, "
              "не упал в 5xx, не таймаут-нул.",
              "",
              "| Сценарий | n | handled (4xx) | HTTP-коды | p50, ms | p95, ms | RPS |",
              "|---|---|---|---|---|---|---|"]
        for s in results:
            md.append(
                f"| {s['scenario']} | {s['requests']} | {s['expected_handled']}/{s['requests']} | "
                f"{s['http_codes']} | {s['latency_ms_p50']} | {s['latency_ms_p95']} | "
                f"{s['throughput_rps']} |"
            )
        out_md = OUT / "loadtest_summary.md"
        existing = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
        out_md.write_text(existing + "\n".join(md), encoding="utf-8")
        print(f"\nAppended error-suite to: {out_md}")
    elif args.suite:
        asyncio.run(suite())
    else:
        s = asyncio.run(run(args.mode, args.concurrency, args.requests))
        print(s)
