# Loadtest summary — sync /predict + async /predict-async (§ 7.5.3)

Target: `http://localhost:8000` (Docker-compose, M-series host, CPU inference). Predictor: `reasoning_pipeline` поверх `d2v_catboost`, reasoning channel: rule-based template (default).

| mode | concurrency | requests | ok/total | throughput, RPS | p50, ms | p95, ms | p99, ms | max, ms |
|---|---|---|---|---|---|---|---|---|
| sync | 1 | 25 | 25/25 | 7.28 | 136.8 | 167.0 | 194.0 | 194.0 |
| sync | 5 | 125 | 125/125 | 7.69 | 661.1 | 847.8 | 1008.8 | 1239.4 |
| sync | 10 | 250 | 250/250 | 6.24 | 1313.3 | 2022.5 | 4753.3 | 6800.5 |
| sync | 25 | 625 | 625/625 | 7.33 | 3376.3 | 3934.3 | 4337.1 | 4682.5 |
| async | 1 | 10 | 10/10 | 0.92 | 524.0 | 6108.6 | 6108.6 | 6108.6 |
| async | 5 | 50 | 50/50 | 6.88 | 526.9 | 1032.2 | 1046.2 | 1046.2 |
| async | 10 | 100 | 100/100 | 6.94 | 1524.4 | 1574.3 | 1629.7 | 1645.4 |

_Сгенерировано 2026-05-11 22:29:17 скриптом `scripts/loadtest_predict.py`._
## Error-handling load-test (§ 7.5.3, robustness)

Проверка корректности обработки невалидных входов на сервисе под concurrent-нагрузкой. Каждый сценарий — 20 параллельных запросов; успех = сервис вернул ожидаемый 4xx-код в каждом из них, не упал в 5xx, не таймаут-нул.

| Сценарий | n | handled (4xx) | HTTP-коды | p50, ms | p95, ms | RPS |
|---|---|---|---|---|---|---|
| empty_image | 20 | 20/20 | [400] | 22.4 | 26.7 | 638.9 |
| non_image_mime | 20 | 20/20 | [400] | 34.0 | 46.1 | 409.1 |
| no_file | 20 | 20/20 | [422] | 25.5 | 38.1 | 468.0 |
| unknown_task | 20 | 20/20 | [404] | 22.2 | 34.8 | 554.2 |
| unknown_result | 20 | 20/20 | [404] | 27.1 | 40.7 | 466.1 |