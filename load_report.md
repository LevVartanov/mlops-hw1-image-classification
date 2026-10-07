- Образ: `ghcr.io/levvartanov/mlops-hw1-image-classification:latest`
- Лимиты: `--cpus=4 --memory=16g`
- Endpoint: `POST /v2/models/efficientnet/infer`
- Concurrency: 4
- Requests: 200
- Хост: Windows 11 + Docker Desktop (WSL2)

## Результаты

| Метрика | Значение |
|---|---|
| Total time | 34.33 с |
| RPS | 5.8 |
| p50 | 691.3 мс |
| p95 | 800.1 мс |
| p99 | 842.3 мс |
| Errors | 0 |

## Вывод

Сервис стабилен под нагрузкой: 0 ошибок, p99 ≈ 842 мс.