import concurrent.futures
import time
import statistics
import requests
import base64

URL = "http://127.0.0.1:8002/v2/models/efficientnet/infer"
N_REQUESTS = 200
CONCURRENCY = 4

with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

PAYLOAD = {
    "id": "load",
    "inputs": [{
        "name": "image",
        "shape": [1],
        "datatype": "BYTES",
        "data": [image_b64],
    }],
}

def one_request():
    t0 = time.perf_counter()
    try:
        r = requests.post(URL, json=PAYLOAD, timeout=30)
        status = r.status_code
    except Exception:
        status = -1
    return status, time.perf_counter() - t0

def main():
    print(f"URL: {URL}")
    print(f"Requests: {N_REQUESTS}")
    print(f"Concurrency: {CONCURRENCY}")

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(lambda _: one_request(), range(N_REQUESTS)))
    t_total = time.perf_counter() - t_start

    latencies = sorted(t for _, t in results)
    errors = sum(1 for s, _ in results if s != 200)

    print()
    print("=" * 40)
    print(f"Total time:   {t_total:.2f} s")
    print(f"RPS:          {N_REQUESTS / t_total:.1f}")
    print(f"Errors:       {errors} ({errors / N_REQUESTS * 100:.2f}%)")
    print(f"Latency p50:  {latencies[len(latencies) // 2] * 1000:.1f} ms")
    print(f"Latency p95:  {latencies[int(len(latencies) * 0.95)] * 1000:.1f} ms")
    print(f"Latency p99:  {latencies[int(len(latencies) * 0.99)] * 1000:.1f} ms")
    print("=" * 40)

if __name__ == "__main__":
    main()