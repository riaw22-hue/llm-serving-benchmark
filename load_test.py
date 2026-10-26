
import asyncio
import time
import statistics
import httpx
import argparse
import json

API_URL = "http://127.0.0.1:8000/generate"


async def send_request(client, engine, semaphore, number):
    async with semaphore:
        start = time.perf_counter()

        try:
            response = await client.post(
                API_URL,
                json={
                    "engine": engine,
                    "prompt": f"Explain AI briefly. Request {number}",
                    "max_new_tokens": 50,
                },
                timeout=180.0,
            )

            elapsed = time.perf_counter() - start

            return {
                "success": response.status_code == 200,
                "latency": elapsed,
                "status": response.status_code,
            }

        except Exception as error:
            return {
                "success": False,
                "latency": time.perf_counter() - start,
                "error": str(error),
            }


async def main(engine, requests_count, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()

        results = await asyncio.gather(*[
            send_request(client, engine, semaphore, i + 1)
            for i in range(requests_count)
        ])

        duration = time.perf_counter() - start

    successful = [r for r in results if r["success"]]
    latencies = sorted(r["latency"] for r in successful)

    def percentile(p):
        if not latencies:
            return None
        index = min(len(latencies) - 1, int(len(latencies) * p))
        return round(latencies[index], 3)

    report = {
        "engine": engine,
        "total_requests": requests_count,
        "concurrency": concurrency,
        "successful_requests": len(successful),
        "failed_requests": requests_count - len(successful),
        "total_duration_seconds": round(duration, 3),
        "requests_per_second": round(len(successful) / duration, 3)
        if duration else 0,
        "average_latency_seconds": round(statistics.mean(latencies), 3)
        if latencies else None,
        "p50_latency_seconds": percentile(0.50),
        "p95_latency_seconds": percentile(0.95),
    }

    filename = f"load_test_{engine}.json"

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n===== LOAD TEST REPORT =====")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ollama", "transformers"],
                        default="ollama")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("Requests and concurrency must be at least 1")

    asyncio.run(main(args.engine, args.requests, args.concurrency))
