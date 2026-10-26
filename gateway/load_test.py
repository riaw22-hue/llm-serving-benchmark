
import asyncio
import time
import statistics
import httpx
import argparse
import json


API_URL = "http://127.0.0.1:8000/generate"


async def send_request(client, engine, prompt, semaphore):
    async with semaphore:
        start = time.perf_counter()

        try:
            response = await client.post(
                API_URL,
                json={
                    "engine": engine,
                    "prompt": prompt,
                    "max_new_tokens": 50,
                },
                timeout=180.0,
            )

            latency = time.perf_counter() - start

            if response.status_code == 200:
                return {
                    "success": True,
                    "latency": latency,
                    "status": response.status_code,
                }

            return {
                "success": False,
                "latency": latency,
                "status": response.status_code,
            }

        except Exception as exc:
            return {
                "success": False,
                "latency": time.perf_counter() - start,
                "error": str(exc),
            }


async def run_benchmark(engine, total_requests, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()

        tasks = [
            send_request(
                client,
                engine,
                f"Explain artificial intelligence in simple words. Request {i + 1}",
                semaphore,
            )
            for i in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    successful = [r for r in results if r["success"]]
    failed = total_requests - len(successful)
    latencies = [r["latency"] for r in successful]

    report = {
        "engine": engine,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful_requests": len(successful),
        "failed_requests": failed,
        "total_duration_seconds": round(total_time, 3),
        "requests_per_second": round(
            len(successful) / total_time, 3
        ) if total_time else 0,
        "average_latency_seconds": round(
            statistics.mean(latencies), 3
        ) if latencies else None,
        "p50_latency_seconds": round(
            statistics.median(latencies), 3
        ) if latencies else None,
        "p95_latency_seconds": round(
            sorted(latencies)[min(
                len(latencies) - 1,
                int(len(latencies) * 0.95)
            )], 3
        ) if latencies else None,
    }

    print("\n===== LOAD TEST REPORT =====")
    print(json.dumps(report, indent=2))

    filename = f"load_test_{engine}.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["transformers", "ollama"],
        default="ollama",
    )
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("Requests and concurrency must be at least 1")

    asyncio.run(
        run_benchmark(
            args.engine,
            args.requests,
            args.concurrency,
        )
    )
