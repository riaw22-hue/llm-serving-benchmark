
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from gateway.main import app

PROMPT = "Explain artificial intelligence in simple words."
MAX_NEW_TOKENS = 50
WARMUP_REQUESTS = 1
BENCHMARK_REQUESTS = 5

RESULTS_FILE = Path("benchmark_results.csv")


def run_benchmark():
    latencies = []
    words_per_second_values = []

    with TestClient(app) as client:
        print("\nWarming up the model...")

        warmup = client.post(
            "/generate",
            json={
                "prompt": PROMPT,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        )

        if warmup.status_code != 200:
            print("Warm-up failed:", warmup.text)
            return

        print("Warm-up completed!")
        print(f"Running {BENCHMARK_REQUESTS} benchmark requests...\n")

        for i in range(BENCHMARK_REQUESTS):
            response = client.post(
                "/generate",
                json={
                    "prompt": PROMPT,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            )

            if response.status_code != 200:
                print(f"Request {i + 1} failed:", response.text)
                continue

            data = response.json()
            latency = data["latency_seconds"]
            speed = data["words_per_second"]

            latencies.append(latency)
            words_per_second_values.append(speed)

            print(
                f"Request {i + 1}: "
                f"Latency = {latency:.3f}s, "
                f"Words/sec = {speed:.2f}"
            )

    if not latencies:
        print("No successful benchmark requests.")
        return

    average_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    average_speed = statistics.mean(words_per_second_values)

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Successful requests: {len(latencies)}")
    print(f"Average latency: {average_latency:.3f}s")
    print(f"Median latency: {median_latency:.3f}s")
    print(f"Minimum latency: {min_latency:.3f}s")
    print(f"Maximum latency: {max_latency:.3f}s")
    print(f"Average words/sec: {average_speed:.2f}")
    print("Device: Check the startup output for cuda:0")
    print("=======================================\n")

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model",
                "device",
                "successful_requests",
                "average_latency_seconds",
                "median_latency_seconds",
                "min_latency_seconds",
                "max_latency_seconds",
                "average_words_per_second",
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            data.get("model", "Qwen/Qwen2.5-1.5B-Instruct"),
            "cuda:0" if data.get("cuda_available") else "unknown",
            len(latencies),
            round(average_latency, 3),
            round(median_latency, 3),
            round(min_latency, 3),
            round(max_latency, 3),
            round(average_speed, 2),
        ])

    print(f"Results saved to: {RESULTS_FILE.resolve()}")


if __name__ == "__main__":
    run_benchmark()
