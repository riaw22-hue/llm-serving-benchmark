
import asyncio
import json
import math
import subprocess
import time
from collections import deque
from threading import Thread, Lock

import psutil
import requests
import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline, TextIteratorStreamer


app = FastAPI(
    title="LLM Serving Benchmark",
    description="LLM inference gateway with streaming, benchmarking and system monitoring",
    version="1.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{port}"
        for port in (5173, 5174, 5175, 5176)
    ] + [
        f"http://127.0.0.1:{port}"
        for port in (5173, 5174, 5175, 5176)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIGURATION ----------------

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434"
LLAMA_CPP_URL = "http://127.0.0.1:8081"
LLAMA_CPP_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

model_pipe = None
benchmark_history = deque(maxlen=1000)
history_lock = Lock()


# ---------------- REQUEST MODEL ----------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    engine: str = Field(default="llama.cpp")


# ---------------- HELPER FUNCTIONS ----------------

def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * p / 100
    low = math.floor(index)
    high = math.ceil(index)

    if low == high:
        return round(values[low], 3)

    result = (
        values[low] * (high - index)
        + values[high] * (index - low)
    )
    return round(result, 3)


def save_record(record):
    with history_lock:
        benchmark_history.append(record)


def get_next_stream_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


# ---------------- MODEL STARTUP AND WARMUP ----------------

@app.on_event("startup")
def load_model():
    global model_pipe

    print("Loading Transformers model...")

    model_pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        dtype="auto",
    )

    print("Transformers model loaded!")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Warming up Transformers model...")

    messages = [{"role": "user", "content": "Say hello briefly."}]
    inputs = model_pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model_pipe.model.device)

    with torch.inference_mode():
        model_pipe.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=model_pipe.tokenizer.eos_token_id,
        )

    print("Transformers warm-up complete!")


# ---------------- HOME ----------------

@app.get("/")
def home():
    return {
        "message": "LLM Serving Benchmark API is running!",
        "version": "1.6.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "system_metrics": "/system-metrics",
        "websocket": "/ws/generate",
        "engines": ["llama.cpp", "ollama", "transformers"],
        "default_engine": "llama.cpp",
        "fallback_engine": "ollama",
    }


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    ollama_available = False
    llama_cpp_available = False

    try:
        response = requests.get(
            f"{LLAMA_CPP_URL}/health",
            timeout=2,
        )
        llama_cpp_available = response.ok
    except requests.RequestException:
        pass

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3,
        )
        ollama_available = response.ok
    except requests.RequestException:
        pass

    return {
        "status": "healthy",
        "transformers_loaded": model_pipe is not None,
        "llama_cpp_available": llama_cpp_available,
        "llama_cpp_url": LLAMA_CPP_URL,
        "llama_cpp_model": LLAMA_CPP_MODEL,
        "transformers_model": MODEL_ID,
        "ollama_available": ollama_available,
        "ollama_model": OLLAMA_MODEL,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not detected"
        ),
    }


# ---------------- GPU MONITORING ----------------

def get_nvidia_smi_metrics():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )

        first_line = result.stdout.strip().splitlines()[0]
        values = [
            float(value.strip())
            for value in first_line.split(",")
        ]

        utilization, memory_used, memory_total, temperature = values

        return {
            "gpu_utilization_percent": utilization,
            "gpu_memory_used_mb": round(memory_used, 2),
            "gpu_memory_total_mb": round(memory_total, 2),
            "gpu_temperature_celsius": temperature,
            "gpu_monitoring_source": "nvidia-smi",
        }

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
        IndexError,
    ):
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_celsius": None,
            "gpu_monitoring_source": "unavailable",
        }


# ---------------- SYSTEM METRICS ----------------

@app.get("/system-metrics")
def system_metrics():
    memory = psutil.virtual_memory()

    result = {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024 ** 3), 2),
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_celsius": None,
        "gpu_monitoring_source": "unavailable",
    }

    result.update(get_nvidia_smi_metrics())

    if torch.cuda.is_available():
        try:
            result["gpu_memory_reserved_mb"] = round(
                torch.cuda.memory_reserved(0) / (1024 ** 2), 2
            )

            if result["gpu_memory_total_mb"] is None:
                result["gpu_memory_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024 ** 2),
                    2,
                )

            if result["gpu_memory_used_mb"] is None:
                result["gpu_memory_used_mb"] = round(
                    torch.cuda.memory_allocated(0) / (1024 ** 2), 2
                )

        except Exception as exc:
            result["gpu_error"] = str(exc)

    return result


# ---------------- TRANSFORMERS GENERATION ----------------

def generate_transformers(request):
    if model_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Transformers model is not loaded",
        )

    tokenizer = model_pipe.tokenizer
    model = model_pipe.model

    messages = [{"role": "user", "content": request.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=300.0,
    )

    kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": request.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    start = time.perf_counter()
    first_chunk = None
    chunks = []
    errors = []

    def run_model():
        try:
            with torch.inference_mode():
                model.generate(**kwargs)
        except Exception as exc:
            errors.append(str(exc))
            try:
                streamer.end()
            except Exception:
                pass

    worker = Thread(target=run_model, daemon=True)
    worker.start()

    try:
        for chunk in streamer:
            if chunk and first_chunk is None:
                first_chunk = time.perf_counter()
            chunks.append(chunk)
    except Exception as exc:
        errors.append(str(exc))

    worker.join(timeout=300)

    if worker.is_alive():
        raise HTTPException(
            status_code=504,
            detail="Generation timed out",
        )

    if errors:
        raise RuntimeError(errors[0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    answer = "".join(chunks).strip()
    latency = end - start
    ttft = first_chunk - start if first_chunk else latency

    output_tokens = len(
        tokenizer.encode(answer, add_special_tokens=False)
    )

    generation_time = end - first_chunk if first_chunk else latency

    record = {
        "engine": "transformers",
        "model": MODEL_ID,
        "latency_seconds": round(latency, 3),
        "ttft_seconds": round(ttft, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(
            output_tokens / generation_time, 2
        ) if generation_time > 0 else 0,
    }

    save_record(record)

    return {
        **record,
        "response": answer,
        "cuda_available": torch.cuda.is_available(),
        "fallback_used": False,
    }


# ---------------- LLAMA.CPP GENERATION ----------------

def generate_llama_cpp(request):
    start = time.perf_counter()

    payload = {
        "prompt": request.prompt,
        "n_predict": request.max_new_tokens,
        "temperature": 0,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{LLAMA_CPP_URL}/completion",
            json=payload,
            timeout=(5, 600),
        )
        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"llama.cpp request failed: {exc}"
        ) from exc

    end = time.perf_counter()
    answer = str(data.get("content", "")).strip()
    timings = data.get("timings") or {}

    input_tokens = int(
        data.get(
            "tokens_evaluated",
            timings.get("prompt_n", 0),
        ) or 0
    )

    output_tokens = int(
        data.get(
            "tokens_predicted",
            timings.get("predicted_n", 0),
        ) or 0
    )

    generation_ms = float(
        timings.get("predicted_ms", 0) or 0
    )

    tokens_per_second = (
        output_tokens / (generation_ms / 1000)
        if generation_ms > 0 else 0
    )

    latency = end - start

    record = {
        "engine": "llama.cpp",
        "model": LLAMA_CPP_MODEL,
        "latency_seconds": round(latency, 3),
        "ttft_seconds": round(latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
    }

    save_record(record)

    return {
        **record,
        "response": answer,
        "cuda_available": torch.cuda.is_available(),
        "fallback_used": False,
    }


# ---------------- OLLAMA GENERATION ----------------

def generate_ollama(request):
    start = time.perf_counter()
    first_chunk = None
    answer_parts = []
    final_data = {}

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": request.prompt,
        "stream": True,
        "options": {
            "num_predict": request.max_new_tokens,
            "temperature": 0,
        },
    }

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, 600),
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)
                text_part = data.get("response", "")

                if text_part:
                    if first_chunk is None:
                        first_chunk = time.perf_counter()
                    answer_parts.append(text_part)

                if data.get("done"):
                    final_data = data

    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running. "
            f"Details: {exc}"
        ) from exc

    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid response from Ollama: {exc}"
        ) from exc

    end = time.perf_counter()
    answer = "".join(answer_parts).strip()
    latency = end - start
    ttft = first_chunk - start if first_chunk else latency

    input_tokens = final_data.get("prompt_eval_count", 0)
    output_tokens = final_data.get("eval_count", 0)
    generation_ns = final_data.get("eval_duration", 0)

    tokens_per_second = (
        output_tokens / (generation_ns / 1_000_000_000)
        if generation_ns else 0
    )

    record = {
        "engine": "ollama",
        "model": OLLAMA_MODEL,
        "latency_seconds": round(latency, 3),
        "ttft_seconds": round(ttft, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
    }

    save_record(record)

    return {
        **record,
        "response": answer,
        "cuda_available": torch.cuda.is_available(),
        "fallback_used": False,
    }


# ---------------- NORMAL GENERATION API ----------------

@app.post("/generate")
def generate(request: GenerateRequest):
    engine = request.engine.strip().lower()

    try:
        if engine in (
            "llama.cpp",
            "llama_cpp",
            "llama-cpp",
            "primary",
        ):
            try:
                return generate_llama_cpp(request)

            except Exception as llama_error:
                try:
                    result = generate_ollama(request)
                    result["requested_engine"] = "llama.cpp"
                    result["fallback_used"] = True
                    result["fallback_reason"] = str(llama_error)
                    return result

                except Exception as ollama_error:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"llama.cpp failed ({llama_error}); "
                            "Ollama fallback also failed "
                            f"({ollama_error})"
                        ),
                    ) from ollama_error

        elif engine == "transformers":
            return generate_transformers(request)

        elif engine == "ollama":
            return generate_ollama(request)

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid engine. Choose 'llama.cpp', "
                    "'ollama', or 'transformers'."
                ),
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {exc}",
        )


# ---------------- WEBSOCKET LIVE STREAMING ----------------
# Connect to ws://127.0.0.1:8000/ws/generate
# Send JSON: {"prompt":"Explain AI","max_new_tokens":50,"engine":"transformers"}
# Events: start, token, done, error

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()

        prompt = str(data.get("prompt", "")).strip()
        max_new_tokens = int(data.get("max_new_tokens", 100))
        engine = str(data.get("engine", "transformers")).lower().strip()

        if not prompt:
            await websocket.send_json({
                "type": "error",
                "message": "Prompt cannot be empty.",
            })
            return

        if not 1 <= max_new_tokens <= 512:
            await websocket.send_json({
                "type": "error",
                "message": "max_new_tokens must be between 1 and 512.",
            })
            return

        if engine != "transformers":
            await websocket.send_json({
                "type": "error",
                "message": (
                    "WebSocket streaming currently supports "
                    "transformers only. Use POST /generate "
                    "for llama.cpp or Ollama."
                ),
            })
            return

        if model_pipe is None:
            await websocket.send_json({
                "type": "error",
                "message": "Transformers model is not loaded.",
            })
            return

        tokenizer = model_pipe.tokenizer
        model = model_pipe.model

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        input_tokens = inputs["input_ids"].shape[-1]

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=300.0,
        )

        generation_args = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }

        start = time.perf_counter()
        first_chunk = None
        chunks = []
        errors = []

        await websocket.send_json({
            "type": "start",
            "engine": "transformers",
            "model": MODEL_ID,
            "input_tokens": input_tokens,
        })

        def run_generation():
            try:
                with torch.inference_mode():
                    model.generate(**generation_args)
            except Exception as exc:
                errors.append(str(exc))
                try:
                    streamer.end()
                except Exception:
                    pass

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        iterator = iter(streamer)

        while True:
            chunk = await asyncio.to_thread(
                get_next_stream_chunk, iterator
            )

            if chunk is None:
                break

            if chunk:
                if first_chunk is None:
                    first_chunk = time.perf_counter()

                chunks.append(chunk)

                await websocket.send_json({
                    "type": "token",
                    "text": chunk,
                })

        await asyncio.to_thread(worker.join, 300)

        if worker.is_alive():
            await websocket.send_json({
                "type": "error",
                "message": "Generation timed out.",
            })
            return

        if errors:
            raise RuntimeError(errors[0])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        answer = "".join(chunks).strip()
        latency = end - start
        ttft = first_chunk - start if first_chunk else latency

        output_tokens = len(
            tokenizer.encode(answer, add_special_tokens=False)
        )

        generation_time = end - first_chunk if first_chunk else latency

        record = {
            "engine": "transformers",
            "model": MODEL_ID,
            "latency_seconds": round(latency, 3),
            "ttft_seconds": round(ttft, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": round(
                output_tokens / generation_time, 2
            ) if generation_time > 0 else 0,
        }

        save_record(record)

        await websocket.send_json({
            "type": "done",
            **record,
            "response": answer,
        })

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")

    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
            })
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------- BENCHMARK METRICS ----------------

@app.get("/metrics")
def metrics(engine: str = Query(default="all")):
    engine = engine.strip().lower()

    if engine not in (
        "all",
        "transformers",
        "ollama",
        "llama.cpp",
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "engine must be all, llama.cpp, transformers, or ollama"
            ),
        )

    with history_lock:
        records = list(benchmark_history)

    if engine != "all":
        records = [
            record for record in records
            if record["engine"] == engine
        ]

    latencies = [
        record["latency_seconds"] for record in records
    ]

    ttfts = [
        record["ttft_seconds"] for record in records
        if record["ttft_seconds"] is not None
    ]

    total_input = sum(
        record["input_tokens"] for record in records
    )
    total_output = sum(
        record["output_tokens"] for record in records
    )
    total_time = sum(latencies)

    return {
        "engine": engine,
        "total_requests": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "average_latency_seconds": round(
            sum(latencies) / len(latencies), 3
        ) if latencies else 0,
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "p99_latency_seconds": percentile(latencies, 99),
        "average_ttft_seconds": round(
            sum(ttfts) / len(ttfts), 3
        ) if ttfts else None,
        "aggregate_output_tokens_per_second": round(
            total_output / total_time, 2
        ) if total_time > 0 else 0,
        "models": sorted(
            set(record["model"] for record in records)
        ),
    }
