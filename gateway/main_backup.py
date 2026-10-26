
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import time
import torch

app = FastAPI(
    title="LLM Serving Benchmark",
    description="API for testing LLM inference performance",
    version="1.0.0"
)

# CORS: Allow the React/Vite frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_pipe = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=512)


@app.on_event("startup")
def load_model():
    global model_pipe

    print("Loading Qwen AI model...")

    model_pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto",
        dtype="auto",
    )

    print("Qwen AI model loaded successfully!")

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "GB"
        )

    print("Pipeline device:", model_pipe.device)

    print(
        "Model device map:",
        getattr(model_pipe.model, "hf_device_map", "not exposed")
    )


@app.get("/")
def home():
    return {
        "message": "LLM Serving Benchmark API is running!",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_pipe is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "Not detected"
        )
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    if model_pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet"
        )

    start_time = time.perf_counter()

    try:
        messages = [
            {"role": "user", "content": request.prompt}
        ]

        result = model_pipe(
            messages,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
        )

        answer = result[0]["generated_text"][-1]["content"]

        latency = time.perf_counter() - start_time

        # Approximate word count, not actual model tokens
        estimated_words = len(answer.split())

        words_per_second = (
            estimated_words / latency if latency > 0 else 0
        )

        return {
            "engine": "transformers",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "response": answer,
            "latency_seconds": round(latency, 3),
            "estimated_words": estimated_words,
            "words_per_second": round(words_per_second, 2),
            "cuda_available": torch.cuda.is_available()
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )
