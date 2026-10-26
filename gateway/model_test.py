
import torch
from transformers import pipeline

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading AI model...")

pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
    torch_dtype="auto",
)

messages = [
    {
        "role": "user",
        "content": "Explain artificial intelligence in simple words."
    }
]

result = pipe(
    messages,
    max_new_tokens=100,
    do_sample=False,
)

print("\nAI Response:")
print(result[0]["generated_text"][-1]["content"])
