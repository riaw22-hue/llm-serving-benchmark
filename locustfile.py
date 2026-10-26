
from locust import HttpUser, task, between
import random


class InferBenchUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def health_check(self):
        self.client.get("/health", name="Health Check")

    @task(3)
    def generate_text(self):
        self.client.post(
            "/generate",
            json={
                "prompt": random.choice([
                    "Explain artificial intelligence in simple terms.",
                    "What is a large language model?",
                    "Explain how GPUs accelerate AI inference."
                ]),
                "max_new_tokens": 30,
                "engine": "ollama"
            },
            name="Generate - Ollama"
        )
