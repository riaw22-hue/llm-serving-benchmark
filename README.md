\# InferBench — LLM Serving Benchmark \& Inference Gateway



A local benchmarking platform for comparing Large Language Model (LLM) inference engines through a unified FastAPI gateway and an interactive React dashboard.



InferBench helps measure model inference performance, track request-level metrics, and compare different inference backends using a consistent API.



\## Overview



Running an LLM locally can produce different performance results depending on the inference engine, model, hardware, and generation settings.



InferBench provides a single interface to send prompts to supported inference backends, collect performance metrics, and visualize benchmark results.



The project currently integrates Hugging Face Transformers and Ollama.



\## Features



\- \*\*Unified Inference API:\*\* Generate text through a FastAPI backend.

\- \*\*Multiple Inference Engines:\*\* Supports Hugging Face Transformers and Ollama.

\- \*\*Performance Metrics:\*\* Track request latency, time to first token (TTFT), input tokens, and output tokens where available.

\- \*\*Benchmark Statistics:\*\* Review aggregate metrics such as average latency, percentile latency, and output throughput.

\- \*\*Interactive Dashboard:\*\* React-based interface for monitoring requests and visualizing performance.

\- \*\*Request History:\*\* Review benchmark activity and export supported results to CSV.

\- \*\*Health Monitoring:\*\* Check backend and model status through a health endpoint.

\- \*\*Load Testing:\*\* Use Locust to simulate concurrent requests and evaluate API behavior under load.



\## Tech Stack



| Component | Technologies |

|---|---|

| Backend API | Python, FastAPI, Pydantic |

| LLM Inference | Hugging Face Transformers, PyTorch, Ollama |

| Models | Qwen2.5-1.5B-Instruct, Qwen2.5 0.5B through Ollama |

| Frontend | React, Vite, JavaScript |

| Data Visualization | Recharts |

| API Testing | FastAPI Swagger UI |

| Load Testing | Locust |

| Version Control | Git, GitHub |



\## Architecture



```text

&#x20;                User

&#x20;                 |

&#x20;                 v

&#x20;       React Dashboard (Vite)

&#x20;                 |

&#x20;                 v

&#x20;         FastAPI Gateway

&#x20;                 |

&#x20;         +-------+-------+

&#x20;         |               |

&#x20;         v               v

&#x20;Hugging Face          Ollama

&#x20;Transformers          Runtime

&#x20;         |               |

&#x20;         v               v

&#x20;  Qwen2.5-1.5B       Qwen2.5 0.5B

&#x20;         |               |

&#x20;         +-------+-------+

&#x20;                 |

&#x20;                 v

&#x20;      Performance Metrics

&#x20;                 |

&#x20;                 v

&#x20;      Dashboard / CSV Export

```



\## Project Structure



```text

llm-serving-benchmark/

├── gateway/

│   ├── main.py

│   └── model\_test.py

├── frontend/

│   ├── src/

│   ├── package.json

│   └── vite.config.js

├── locustfile.py

├── .gitignore

└── README.md

```



\*The exact files may vary as the project evolves.\*



\## Requirements



\- Windows, Linux, or macOS

\- Python 3.10 or a compatible Python version for the installed dependencies

\- Node.js and npm

\- Git

\- Ollama for running the Ollama backend

\- A compatible local model and sufficient RAM/VRAM



GPU acceleration depends on the installed PyTorch build, drivers, and supported hardware. CPU execution may also be possible, but performance will differ.



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/riaw22-hue/llm-serving-benchmark.git

cd llm-serving-benchmark

```



\### 2. Set up the Python environment



On Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script execution, run this in the current terminal:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\\.venv\\Scripts\\Activate.ps1

```



Install the project's required Python dependencies. If a `requirements.txt` file is present, run:



```powershell

python -m pip install -r requirements.txt

```



If the repository does not contain `requirements.txt`, install the dependencies listed in the project setup or create the file from the packages used by the backend.



\### 3. Start Ollama



Install Ollama from its official website:



https://ollama.com/



Download the model used by this project:



```powershell

ollama pull qwen2.5:0.5b

```



Verify that Ollama is running and the model is available:



```powershell

ollama list

```



Ollama must be running locally for requests using the Ollama backend.



\### 4. Start the FastAPI backend



From the project root, use the command matching the actual Python module and FastAPI application object in your repository.



For example, if the application object is `app` inside `gateway/main.py`:



```powershell

python -m uvicorn gateway.main:app --reload

```



The API will normally be available at:



http://127.0.0.1:8000



Interactive API documentation:



http://127.0.0.1:8000/docs



Health endpoint:



http://127.0.0.1:8000/health



\### 5. Start the React dashboard



Open a second terminal:



```powershell

cd frontend

npm install

npm run dev

```



Open the local URL displayed by Vite in your terminal, commonly:



http://localhost:5173



Keep the backend running while using the dashboard. If the frontend cannot connect, verify the API URL and backend CORS configuration.



\## API Usage



\### Health Check



```http

GET /health

```



This endpoint reports the backend's health and relevant model or hardware status supported by the implementation.



\### Generate Text



```http

POST /generate

Content-Type: application/json

```



Example request for the Transformers backend:



```json

{

&#x20; "prompt": "Explain artificial intelligence in two sentences.",

&#x20; "max\_new\_tokens": 30,

&#x20; "engine": "transformers"

}

```



Example request for Ollama:



```json

{

&#x20; "prompt": "Explain artificial intelligence in two sentences.",

&#x20; "max\_new\_tokens": 30,

&#x20; "engine": "ollama"

}

```



The accepted engine names and request fields depend on the current backend implementation. Use `/docs` to inspect the live API schema.



\## Performance Metrics



InferBench is designed to help analyze:



| Metric | Description |

|---|---|

| Request Latency | Total time taken to complete a request |

| Time to First Token (TTFT) | Time until the first generated output becomes available |

| Input Tokens | Number of tokens in the prompt, when measured |

| Output Tokens | Number of generated tokens, when measured |

| Average Latency | Mean request latency across recorded requests |

| P50 Latency | Median request latency |

| P95 Latency | Latency at the 95th percentile |

| P99 Latency | Latency at the 99th percentile |

| Output Throughput | Generated tokens per second, according to the implementation's measurement method |



Actual metrics depend on the engine, model, hardware, generation settings, and measurement methodology. Compare engines using the same workload wherever possible.



\## Load Testing with Locust



Locust can simulate multiple users sending requests to the inference API.



If `locustfile.py` is configured for the project's current API, start the load test from the project root:



```powershell

python -m pip install locust

python -m locust -f locustfile.py --host http://127.0.0.1:8000

```



Open the local web interface shown by Locust, commonly:



http://localhost:8089



Start with a small number of users and increase concurrency gradually. Local model inference can consume significant memory and GPU resources.



Record the number of users, request rate, failure rate, latency percentiles, model, engine, and generation settings when comparing results.



\## Reproducible Benchmarking



For a meaningful comparison:



1\. Use the same prompt set and generation settings.

2\. Warm up each engine before collecting measurements.

3\. Run multiple requests for each configuration.

4\. Keep the hardware and background workload as consistent as possible.

5\. Record latency, TTFT, throughput, and request failures.

6\. Report the actual results instead of assuming one engine is faster.



Transformers and Ollama may use different model sizes and runtime configurations in this project, so results should include the model and engine used for each test.



\## Limitations



\- Performance depends on the available hardware and model configuration.

\- Large models may not fit into limited GPU memory.

\- Different engines and model sizes are not directly comparable without clearly stated test conditions.

\- Results from a local laptop should not be treated as production-server benchmarks.

\- vLLM and llama.cpp should only be considered supported once their integrations have been implemented and tested.



\## Future Improvements



\- Add vLLM and llama.cpp backend integrations.

\- Add repeatable benchmark suites and downloadable result reports.

\- Compare throughput across controlled concurrency levels.

\- Add automated benchmark runs and historical result comparisons.

\- Improve error handling, configuration management, and automated tests.

\- Add deployment support for a suitable server environment.



\## Author



\*\*Riya Wadhe\*\*



GitHub: https://github.com/riaw22-hue



\## License



This project is intended for educational and portfolio use. Add a `LICENSE` file to specify the terms under which others may use, modify, and distribute the code.

