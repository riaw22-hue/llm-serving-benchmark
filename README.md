\# InferBench — LLM Serving Benchmark \& Inference Gateway



A local benchmarking platform for comparing Large Language Model (LLM) inference engines through a unified FastAPI gateway and an interactive React dashboard.



InferBench helps measure inference performance, monitor request-level metrics, and compare supported inference backends through a consistent API.



\## Overview



Large Language Models can behave differently depending on the inference engine, model size, hardware, and generation settings.



InferBench provides a unified interface for sending prompts to supported inference backends, collecting performance metrics, and visualizing benchmark results through a web dashboard.



The project currently integrates Hugging Face Transformers and Ollama.



\## Features



\- \*\*Unified Inference API:\*\* Generate text through a FastAPI gateway.

\- \*\*Multiple Inference Engines:\*\* Supports Transformers and Ollama.

\- \*\*Performance Monitoring:\*\* Track request latency and time to first token (TTFT), where available.

\- \*\*Token Metrics:\*\* Track input and output token counts supported by the implementation.

\- \*\*Benchmark Statistics:\*\* Review average, P50, P95, and P99 latency, along with output throughput.

\- \*\*Interactive Dashboard:\*\* Visualize inference performance using a React frontend.

\- \*\*Request History:\*\* Review recorded requests and export supported results to CSV.

\- \*\*Health Check:\*\* Monitor backend and model status.

\- \*\*Load Testing:\*\* Use Locust to simulate concurrent API requests.



\## Tech Stack



| Component | Technology |

|---|---|

| Programming Language | Python, JavaScript |

| Backend | FastAPI, Pydantic |

| LLM Inference | Hugging Face Transformers, PyTorch, Ollama |

| Models | Qwen2.5-1.5B-Instruct, Qwen2.5 0.5B |

| Frontend | React, Vite |

| Data Visualization | Recharts |

| API Documentation | Swagger UI |

| Load Testing | Locust |

| Version Control | Git, GitHub |



\## System Architecture



```text

&#x20;                    User

&#x20;                     |

&#x20;                     v

&#x20;            React Dashboard

&#x20;               (Vite)

&#x20;                     |

&#x20;                     v

&#x20;              FastAPI Gateway

&#x20;                     |

&#x20;             +-------+-------+

&#x20;             |               |

&#x20;             v               v

&#x20;       Transformers        Ollama

&#x20;             |               |

&#x20;             v               v

&#x20;     Qwen2.5-1.5B       Qwen2.5 0.5B

&#x20;             |               |

&#x20;             +-------+-------+

&#x20;                     |

&#x20;                     v

&#x20;            Performance Metrics

&#x20;                     |

&#x20;                     v

&#x20;             Dashboard / CSV

```



\## Project Structure



```text

llm-serving-benchmark/

|

|-- gateway/

|   |-- main.py

|   |-- model\_test.py

|

|-- frontend/

|   |-- src/

|   |-- package.json

|   |-- vite.config.js

|

|-- locustfile.py

|-- .gitignore

|-- README.md

```



The exact project structure may change as additional features are developed.



\## Requirements



\- Python and pip

\- Node.js and npm

\- Git

\- Ollama for the Ollama inference backend

\- A compatible local model

\- Sufficient RAM and GPU memory for the selected model



GPU acceleration depends on the installed PyTorch build, NVIDIA drivers, and compatible hardware. CPU execution may also be possible, with different performance.



\## Installation



\### 1. Clone the Repository



```bash

git clone https://github.com/riaw22-hue/llm-serving-benchmark.git

cd llm-serving-benchmark

```



\### 2. Create a Python Virtual Environment



On Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script execution, run:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\\.venv\\Scripts\\Activate.ps1

```



Install the Python dependencies using the project's dependency file, if available:



```powershell

python -m pip install -r requirements.txt

```



If `requirements.txt` is not present, install the dependencies used by the backend or add a dependency file to the repository.



\### 3. Set Up Ollama



Install Ollama from its official website:



https://ollama.com/



Download the model:



```powershell

ollama pull qwen2.5:0.5b

```



Verify the installed model:



```powershell

ollama list

```



Make sure the Ollama service is running before sending requests to the Ollama backend.



\### 4. Start the FastAPI Backend



From the project root, run the command matching the location of your FastAPI application.



If the application object is named `app` inside `gateway/main.py`, use:



```powershell

python -m uvicorn gateway.main:app --reload

```



The backend will normally run at:



http://127.0.0.1:8000



Interactive API documentation:



http://127.0.0.1:8000/docs



Health endpoint:



http://127.0.0.1:8000/health



Keep the backend terminal running while using the frontend.



\### 5. Start the React Frontend



Open a second terminal:



```powershell

cd frontend

npm install

npm run dev

```



Open the local URL displayed by Vite in the terminal. It is commonly:



http://localhost:5173



If the frontend cannot connect to the backend, check that the backend is running and the API URL and CORS settings are correct.



\## API Usage



\### Health Check



\*\*Endpoint:\*\* `GET /health`



This endpoint reports backend health and the model or hardware status provided by the implementation.



\### Generate Text



\*\*Endpoint:\*\* `POST /generate`



\*\*Content-Type:\*\* `application/json`



\#### Example: Transformers



```json

{

&#x20; "prompt": "Explain artificial intelligence in two sentences.",

&#x20; "max\_new\_tokens": 30,

&#x20; "engine": "transformers"

}

```



\#### Example: Ollama



```json

{

&#x20; "prompt": "Explain artificial intelligence in two sentences.",

&#x20; "max\_new\_tokens": 30,

&#x20; "engine": "ollama"

}

```



The accepted engine names and request parameters depend on the backend implementation. Visit `/docs` to see the actual request schema.



\## Performance Metrics



InferBench is designed to help evaluate inference performance using metrics such as:



| Metric | Description |

|---|---|

| Request Latency | Total time taken to complete a request |

| Time to First Token (TTFT) | Time until the first generated output becomes available |

| Input Tokens | Number of tokens in the input prompt, when measured |

| Output Tokens | Number of generated tokens, when measured |

| Average Latency | Mean request latency |

| P50 Latency | Median request latency |

| P95 Latency | Latency at the 95th percentile |

| P99 Latency | Latency at the 99th percentile |

| Output Throughput | Generated tokens per second, according to the implementation's measurement method |



Actual results depend on the selected model, inference engine, hardware, prompt, generation settings, and measurement methodology.



\## Load Testing with Locust



Locust can simulate multiple users sending requests to the inference API.



Install Locust:



```powershell

python -m pip install locust

```



From the project root, run the following command if `locustfile.py` is configured for the current API:



```powershell

python -m locust -f locustfile.py --host http://127.0.0.1:8000

```



Open the URL shown by Locust, commonly:



http://localhost:8089



Start with a small number of users and increase concurrency gradually. Local LLM inference can use significant CPU, GPU, and memory resources.



For each test, record the engine, model, number of users, request rate, failure rate, latency percentiles, and generation settings.



\## Benchmarking Methodology



For a fair comparison between inference engines:



1\. Use the same prompts and generation settings where possible.

2\. Warm up each engine before measuring performance.

3\. Run multiple requests for each configuration.

4\. Keep hardware and background workloads as consistent as possible.

5\. Record latency, TTFT, throughput, and request failures.

6\. Document the model and engine used for every result.

7\. Report measured results rather than assuming one engine is faster.



The Transformers and Ollama configurations may use different model sizes and runtime implementations. Document these differences when interpreting results.



\## Limitations



\- Performance depends on hardware, model size, and runtime configuration.

\- Larger models may exceed available GPU memory.

\- Results from different model sizes are not directly comparable without explaining the test conditions.

\- Laptop benchmark results may differ significantly from production-server results.

\- vLLM and llama.cpp are not claimed as supported backends in the current implementation.



\## Future Improvements



\- Integrate additional inference engines such as vLLM and llama.cpp.

\- Add repeatable benchmark suites and automated result reports.

\- Compare throughput under controlled concurrency levels.

\- Add historical benchmark comparisons.

\- Expand automated testing and error handling.

\- Add deployment support for suitable server environments.



\## Author



\*\*Riya Wadhe\*\*



GitHub: https://github.com/riaw22-hue



\## License



This project is intended for educational and portfolio purposes. A `LICENSE` file can be added to define the permissions for using, modifying, and distributing the project.

