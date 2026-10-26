
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  ArrowUpRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Cpu,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";
const HISTORY_LIMIT = 50;
const MAX_TOKENS = 1000;

const ENGINES = {
  ollama: {
    name: "Ollama",
    model: "qwen2.5:0.5b",
  },
  transformers: {
    name: "Transformers",
    model: "Qwen2.5-1.5B-Instruct",
  },
};

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits = 2) {
  const n = numberOrNull(value);
  return n === null ? "—" : n.toFixed(digits);
}

function percent(value) {
  const n = numberOrNull(value);
  return n === null ? "—" : `${n.toFixed(1)}%`;
}

function firstDefined(...values) {
  return values.find((value) => value !== null && value !== undefined);
}

function extractOutput(data) {
  if (data === null || data === undefined) {
    return "No response received from the backend.";
  }

  const keys = [
    "generated_text",
    "response",
    "output",
    "text",
    "result",
    "content",
    "message",
    "answer",
  ];

  // Prefer a normal text value at the top level.
  for (const key of keys) {
    if (typeof data?.[key] === "string") {
      return data[key];
    }
  }

  // Some APIs wrap the generated text inside nested objects or arrays.
  const findText = (value, depth = 0) => {
    if (typeof value === "string") return value;
    if (value === null || value === undefined || depth > 6) return null;

    if (Array.isArray(value)) {
      const parts = value
        .map((item) => findText(item, depth + 1))
        .filter((item) => typeof item === "string" && item.trim());

      return parts.length ? parts.join("\n") : null;
    }

    if (typeof value === "object") {
      for (const key of keys) {
        if (typeof value[key] === "string") {
          return value[key];
        }
      }

      for (const key of keys) {
        if (value[key] !== undefined && value[key] !== null) {
          const nested = findText(value[key], depth + 1);
          if (nested !== null) return nested;
        }
      }
    }

    return null;
  };

  for (const key of keys) {
    const candidate = data?.[key];
    if (candidate !== undefined && candidate !== null) {
      const nestedText = findText(candidate);
      if (nestedText !== null) return nestedText;
    }
  }

  // JSON.stringify prevents JavaScript's "[object Object]" display.
  try {
    return JSON.stringify(data, null, 2);
  } catch {
    return String(data);
  }
}

function extractMetric(data, ...keys) {
  for (const key of keys) {
    const value = numberOrNull(data?.[key]);
    if (value !== null) return value;
  }
  return null;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const raw = await response.text();

  let data = {};
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    data = { detail: raw };
  }

  if (!response.ok) {
    throw new Error(
      data?.detail || data?.error || `Request failed (${response.status})`
    );
  }

  return data;
}

function StatCard({ label, value, detail, icon: Icon, accent = "violet" }) {
  return (
    <article className={`stat-card accent-${accent}`}>
      <div className="stat-card-top">
        <span>{label}</span>
        <span className="stat-icon"><Icon size={18} /></span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </article>
  );
}

function SectionHeading({ eyebrow, title, icon: Icon, action }) {
  return (
    <div className="section-heading">
      <div className="section-heading-left">
        {Icon && <span className="section-icon"><Icon size={18} /></span>}
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
        </div>
      </div>
      {action}
    </div>
  );
}

function EmptyChart({ message }) {
  return (
    <div className="chart-empty">
      <BarChart3 size={28} />
      <p>{message}</p>
    </div>
  );
}

export default function App() {
  const [engine, setEngine] = useState("ollama");
  const [prompt, setPrompt] = useState(
    "Explain how a large language model generates a response."
  );
  const [maxTokens, setMaxTokens] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [iterations, setIterations] = useState(5);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [health, setHealth] = useState(null);
  const [serverMetrics, setServerMetrics] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const [history, setHistory] = useState([]);
  const [historyFilter, setHistoryFilter] = useState("all");
  const [activePage, setActivePage] = useState("overview");
  const [chartMetric, setChartMetric] = useState("latency");

  const refreshData = useCallback(async () => {
    try {
      const data = await fetchJSON(`${API}/health`);
      setHealth(data);
      setBackendOnline(true);
    } catch {
      setHealth(null);
      setBackendOnline(false);
    }

    try {
      const data = await fetchJSON(`${API}/metrics`);
      setServerMetrics(data);
      setMetricsError("");
    } catch {
      setMetricsError("Metrics endpoint is unavailable.");
    }

    try {
      const data = await fetchJSON(`${API}/system-metrics`);
      setSystemMetrics(data);
    } catch {
      setSystemMetrics(null);
    }
  }, []);

  useEffect(() => {
    refreshData();
    const timer = setInterval(refreshData, 4000);
    return () => clearInterval(timer);
  }, [refreshData]);

  const runInference = async (event) => {
    event.preventDefault();

    if (!prompt.trim()) {
      setError("Enter a prompt before running inference.");
      return;
    }

    const requestedTokens = Math.max(
      1,
      Math.min(MAX_TOKENS, Number(maxTokens) || 1)
    );

    setMaxTokens(requestedTokens);
    setLoading(true);
    setError("");
    setNotice("");
    setOutput("");

    const start = performance.now();
    const requestEngine = engine;

    try {
      const data = await fetchJSON(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engine: requestEngine,
          prompt: prompt.trim(),
          max_new_tokens: requestedTokens,
        }),
      });

      const elapsed = (performance.now() - start) / 1000;
      const text = extractOutput(data);

      const record = {
        id: `${Date.now()}-${Math.random()}`,
        engine: requestEngine,
        prompt: prompt.trim(),
        output: text,
        timestamp: new Date().toISOString(),
        latency:
          extractMetric(
            data,
            "latency_seconds",
            "total_latency_seconds",
            "latency"
          ) ?? elapsed,
        ttft: extractMetric(
          data,
          "ttft_seconds",
          "time_to_first_token_seconds",
          "average_ttft_seconds"
        ),
        inputTokens: extractMetric(
          data,
          "total_input_tokens",
          "input_tokens",
          "prompt_tokens"
        ),
        outputTokens: extractMetric(
          data,
          "total_output_tokens",
          "output_tokens",
          "completion_tokens"
        ),
        tokensPerSecond: extractMetric(
          data,
          "tokens_per_second",
          "output_tokens_per_second"
        ),
        success: true,
      };

      setOutput(text);
      setHistory((previous) =>
        [record, ...previous].slice(0, HISTORY_LIMIT)
      );
      setNotice("Inference completed successfully.");
      await refreshData();
    } catch (err) {
      const elapsed = (performance.now() - start) / 1000;
      const message =
        err?.message || "Unable to reach the inference server.";

      setError(message);
      setHistory((previous) =>
        [
          {
            id: `${Date.now()}-${Math.random()}`,
            engine: requestEngine,
            prompt: prompt.trim(),
            output: "",
            timestamp: new Date().toISOString(),
            latency: elapsed,
            ttft: null,
            inputTokens: null,
            outputTokens: null,
            tokensPerSecond: null,
            success: false,
            error: message,
          },
          ...previous,
        ].slice(0, HISTORY_LIMIT)
      );
    } finally {
      setLoading(false);
    }
  };

  const runBenchmark = async () => {
    if (!prompt.trim()) {
      setError("Enter a prompt before running the benchmark.");
      return;
    }

    const totalRuns = Number(iterations);
    const requestedTokens = Math.max(
      1,
      Math.min(MAX_TOKENS, Number(maxTokens) || 1)
    );
    const requestEngine = engine;
    const requestPrompt = prompt.trim();
    const benchmarkRecords = [];
    let lastOutput = "";

    setLoading(true);
    setBenchmarkRunning(true);
    setError("");
    setNotice("");
    setOutput("");
    setMaxTokens(requestedTokens);

    try {
      for (let i = 0; i < totalRuns; i += 1) {
        setNotice(`Benchmark running: ${i + 1} of ${totalRuns}...`);
        const start = performance.now();

        const data = await fetchJSON(`${API}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            engine: requestEngine,
            prompt: requestPrompt,
            max_new_tokens: requestedTokens,
          }),
        });

        const elapsed = (performance.now() - start) / 1000;
        lastOutput = extractOutput(data);
        benchmarkRecords.push({
          id: `${Date.now()}-${i}-${Math.random()}`,
          engine: requestEngine,
          prompt: requestPrompt,
          output: lastOutput,
          timestamp: new Date().toISOString(),
          latency:
            extractMetric(data, "latency_seconds", "total_latency_seconds", "latency") ?? elapsed,
          ttft: extractMetric(data, "ttft_seconds", "time_to_first_token_seconds", "average_ttft_seconds"),
          inputTokens: extractMetric(data, "total_input_tokens", "input_tokens", "prompt_tokens"),
          outputTokens: extractMetric(data, "total_output_tokens", "output_tokens", "completion_tokens"),
          tokensPerSecond: extractMetric(data, "tokens_per_second", "output_tokens_per_second"),
          success: true,
        });
      }

      setOutput(lastOutput);
      setHistory((previous) => [...benchmarkRecords, ...previous].slice(0, HISTORY_LIMIT));
      const averageLatency = benchmarkRecords.reduce((sum, item) => sum + item.latency, 0) / benchmarkRecords.length;
      setNotice(`Benchmark complete: ${benchmarkRecords.length} successful runs. Average latency: ${averageLatency.toFixed(2)} seconds.`);
      await refreshData();
    } catch (err) {
      if (benchmarkRecords.length > 0) {
        setHistory((previous) => [...benchmarkRecords, ...previous].slice(0, HISTORY_LIMIT));
        setOutput(lastOutput);
      }
      setError(`Benchmark stopped after ${benchmarkRecords.length} successful run(s): ${err?.message || "Unknown error"}`);
      await refreshData();
    } finally {
      setLoading(false);
      setBenchmarkRunning(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    setNotice("Session request history cleared.");
  };

  const exportCSV = () => {
    if (!history.length) {
      setError("There is no request history to export.");
      return;
    }

    const columns = [
      "timestamp",
      "engine",
      "prompt",
      "latency",
      "ttft",
      "inputTokens",
      "outputTokens",
      "tokensPerSecond",
      "success",
      "error",
    ];

    const escapeCSV = (value) =>
      `"${String(value ?? "").replace(/"/g, '""')}"`;

    const csv = [
      columns.join(","),
      ...history.map((item) =>
        columns.map((column) => escapeCSV(item[column])).join(",")
      ),
    ].join("\r\n");

    const blob = new Blob(["\uFEFF" + csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "inferbench-request-history.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const successful = history.filter((item) => item.success).length;
  const failed = history.filter((item) => !item.success).length;

  const filteredHistory = useMemo(
    () =>
      historyFilter === "all"
        ? history
        : history.filter((item) => item.engine === historyFilter),
    [history, historyFilter]
  );

  const chartData = useMemo(
    () =>
      [...history]
        .filter((item) => item.success)
        .reverse()
        .map((item, index) => ({
          request: `#${index + 1}`,
          latency: item.latency,
          ttft: item.ttft,
          tokensPerSecond: item.tokensPerSecond,
        })),
    [history]
  );

  const serverTotalRequests = extractMetric(
    serverMetrics,
    "total_requests",
    "request_count"
  );

  const averageLatency = extractMetric(
    serverMetrics,
    "average_latency_seconds",
    "avg_latency_seconds"
  );

  const p50 = extractMetric(serverMetrics, "p50_latency_seconds");
  const p95 = extractMetric(serverMetrics, "p95_latency_seconds");
  const p99 = extractMetric(serverMetrics, "p99_latency_seconds");

  const throughput = extractMetric(
    serverMetrics,
    "aggregate_output_tokens_per_second",
    "tokens_per_second",
    "output_tokens_per_second"
  );

  const averageTTFT = extractMetric(
    serverMetrics,
    "average_ttft_seconds",
    "average_ttft",
    "ttft_seconds"
  );

  const cpu = firstDefined(
    systemMetrics?.cpu_percent,
    systemMetrics?.cpu_usage_percent
  );

  const ram = firstDefined(
    systemMetrics?.ram_percent,
    systemMetrics?.memory_percent
  );

  const ramUsed = firstDefined(
    systemMetrics?.ram_used_gb,
    systemMetrics?.memory_used_gb
  );

  const ramTotal = firstDefined(
    systemMetrics?.ram_total_gb,
    systemMetrics?.memory_total_gb
  );

  const gpuName = systemMetrics?.gpu_name || "GPU not detected";
  const gpuAvailable = Boolean(systemMetrics?.gpu_available);

  const gpuAllocated = firstDefined(
    systemMetrics?.gpu_memory_allocated_gb,
    systemMetrics?.gpu_allocated_gb
  );

  const gpuTotal = firstDefined(
    systemMetrics?.gpu_memory_total_gb,
    systemMetrics?.gpu_total_gb
  );

  const gpuUtilization = systemMetrics?.gpu_utilization_percent;

  const navigate = (page) => {
    setActivePage(page);
    document.getElementById(page)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <a
          href="#overview"
          className="brand"
          onClick={() => setActivePage("overview")}
        >
          <span className="brand-mark"><Layers3 size={23} /></span>
          <span className="brand-copy">
            <strong>infer<span>bench</span></strong>
            <small>LLM OBSERVABILITY</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="side-nav">
          <button
            className={activePage === "overview" ? "nav-link selected" : "nav-link"}
            onClick={() => navigate("overview")}
          >
            <Activity size={18} /> Overview
          </button>
          <button
            className={activePage === "playground" ? "nav-link selected" : "nav-link"}
            onClick={() => navigate("playground")}
          >
            <Sparkles size={18} /> Playground
          </button>
          <button
            className={activePage === "analytics" ? "nav-link selected" : "nav-link"}
            onClick={() => navigate("analytics")}
          >
            <BarChart3 size={18} /> Analytics
          </button>
          <button
            className={activePage === "requests" ? "nav-link selected" : "nav-link"}
            onClick={() => navigate("requests")}
          >
            <Clock3 size={18} /> Request logs
            {history.length > 0 && (
              <span className="nav-count">{history.length}</span>
            )}
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-engine-card">
          <div className="mini-card-title">
            <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
            API CONNECTION
          </div>
          <strong>{backendOnline ? "Connected" : "Disconnected"}</strong>
          <span className="sidebar-url">127.0.0.1:8000</span>
          <button className="sidebar-refresh" onClick={refreshData}>
            <RefreshCw size={14} /> Check connection
          </button>
        </div>

        <div className="sidebar-footer">
          <span className="avatar">AI</span>
          <span className="profile-copy">
            <strong>Local workspace</strong>
            <small>Developer mode</small>
          </span>
          <Settings2 size={17} className="muted-icon" />
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-slash">/</span>
            <strong>LLM Serving</strong>
          </div>
          <div className="topbar-right">
            <div className={`connection-pill ${backendOnline ? "connected" : "disconnected"}`}>
              <span className={`status-dot ${backendOnline ? "online" : "offline"}`} />
              {backendOnline ? "API connected" : "API offline"}
            </div>
            <button
              className="icon-action"
              title="Refresh metrics"
              onClick={refreshData}
            >
              <RefreshCw size={17} />
            </button>
            <span className="top-avatar">R</span>
          </div>
        </header>

        <div className="page-content">
          <section id="overview" className="welcome-row">
            <div>
              <div className="live-overline">
                <span className="live-pulse" />
                INFERENCE CONTROL CENTER
              </div>
              <h1>LLM Performance <span>Overview</span></h1>
              <p className="welcome-description">
                Monitor serving performance, inspect requests, and compare your inference engines.
              </p>
            </div>
            <button className="button-secondary" onClick={refreshData}>
              <RefreshCw size={16} /> Refresh data
            </button>
          </section>

          {!backendOnline && (
            <div className="notice notice-error">
              <AlertCircle size={19} />
              <div>
                <strong>Backend unavailable</strong>
                <p>Start your FastAPI server on port 8000 to run inference and load metrics.</p>
              </div>
            </div>
          )}

          {notice && (
            <div className="notice notice-success">
              <CheckCircle2 size={18} />
              <span>{notice}</span>
              <button onClick={() => setNotice("")} aria-label="Dismiss notice">×</button>
            </div>
          )}

          <section className="stats-grid">
            <StatCard
              label="Total API requests"
              value={serverTotalRequests === null ? "—" : serverTotalRequests}
              detail="Reported by backend metrics"
              icon={Activity}
              accent="violet"
            />
            <StatCard
              label="Average latency"
              value={averageLatency === null ? "—" : `${fmt(averageLatency)} s`}
              detail="Backend-reported average"
              icon={Clock3}
              accent="blue"
            />
            <StatCard
              label="P95 latency"
              value={p95 === null ? "—" : `${fmt(p95)} s`}
              detail="95th percentile"
              icon={Gauge}
              accent="teal"
            />
            <StatCard
              label="Output throughput"
              value={throughput === null ? "—" : `${fmt(throughput)} tok/s`}
              detail="Backend-reported metric"
              icon={Zap}
              accent="amber"
            />
          </section>

          <section className="overview-grid">
            <article className="panel system-panel">
              <SectionHeading
                eyebrow="HOST TELEMETRY"
                title="System resources"
                icon={Cpu}
                action={
                  <span className="refresh-label">
                    <span className="live-pulse" /> 4s refresh
                  </span>
                }
              />

              <div className="resource-block">
                <div className="resource-line">
                  <span className="resource-name"><Cpu size={16} /> CPU usage</span>
                  <strong>{percent(cpu)}</strong>
                </div>
                <div className="resource-track">
                  <div
                    className="resource-fill fill-violet"
                    style={{
                      width: `${Math.min(100, Math.max(0, Number(cpu) || 0))}%`,
                    }}
                  />
                </div>
                <div className="resource-subline">
                  <span>Logical cores</span>
                  <span>
                    {firstDefined(
                      systemMetrics?.cpu_count,
                      systemMetrics?.cpu_logical_count,
                      "—"
                    )}
                  </span>
                </div>
              </div>

              <div className="resource-block">
                <div className="resource-line">
                  <span className="resource-name"><Layers3 size={16} /> Memory usage</span>
                  <strong>{percent(ram)}</strong>
                </div>
                <div className="resource-track">
                  <div
                    className="resource-fill fill-teal"
                    style={{
                      width: `${Math.min(100, Math.max(0, Number(ram) || 0))}%`,
                    }}
                  />
                </div>
                <div className="resource-subline">
                  <span>Used / total RAM</span>
                  <span>
                    {ramUsed == null || ramTotal == null
                      ? "—"
                      : `${fmt(ramUsed)} / ${fmt(ramTotal)} GB`}
                  </span>
                </div>
              </div>

              <div className="gpu-card">
                <div className="gpu-symbol"><Zap size={20} /></div>
                <div className="gpu-copy">
                  <strong>{gpuName}</strong>
                  <span>
                    {gpuAvailable
                      ? "GPU available to PyTorch"
                      : "No GPU reported by backend"}
                  </span>
                </div>
                <div className="gpu-memory">
                  <strong>
                    {gpuAllocated == null ? "—" : `${fmt(gpuAllocated)} GB`}
                  </strong>
                  <span>Allocated memory</span>
                </div>
              </div>

              <div className="gpu-footnote">
                <CircleHelp size={14} />
                GPU utilization:{" "}
                {gpuUtilization == null
                  ? "not measured by the current backend"
                  : `${fmt(gpuUtilization, 1)}%`}
                {gpuTotal != null && ` · ${fmt(gpuTotal)} GB total memory`}
              </div>
              {metricsError && <p className="subtle-warning">{metricsError}</p>}
            </article>

            <article id="playground" className="panel playground-panel">
              <SectionHeading
                eyebrow="INTERACTIVE INFERENCE"
                title="Model playground"
                icon={Sparkles}
                action={<span className="panel-tag">LIVE API</span>}
              />

              <form onSubmit={runInference}>
                <label className="input-label" htmlFor="engine-select">
                  Inference engine
                </label>
                <div className="engine-select-wrap">
                  <select
                    id="engine-select"
                    value={engine}
                    onChange={(event) => setEngine(event.target.value)}
                    disabled={loading}
                    className="field-control"
                  >
                    <option value="ollama">Ollama</option>
                    <option value="transformers">Transformers</option>
                  </select>
                  <ChevronDown size={16} />
                </div>

                <div className="model-detail">
                  <span className={`engine-dot ${engine}`} />
                  <span>{ENGINES[engine].model}</span>
                  <span className="model-local">Local engine</span>
                </div>

                <label className="input-label prompt-label" htmlFor="prompt-input">
                  Prompt
                </label>
                <textarea
                  id="prompt-input"
                  className="field-control prompt-area"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder="Ask your model something..."
                  rows={5}
                  disabled={loading}
                />

                <div className="playground-bottom">
                  <label className="token-label" htmlFor="token-limit">
                    Max new tokens
                  </label>
                  <input
                    id="token-limit"
                    className="token-input"
                    type="number"
                    min="1"
                    max={MAX_TOKENS}
                    value={maxTokens}
                    onChange={(event) =>
                      setMaxTokens(
                        Math.max(
                          1,
                          Math.min(
                            MAX_TOKENS,
                            Number(event.target.value) || 1
                          )
                        )
                      )
                    }
                    disabled={loading}
                  />
                  <button
                    type="submit"
                    className="button-primary"
                    disabled={loading || !backendOnline}
                  >
                    {loading ? (
                      <RefreshCw size={16} className="spin" />
                    ) : (
                      <Play size={16} fill="currentColor" />
                    )}
                    {loading ? "Generating..." : "Run inference"}
                  </button>
                </div>
                <div className="benchmark-controls">
                  <label className="token-label" htmlFor="benchmark-runs">
                    Benchmark runs
                  </label>
                  <select
                    id="benchmark-runs"
                    className="field-control benchmark-select"
                    value={iterations}
                    onChange={(event) => setIterations(Number(event.target.value))}
                    disabled={loading}
                  >
                    <option value={1}>1 run</option>
                    <option value={5}>5 runs</option>
                    <option value={10}>10 runs</option>
                  </select>
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={runBenchmark}
                    disabled={loading || !backendOnline}
                  >
                    {benchmarkRunning ? "Benchmarking..." : "Run benchmark"}
                  </button>
                </div>

                <p className="token-help">
                  Generate up to {MAX_TOKENS} new tokens per request. Actual output
                  length depends on the model and backend limits.
                </p>
              </form>

              {error && (
                <div className="notice notice-error playground-error">
                  <AlertCircle size={17} />
                  <span>{error}</span>
                  <button onClick={() => setError("")} aria-label="Dismiss error">×</button>
                </div>
              )}

              <div className="response-area">
                <div className="response-header">
                  <span><Sparkles size={15} /> MODEL RESPONSE</span>
                  {output && (
                    <span className="response-ready">
                      <Check size={13} /> Completed
                    </span>
                  )}
                </div>
                {output ? (
                  <pre className="response-text">{output}</pre>
                ) : (
                  <div className="response-placeholder">
                    <div className="placeholder-icon"><ArrowUpRight size={20} /></div>
                    <p>Your model response will appear here.</p>
                    <span>Choose an engine, enter a prompt, and run inference.</span>
                  </div>
                )}
              </div>
            </article>
          </section>

          <section id="analytics" className="panel analytics-panel">
            <SectionHeading
              eyebrow="OBSERVABILITY"
              title="Request performance"
              icon={BarChart3}
              action={
                <div className="segmented-control">
                  <button
                    className={chartMetric === "latency" ? "active" : ""}
                    onClick={() => setChartMetric("latency")}
                  >
                    Latency
                  </button>
                  <button
                    className={chartMetric === "ttft" ? "active" : ""}
                    onClick={() => setChartMetric("ttft")}
                  >
                    TTFT
                  </button>
                  <button
                    className={chartMetric === "tokensPerSecond" ? "active" : ""}
                    onClick={() => setChartMetric("tokensPerSecond")}
                  >
                    Tokens/sec
                  </button>
                </div>
              }
            />

            <div className="chart-summary-row">
              <div>
                <span className="summary-label">
                  {chartMetric === "latency"
                    ? "P50 latency"
                    : chartMetric === "ttft"
                    ? "Average TTFT"
                    : "Output throughput"}
                </span>
                <strong className="summary-value">
                  {chartMetric === "latency"
                    ? p50 === null ? "—" : `${fmt(p50)} s`
                    : chartMetric === "ttft"
                    ? averageTTFT === null ? "—" : `${fmt(averageTTFT)} s`
                    : throughput === null ? "—" : `${fmt(throughput)} tok/s`}
                </strong>
              </div>
              <div className="chart-legend-note">
                <span className="legend-dot violet-dot" />
                Successful requests from this session
              </div>
            </div>

            <div className="main-chart">
              {chartData.length === 0 ||
              !chartData.some((point) => point[chartMetric] != null) ? (
                <EmptyChart message="Run successful requests to populate this chart with real measurements." />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={chartData}
                    margin={{ top: 10, right: 14, left: -12, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.32} />
                        <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#252b3b" strokeDasharray="4 5" vertical={false} />
                    <XAxis dataKey="request" stroke="#747c91" tickLine={false} axisLine={false} />
                    <YAxis stroke="#747c91" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: "#151827",
                        border: "1px solid #30364b",
                        borderRadius: 12,
                        color: "#f4f3ff",
                      }}
                      labelStyle={{ color: "#b8b2d7" }}
                      formatter={(value, name) => [fmt(value), name]}
                    />
                    <Area
                      type="monotone"
                      dataKey={chartMetric}
                      name={
                        chartMetric === "latency"
                          ? "Latency (s)"
                          : chartMetric === "ttft"
                          ? "TTFT (s)"
                          : "Tokens/sec"
                      }
                      stroke="#a78bfa"
                      strokeWidth={2.5}
                      fill="url(#latencyGradient)"
                      connectNulls={false}
                      dot={{ r: 3, fill: "#a78bfa", strokeWidth: 0 }}
                      activeDot={{ r: 5 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="percentile-grid">
              <div><span>P50 latency</span><strong>{p50 === null ? "—" : `${fmt(p50)} s`}</strong></div>
              <div><span>P95 latency</span><strong>{p95 === null ? "—" : `${fmt(p95)} s`}</strong></div>
              <div><span>P99 latency</span><strong>{p99 === null ? "—" : `${fmt(p99)} s`}</strong></div>
              <div><span>Average TTFT</span><strong>{averageTTFT === null ? "—" : `${fmt(averageTTFT)} s`}</strong></div>
            </div>
          </section>

          <section id="requests" className="panel requests-panel">
            <SectionHeading
              eyebrow="ACTIVITY"
              title="Recent requests"
              icon={Clock3}
              action={
                <div className="request-actions">
                  <button
                    className="button-secondary compact"
                    onClick={exportCSV}
                    disabled={!history.length}
                  >
                    <ArrowDownToLine size={15} /> Export CSV
                  </button>
                  <button
                    className="icon-action danger-action"
                    title="Clear session history"
                    onClick={clearHistory}
                    disabled={!history.length}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              }
            />

            <div className="request-toolbar">
              <div className="request-totals">
                <span><span className="status-dot online" /> {successful} successful this session</span>
                <span><span className="status-dot offline" /> {failed} failed this session</span>
              </div>
              <select
                className="filter-select"
                value={historyFilter}
                onChange={(event) => setHistoryFilter(event.target.value)}
                aria-label="Filter requests by engine"
              >
                <option value="all">All engines</option>
                <option value="ollama">Ollama</option>
                <option value="transformers">Transformers</option>
              </select>
            </div>

            {filteredHistory.length === 0 ? (
              <div className="table-empty">
                <div className="placeholder-icon"><Clock3 size={21} /></div>
                <strong>No request records yet</strong>
                <p>Requests you run in the playground will appear here.</p>
              </div>
            ) : (
              <div className="table-scroll">
                <table className="requests-table">
                  <thead>
                    <tr>
                      <th>REQUEST</th>
                      <th>ENGINE</th>
                      <th>LATENCY</th>
                      <th>OUTPUT TOKENS/S</th>
                      <th>TIME</th>
                      <th>STATUS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHistory.map((item) => (
                      <tr key={item.id}>
                        <td>
                          <div className="request-prompt" title={item.prompt}>
                            {item.prompt}
                          </div>
                          <div className="request-id">ID {item.id.slice(0, 12)}</div>
                        </td>
                        <td>
                          <span className={`engine-pill ${item.engine}`}>
                            <span className={`engine-dot ${item.engine}`} />
                            {ENGINES[item.engine]?.name || item.engine}
                          </span>
                        </td>
                        <td>{fmt(item.latency)} s</td>
                        <td>
                          {item.tokensPerSecond == null
                            ? "—"
                            : fmt(item.tokensPerSecond)}
                        </td>
                        <td>{new Date(item.timestamp).toLocaleTimeString()}</td>
                        <td>
                          <span className={`status-label ${item.success ? "ok" : "bad"}`}>
                            {item.success
                              ? <CheckCircle2 size={14} />
                              : <AlertCircle size={14} />}
                            {item.success ? "Success" : "Failed"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="table-footer">
              <span>Showing up to {HISTORY_LIMIT} requests from the current dashboard session.</span>
              <span>{filteredHistory.length} records</span>
            </div>
          </section>

          <footer className="page-footer">
            <div className="footer-brand">
              <ShieldCheck size={16} /> INFERBENCH <span>·</span> Local inference observability
            </div>
            <div>
              Metrics are shown only when reported by your backend or measured in this session.
            </div>
          </footer>
        </div>
      </main>
    </div>
  );
}
