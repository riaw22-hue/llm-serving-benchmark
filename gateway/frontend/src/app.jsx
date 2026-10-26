
import { useEffect, useState } from "react";
import {
  Activity, Cpu, Gauge, Send, RefreshCw,
  Download, Zap, Server, Clock, Trash2
} from "lucide-react";
import {
  ResponsiveContainer, LineChart, Line,
  CartesianGrid, XAxis, YAxis, Tooltip
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [backend, setBackend] = useState("Checking...");
  const [prompt, setPrompt] = useState(
    "Explain how a Large Language Model generates text."
  );
  const [maxTokens, setMaxTokens] = useState(100);
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  async function checkBackend() {
    try {
      const response = await fetch(`${API}/health`);
      if (!response.ok) throw new Error("Backend unavailable");
      setBackend("Online");
    } catch {
      setBackend("Offline");
    }
  }

  useEffect(() => {
    checkBackend();
  }, []);

  async function generate() {
    if (!prompt.trim()) {
      setError("Please enter a prompt.");
      return;
    }

    setLoading(true);
    setError("");
    setOutput("");
    const start = performance.now();

    try {
      const response = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: Number(maxTokens)
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Generation failed");
      }

      const text =
        data.generated_text ??
        data.response ??
        data.output ??
        data.text ??
        JSON.stringify(data, null, 2);

      setOutput(
        typeof text === "string" ? text : JSON.stringify(text, null, 2)
      );

      const latency = Number(
        ((performance.now() - start) / 1000).toFixed(2)
      );

      setHistory(previous => [
        {
          id: Date.now(),
          prompt,
          latency,
          time: new Date().toLocaleTimeString(),
          status: "Success"
        },
        ...previous
      ].slice(0, 20));

      setBackend("Online");
    } catch (err) {
      setError(
        `${err.message}. Check the backend and its CORS settings.`
      );
      checkBackend();
    } finally {
      setLoading(false);
    }
  }

  function exportCSV() {
    const rows = [
      ["Time", "Prompt", "Latency (seconds)", "Status"],
      ...history.map(item => [
        item.time, item.prompt, item.latency, item.status
      ])
    ];

    const csv = rows
      .map(row =>
        row.map(value =>
          `"${String(value).replace(/"/g, '""')}"`
        ).join(",")
      ).join("\n");

    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8;" })
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "llm-benchmark-history.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  const average = history.length
    ? (
        history.reduce((sum, item) => sum + item.latency, 0) /
        history.length
      ).toFixed(2)
    : "—";

  const chartData = [...history].reverse().map((item, index) => ({
    request: index + 1,
    latency: item.latency
  }));

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Activity size={24} /></div>
          <div>
            <h2>InferBench</h2>
            <p>LLM PERFORMANCE LAB</p>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <div className="nav-item active"><Gauge size={18} /> Overview</div>
        <div className="nav-item"><Zap size={18} /> Playground</div>
        <div className="nav-item"><Activity size={18} /> Benchmarks</div>

        <div className="sidebar-bottom">
          <div className="connection-card">
            <span className={`status-dot ${backend === "Online" ? "online" : ""}`} />
            <div>
              <strong>Inference API</strong>
              <p>{backend}</p>
            </div>
            <button className="icon-button" onClick={checkBackend} title="Refresh">
              <RefreshCw size={15} />
            </button>
          </div>
          <p className="sidebar-footer">LOCAL DEVELOPMENT · V1.0</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">MODEL OBSERVABILITY / OVERVIEW</p>
            <h1>LLM Serving Benchmark</h1>
            <p className="subtitle">
              Measure inference performance. Understand every request.
            </p>
          </div>
          <span className={`status-pill ${backend === "Online" ? "pill-online" : ""}`}>
            <span className={`status-dot ${backend === "Online" ? "online" : ""}`} />
            {backend}
          </span>
        </header>

        <section className="metric-grid">
          <div className="metric-card">
            <div className="metric-top">Backend Status <Server size={18} /></div>
            <h2 className="metric-value">{backend}</h2>
            <p className="metric-note">FastAPI inference server</p>
          </div>
          <div className="metric-card">
            <div className="metric-top">Requests Completed <Activity size={18} /></div>
            <h2 className="metric-value">{history.length}</h2>
            <p className="metric-note">This dashboard session</p>
          </div>
          <div className="metric-card">
            <div className="metric-top">Average Latency <Clock size={18} /></div>
            <h2 className="metric-value">{average}{average !== "—" ? " s" : ""}</h2>
            <p className="metric-note">Measured request duration</p>
          </div>
          <div className="metric-card">
            <div className="metric-top">Inference Engine <Cpu size={18} /></div>
            <h2 className="metric-value engine-value">Transformers</h2>
            <p className="metric-note">Qwen 2.5 · 1.5B Instruct</p>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">INTERACTIVE TESTING</p>
                <h2>Inference Playground</h2>
              </div>
              <span className="panel-icon"><Zap size={19} /></span>
            </div>

            <label htmlFor="prompt">Your prompt</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              rows={5}
              placeholder="Enter a prompt..."
            />

            <div className="controls-row">
              <div className="token-control">
                <label htmlFor="tokens">Max new tokens</label>
                <select
                  id="tokens"
                  value={maxTokens}
                  onChange={event => setMaxTokens(event.target.value)}
                >
                  <option value={50}>50 tokens</option>
                  <option value={100}>100 tokens</option>
                  <option value={150}>150 tokens</option>
                  <option value={200}>200 tokens</option>
                </select>
              </div>
              <button className="primary-button" onClick={generate} disabled={loading}>
                {loading
                  ? <RefreshCw size={17} className="spin" />
                  : <Send size={17} />}
                {loading ? "Generating..." : "Run inference"}
              </button>
            </div>

            {error && <div className="error-box">{error}</div>}

            <div className="output-heading">
              <span>MODEL RESPONSE</span>
              {output && <span className="success-label">● Complete</span>}
            </div>
            <div className="output-box">
              {loading
                ? <p className="muted">Model is generating a response...</p>
                : output
                  ? <pre>{output}</pre>
                  : <p className="muted">Your AI response will appear here.</p>}
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">PERFORMANCE MONITORING</p>
                <h2>Latency Overview</h2>
              </div>
              <span className="panel-icon"><Gauge size={19} /></span>
            </div>
            <p className="chart-description">Request duration in seconds</p>
            <div className="chart-container">
              {chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#263244" strokeDasharray="4 4" />
                    <XAxis dataKey="request" stroke="#7f8da3" />
                    <YAxis stroke="#7f8da3" />
                    <Tooltip
                      contentStyle={{
                        background: "#111a29",
                        border: "1px solid #2b394d",
                        borderRadius: 8
                      }}
                      formatter={value => [`${value}s`, "Latency"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="latency"
                      stroke="#65e6bd"
                      strokeWidth={2.5}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-chart">
                  <Activity size={30} />
                  <p>No benchmark data yet</p>
                  <span>Run inference to populate this chart.</span>
                </div>
              )}
            </div>
            <div className="chart-footer">
              <span><span className="legend-dot" /> Request latency</span>
              <span>{history.length} data points</span>
            </div>
          </div>
        </section>

        <section className="panel history-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">REQUEST LOGS</p>
              <h2>Recent Requests</h2>
            </div>
            <div className="history-actions">
              <button className="secondary-button" onClick={exportCSV} disabled={!history.length}>
                <Download size={16} /> Export CSV
              </button>
              <button
                className="icon-button"
                onClick={() => setHistory([])}
                disabled={!history.length}
                title="Clear history"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>

          {history.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>TIME</th><th>PROMPT</th><th>LATENCY</th><th>STATUS</th></tr>
                </thead>
                <tbody>
                  {history.map(item => (
                    <tr key={item.id}>
                      <td>{item.time}</td>
                      <td className="prompt-cell">{item.prompt}</td>
                      <td>{item.latency}s</td>
                      <td className="table-status">● {item.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-history">
              No requests recorded yet. Run a prompt above to get started.
            </div>
          )}
        </section>

        <footer className="page-footer">
          <span>INFERBENCH / LOCAL MODEL EVALUATION</span>
          <span>Built with FastAPI · React · Transformers</span>
        </footer>
      </main>
    </div>
  );
}
