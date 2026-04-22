import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from backend");
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error("Error:", err);
      setError("Could not connect to the backend. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="container">
        <header className="hero">
          <p className="eyebrow">Week 3 Project</p>
          <h1>Decision Intelligence Assistant</h1>
          <p className="hero-text">
            Compare RAG, non-RAG, ML priority prediction, and LLM zero-shot prediction
            in one place.
          </p>
        </header>

        <section className="query-panel card">
          <h2>Ask a Support Query</h2>
          <div className="query-controls">
            <textarea
              className="query-input"
              placeholder="Example: HELP!!! I have been trying to get a refund for three days and nobody is responding"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
              {loading ? "Analyzing..." : "Submit Query"}
            </button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </section>

        {data && (
          <>
            <section className="grid two-col">
              <div className="card">
                <h2>RAG Answer</h2>
                <p>{data.rag_answer}</p>
              </div>

              <div className="card">
                <h2>Non-RAG Answer</h2>
                <p>{data.non_rag_answer}</p>
              </div>
            </section>

            <section className="grid two-col">
              <div className="card">
                <h2>ML Prediction</h2>
                <div className="metric-row">
                  <span className="metric-label">Label</span>
                  <span className="badge">{data.ml_prediction.label}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Confidence</span>
                  <span>{data.ml_prediction.confidence}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Latency</span>
                  <span>{data.ml_prediction.latency_ms} ms</span>
                </div>
              </div>

              <div className="card">
                <h2>LLM Zero-Shot Prediction</h2>
                <div className="metric-row">
                  <span className="metric-label">Label</span>
                  <span className="badge">{data.llm_zero_shot_prediction.label}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Confidence</span>
                  <span>{data.llm_zero_shot_prediction.confidence}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Latency</span>
                  <span>{data.llm_zero_shot_prediction.latency_ms} ms</span>
                </div>
              </div>
            </section>

            <section className="grid three-col">
              <div className="card">
                <h2>Latency</h2>
                <div className="metric-row">
                  <span className="metric-label">Retrieval</span>
                  <span>{data.retrieval_latency_ms} ms</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">RAG</span>
                  <span>{data.rag_latency_ms} ms</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Non-RAG</span>
                  <span>{data.non_rag_latency_ms} ms</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Total</span>
                  <span>{data.total_latency_ms} ms</span>
                </div>
              </div>

              <div className="card">
                <h2>Cost</h2>
                <div className="metric-row">
                  <span className="metric-label">RAG</span>
                  <span>${data.cost.rag_cost_usd}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Non-RAG</span>
                  <span>${data.cost.non_rag_cost_usd}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Zero-Shot</span>
                  <span>${data.cost.zero_shot_cost_usd}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Total</span>
                  <span>${data.cost.total_llm_cost_usd}</span>
                </div>
              </div>

              <div className="card">
                <h2>Query Summary</h2>
                <p className="summary-box">{data.query}</p>
              </div>
            </section>

            <section className="card">
              <h2>Retrieved Sources</h2>
              <div className="sources-list">
                {data.sources.map((source, index) => (
                  <div className="source-item" key={index}>
                    <div className="source-top">
                      <span className="source-id">Tweet ID: {source.tweet_id}</span>
                      <span className="source-score">Score: {source.score.toFixed(4)}</span>
                    </div>
                    <p>{source.text}</p>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}

export default App;