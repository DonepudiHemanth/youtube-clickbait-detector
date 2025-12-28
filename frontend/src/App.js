import React, { useState } from "react";
import "./App.css";

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    if (!url) return alert("Please enter a YouTube URL");

    setLoading(true);
    setResult("");

    try {
      const res = await fetch("https://clickbait-backend-rywk.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });

      const data = await res.json();
      setResult(data.result);
    } catch (err) {
      console.error(err);
      setResult("Server Error");
    }

    setLoading(false);
  };

  // ✅ Background logic
const lower = typeof result === "string" ? result.toLowerCase() : "";
  const bgClass = lower.includes("not")
    ? "bg-good"
    : lower.includes("clickbait")
    ? "bg-bad"
    : "";

  return (
    <div className={`app ${bgClass}`}>
      <header className="navbar">
        <h2>ClickbaitGuard</h2>
      </header>

      <main className="container">
        <div className="left">
          <h1>YouTube Clickbait Detection</h1>
          <p className="desc">
            Analyze YouTube videos using AI to detect misleading clickbait content.
          </p>

          <div className="input-group">
            <input
              type="text"
              placeholder="Paste YouTube video URL..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button onClick={handleDetect} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {loading && (
            <div className="progress">
              <div className="bar"></div>
              <span>Processing video, please wait...</span>
            </div>
          )}

          {result && (
            <div
              className={`result-pill ${
                lower.includes("not") ? "good" : "bad"
              }`}
            >
              {result}
            </div>
          )}
        </div>

        <div className="right">
          <div className="info-card">
            <h3>How it works</h3>
            <ul>
              <li>Fetch video metadata</li>
              <li>Analyze thumbnail & title</li>
              <li>Process transcript</li>
              <li>Evaluate engagement</li>
              <li>ML-based prediction</li>
            </ul>
          </div>
        </div>
      </main>

      <footer className="footer">
        © 2025 ClickbaitGuard — AI-powered Content Analysis
      </footer>
    </div>
  );
}

export default App;
