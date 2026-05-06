import React, { useState } from "react";
import "./App.css";

const BASE_URL = "http://127.0.0.1:5000"; 

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImagePreview(URL.createObjectURL(file));
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      // This calls the specific 'compare' route in your backend
      const res = await fetch(`${BASE_URL}/api/compare`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error connecting to backend:", err);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Deep Learning for Brain Tumor Detection</h1>
        <p>Under Adversarial Threats</p>
      </header>

      <main className="container">
        {/* LEFT CARD: Upload Section */}
        <section className="card">
          <h2>Upload MRI Scan</h2>
          <div className="upload-box">
            <input type="file" onChange={handleUpload} accept="image/*" id="upload" hidden />
            <label htmlFor="upload" className="upload-label">
              Click to browse Brain MRI Scan (JPG, PNG)
            </label>
          </div>
          {imagePreview && <img src={imagePreview} alt="Preview" className="preview-img" />}
          {loading && <p>Processing through models...</p>}
        </section>

        {/* RIGHT CARD: Results Section */}
        <section className="card">
          {!result ? (
            <div className="no-result">
              <h3>No prediction yet</h3>
              <p>Upload an MRI scan to compare models</p>
            </div>
          ) : (
            <div className="result-display">
              <h2>Model Comparison</h2>
              {Object.entries(result.results).map(([model, data]) => (
                <div key={model} className="model-row">
                  <strong>{model}</strong>
                  <p>Prediction: {data.class}</p>
                  <p>Confidence: {(data.confidence * 100).toFixed(2)}%</p>
                </div>
              ))}
              <div className="best-model">
                <h3>Final Verdict: {result.best_model}</h3>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
