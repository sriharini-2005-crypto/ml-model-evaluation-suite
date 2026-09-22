import React, { useState, useEffect } from 'react';

const API_BASE = 'http://127.0.0.1:8000/api';

// Reusable SVG Linear Progress Bar Component
function ProgressBar({ value, color = '#06b6d4', height = 8 }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  return (
    <div className="progress-bar-bg" style={{ height: `${height}px` }}>
      <div
        className="progress-bar-fill"
        style={{
          width: `${pct}%`,
          background: color,
        }}
      />
    </div>
  );
}

// Reusable Circular Metric Ring Component
function CircularMetric({ score, label, color = '#06b6d4', size = 90 }) {
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.min(Math.max(score, 0), 1);
  const strokeDashoffset = circumference - pct * circumference;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
      <div style={{ position: 'relative', width: `${size}px`, height: `${size}px` }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
        </svg>
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700,
            fontSize: '0.95rem',
            color: '#f8fafc',
          }}
        >
          {(pct * 100).toFixed(1)}%
        </div>
      </div>
      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</span>
    </div>
  );
}

export default function App() {
  // Navigation tabs in user requested order: Dataset Upload -> Visual Benchmarking -> Model Selection -> Patient Inference -> Model Registry -> Overview
  const [activeTab, setActiveTab] = useState('upload');
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState([]);
  const [selectedCriterion, setSelectedCriterion] = useState('Accuracy');
  
  // Patient prediction states
  const [patientFeatures, setPatientFeatures] = useState({});
  const [featureRanges, setFeatureRanges] = useState({});
  const [selectedModel, setSelectedModel] = useState('Random Forest');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [modelRegistry, setModelRegistry] = useState([]);

  useEffect(() => {
    fetchDefaultDataset();
    fetchBenchmark();
    fetchRegistry();
  }, []);

  const fetchDefaultDataset = async () => {
    try {
      const res = await fetch(`${API_BASE}/dataset/default`);
      if (res.ok) {
        const data = await res.json();
        setDatasetInfo(data);
        if (data.sample_defaults) {
          setPatientFeatures(data.sample_defaults);
          
          const ranges = {};
          Object.keys(data.sample_defaults).forEach((key) => {
            const val = data.sample_defaults[key];
            const min = val > 0 ? val * 0.2 : 0;
            const max = val > 0 ? val * 2.5 : 10;
            ranges[key] = { min: parseFloat(min.toFixed(2)), max: parseFloat(max.toFixed(2)) };
          });
          setFeatureRanges(ranges);
        }
      }
    } catch (err) {
      console.error('Failed to fetch dataset info:', err);
    }
  };

  const fetchBenchmark = async () => {
    try {
      const res = await fetch(`${API_BASE}/benchmark`);
      if (res.ok) {
        const data = await res.json();
        setBenchmarkData(data.metrics || []);
      }
    } catch (err) {
      console.error('Failed to fetch benchmark:', err);
    }
  };

  const fetchRegistry = async () => {
    try {
      const res = await fetch(`${API_BASE}/registry`);
      if (res.ok) {
        const data = await res.json();
        setModelRegistry(data || []);
      }
    } catch (err) {
      console.error('Failed to fetch registry:', err);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('Uploading and parsing dataset CSV...');
    try {
      const res = await fetch(`${API_BASE}/upload-dataset`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`✅ ${data.status} (${data.samples} samples, ${data.features_count} features)`);
        fetchDefaultDataset();
      } else {
        setUploadStatus(`❌ Error: ${data.detail}`);
      }
    } catch (err) {
      setUploadStatus(`❌ Upload failed: ${err.message}`);
    }
  };

  const runTrainingPipeline = async () => {
    setIsTraining(true);
    try {
      const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setBenchmarkData(data.metrics || []);
        fetchRegistry();
        alert('🎉 Training and benchmarking complete!');
      } else {
        alert(`Error: ${data.detail}`);
      }
    } catch (err) {
      alert(`Training failed: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handlePatientPrediction = async () => {
    setIsPredicting(true);
    setPredictionResult(null);

    const featureValues = Object.values(patientFeatures).map((v) => parseFloat(v) || 0.0);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: featureValues,
          selected_model: selectedModel,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setPredictionResult(data);
      } else {
        alert(`Prediction failed: ${data.detail}`);
      }
    } catch (err) {
      alert(`Prediction error: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const resetSlidersToMean = () => {
    if (datasetInfo && datasetInfo.sample_defaults) {
      setPatientFeatures(datasetInfo.sample_defaults);
    }
  };

  const getTopModelForCriterion = () => {
    if (!benchmarkData || benchmarkData.length === 0) return null;
    const sorted = [...benchmarkData].sort((a, b) => {
      if (selectedCriterion === 'Training Time (s)') {
        return (a['Training Time (s)'] || 0) - (b['Training Time (s)'] || 0);
      }
      return (b[selectedCriterion] || 0) - (a[selectedCriterion] || 0);
    });
    return sorted[0];
  };

  const topModel = getTopModelForCriterion();

  return (
    <div style={{ maxWidth: '1320px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Header Bar */}
      <header style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div className="badge badge-purple" style={{ marginBottom: '0.5rem' }}>SIH Prototype — Disease Risk Classification</div>
            <h1 style={{ fontSize: '2.4rem' }}>
              Hybrid Quantum-Classical <span className="gradient-text">Disease Platform</span>
            </h1>
          </div>
          <button className="btn-primary" onClick={runTrainingPipeline} disabled={isTraining}>
            {isTraining ? '🔄 Training Pipeline...' : '⚡ Run Hybrid Benchmark'}
          </button>
        </div>

        {/* Medical Research Disclaimer */}
        <div className="disclaimer-banner">
          <span style={{ fontSize: '1.3rem' }}>⚠️</span>
          <div>
            <strong>Medical Decision-Support Prototype:</strong> This platform compares classical and quantum ML algorithms on biomedical datasets. It is <strong>NOT</strong> a clinical diagnostic system.
          </div>
        </div>
      </header>

      {/* Navigation Tab Bar (Re-ordered according to user preference) */}
      <nav style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
        <button className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
          📁 Dataset Upload
        </button>
        <button className={`nav-tab ${activeTab === 'benchmark' ? 'active' : ''}`} onClick={() => setActiveTab('benchmark')}>
          📊 Visual Benchmarking
        </button>
        <button className={`nav-tab ${activeTab === 'selection' ? 'active' : ''}`} onClick={() => setActiveTab('selection')}>
          🎯 Model Selection
        </button>
        <button className={`nav-tab ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>
          🩺 Patient Inference
        </button>
        <button className={`nav-tab ${activeTab === 'registry' ? 'active' : ''}`} onClick={() => setActiveTab('registry')}>
          💡 Model Registry
        </button>
        <button className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
          📌 Platform Overview
        </button>
      </nav>

      {/* TAB 1: DATASET UPLOAD (FIRST) */}
      {activeTab === 'upload' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <h3>📁 Doctor & User Dataset Upload</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Upload custom CSV datasets to train and compare classical and quantum models.
            </p>
            <div className="dropzone" onClick={() => document.getElementById('file-upload-input').click()}>
              <input id="file-upload-input" type="file" accept=".csv" onChange={handleFileUpload} style={{ display: 'none' }} />
              <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>📤</div>
              <p style={{ fontWeight: 600 }}>Click or Drag CSV Dataset File Here</p>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Supported format: .csv with numerical features and target column</span>
            </div>
            {uploadStatus && (
              <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', fontSize: '0.9rem' }}>
                {uploadStatus}
              </div>
            )}
          </div>

          {datasetInfo && (
            <div className="glass-panel">
              <h3>📊 Current Dataset Inspection ({datasetInfo.dataset_name})</h3>
              <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <div><strong>Total Samples:</strong> {datasetInfo.samples}</div>
                <div><strong>Features:</strong> {datasetInfo.features_count}</div>
                <div><strong>Malignant (1):</strong> {datasetInfo.class_distribution?.['Malignant (1)']}</div>
                <div><strong>Benign (0):</strong> {datasetInfo.class_distribution?.['Benign (0)']}</div>
              </div>

              <h4>Feature Preview (First 5 records)</h4>
              <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
                <table className="custom-table">
                  <thead>
                    <tr>
                      {datasetInfo.feature_names && datasetInfo.feature_names.slice(0, 8).map((f) => (
                        <th key={f}>{f}</th>
                      ))}
                      <th>target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetInfo.preview && datasetInfo.preview.slice(0, 5).map((row, i) => (
                      <tr key={i}>
                        {datasetInfo.feature_names.slice(0, 8).map((f) => (
                          <td key={f}>{row[f] ? row[f].toFixed(3) : '-'}</td>
                        ))}
                        <td>
                          <span className={`badge ${row.target === 1 ? 'badge-rose' : 'badge-emerald'}`}>
                            {row.target === 1 ? 'Malignant' : 'Benign'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: VISUAL BENCHMARKING */}
      {activeTab === 'benchmark' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3>📊 Visual Benchmark Performance</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
                  Comparing models evaluated on identical 4-PCA components.
                </p>
              </div>
              <button className="btn-primary" onClick={runTrainingPipeline} disabled={isTraining}>
                {isTraining ? 'Training...' : 'Re-Run Benchmark'}
              </button>
            </div>

            {benchmarkData.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="grid-3">
                  {benchmarkData.map((modelRow, i) => (
                    <div key={i} className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.5)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                        <h4>{modelRow.Model}</h4>
                        <span className="badge badge-purple">{modelRow.Qubits ? `${modelRow.Qubits} Qubits` : 'Classical'}</span>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-around', margin: '1rem 0' }}>
                        <CircularMetric score={modelRow.Accuracy} label="Accuracy" color="#06b6d4" size={75} />
                        <CircularMetric score={modelRow.Sensitivity} label="Sensitivity" color="#10b981" size={75} />
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                            <span>Specificity</span>
                            <span>{(modelRow.Specificity * 100).toFixed(1)}%</span>
                          </div>
                          <ProgressBar value={modelRow.Specificity} color="#a855f7" height={6} />
                        </div>
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                            <span>F1-Score</span>
                            <span>{(modelRow.F1 * 100).toFixed(1)}%</span>
                          </div>
                          <ProgressBar value={modelRow.F1} color="#f43f5e" height={6} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                  <table className="custom-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Sensitivity</th>
                        <th>Specificity</th>
                        <th>Precision</th>
                        <th>F1-Score</th>
                        <th>ROC-AUC</th>
                        <th>Training Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkData.map((row, idx) => (
                        <tr key={idx}>
                          <td><strong>{row.Model}</strong></td>
                          <td>{(row.Accuracy * 100).toFixed(2)}%</td>
                          <td>{(row.Sensitivity * 100).toFixed(2)}%</td>
                          <td>{(row.Specificity * 100).toFixed(2)}%</td>
                          <td>{(row.Precision * 100).toFixed(2)}%</td>
                          <td>{(row.F1 * 100).toFixed(2)}%</td>
                          <td>{row['ROC-AUC'] ? (row['ROC-AUC']).toFixed(4) : 'N/A'}</td>
                          <td>{row['Training Time (s)'] ? `${row['Training Time (s)'].toFixed(3)}s` : '0s'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>No benchmark metrics. Click 'Run Hybrid Benchmark' to train.</p>
            )}
          </div>

          <div className="grid-2">
            <div className="glass-panel">
              <h3>📉 Combined ROC Curves</h3>
              <img src="http://127.0.0.1:8000/results/roc_curves.png" alt="ROC Curves" style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }} onError={(e) => (e.target.style.display = 'none')} />
            </div>
            <div className="glass-panel">
              <h3>🎯 Confusion Matrix (Random Forest)</h3>
              <img src="http://127.0.0.1:8000/results/confusion_random_forest.png" alt="Random Forest CM" style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }} onError={(e) => (e.target.style.display = 'none')} />
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: MODEL SELECTION */}
      {activeTab === 'selection' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <h3>🎯 Criteria-Based Model Selection</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1.5rem' }}>
              Select an evaluation metric to dynamically highlight the optimal decision model.
            </p>

            <div style={{ maxWidth: '320px', marginBottom: '2rem' }}>
              <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                Select Evaluation Criterion:
              </label>
              <select className="form-select" value={selectedCriterion} onChange={(e) => setSelectedCriterion(e.target.value)}>
                <option value="Accuracy">Accuracy</option>
                <option value="Sensitivity">Sensitivity (Recall)</option>
                <option value="Specificity">Specificity</option>
                <option value="F1">F1-Score</option>
                <option value="ROC-AUC">ROC-AUC</option>
                <option value="Training Time (s)">Training Speed</option>
              </select>
            </div>

            {topModel ? (
              <div>
                <span className="badge badge-emerald" style={{ marginBottom: '0.75rem' }}>Highest-Performing Model Under Selected Criterion</span>
                <h1 style={{ fontSize: '2.8rem', color: 'var(--accent-cyan)', marginBottom: '1.5rem' }}>
                  🏆 {topModel.Model}
                </h1>

                <div className="grid-4">
                  <div className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.6)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Accuracy</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>
                      {(topModel.Accuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.6)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Sensitivity</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                      {(topModel.Sensitivity * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.6)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Specificity</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                      {(topModel.Specificity * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.6)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>F1-Score</div>
                    <div style={{ fontSize: '1.8rem', fontWeight: '700', color: 'var(--accent-rose)' }}>
                      {(topModel.F1 * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Run training pipeline first to perform model selection.</p>
            )}
          </div>
        </div>
      )}

      {/* TAB 4: INTERACTIVE PATIENT INFERENCE */}
      {activeTab === 'predict' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3>🩺 Interactive Patient Measurement Sliders</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
                  Adjust 30 cell nucleus parameters via interactive range sliders to simulate real-time patient diagnosis.
                </p>
              </div>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={resetSlidersToMean}>
                  🔄 Reset Sliders to Mean
                </button>
                <div style={{ width: '220px' }}>
                  <select className="form-select" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                    <option value="Random Forest">Random Forest (Classical)</option>
                    <option value="SVM">SVM (Classical)</option>
                    <option value="XGBoost">XGBoost (Classical)</option>
                    <option value="QSVM">QSVM (Quantum Kernel)</option>
                    <option value="VQC">VQC (Variational Quantum)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* 30 Interactive Feature Sliders */}
            <div className="grid-3" style={{ maxHeight: '420px', overflowY: 'auto', paddingRight: '0.75rem', gap: '1rem' }}>
              {Object.keys(patientFeatures).map((key) => {
                const val = parseFloat(patientFeatures[key]) || 0.0;
                const range = featureRanges[key] || { min: 0, max: val * 2 || 10 };
                return (
                  <div key={key} className="custom-slider-container">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600 }}>
                      <span style={{ color: 'var(--text-primary)' }}>{key}</span>
                      <span style={{ color: 'var(--accent-cyan)' }}>{val.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min={range.min}
                      max={range.max}
                      step={(range.max - range.min) / 100 || 0.01}
                      className="custom-range-slider"
                      value={val}
                      onChange={(e) => setPatientFeatures({ ...patientFeatures, [key]: parseFloat(e.target.value) })}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      <span>Min: {range.min}</span>
                      <span>Max: {range.max}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
              <button className="btn-primary" onClick={handlePatientPrediction} disabled={isPredicting}>
                {isPredicting ? 'Calculating Inference...' : '🩺 PREDICT PATIENT RISK NOW'}
              </button>
            </div>
          </div>

          {/* PREDICTION RESULT CARD */}
          {predictionResult && (
            <div className="glass-panel" style={{ borderLeft: predictionResult.predicted_label === 'Malignant' ? '6px solid #f43f5e' : '6px solid #10b981' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <span className="badge badge-purple" style={{ marginBottom: '0.5rem' }}>Prediction Outcome</span>
                  <h2>Diagnostic Risk Classification</h2>
                </div>
                <div className="badge badge-cyan" style={{ fontSize: '0.9rem', padding: '0.5rem 1rem' }}>
                  Model: {predictionResult.selected_model}
                </div>
              </div>

              <div className="grid-3" style={{ alignItems: 'center', marginBottom: '1.5rem' }}>
                <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1.25rem', borderRadius: '14px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Predicted Risk Class</div>
                  <span
                    className={`badge ${predictionResult.predicted_label === 'Malignant' ? 'badge-rose' : 'badge-emerald'}`}
                    style={{ fontSize: '1.3rem', padding: '0.6rem 1.5rem' }}
                  >
                    {predictionResult.predicted_label}
                  </span>
                </div>

                <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1.25rem', borderRadius: '14px', display: 'flex', justifyContent: 'center' }}>
                  <CircularMetric
                    score={(predictionResult.confidence_percentage || 0) / 100}
                    label="Prediction Confidence"
                    color={predictionResult.predicted_label === 'Malignant' ? '#f43f5e' : '#10b981'}
                    size={100}
                  />
                </div>

                <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1.25rem', borderRadius: '14px' }}>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>4 PCA Features (4 Qubits)</div>
                  {predictionResult.pca_features && predictionResult.pca_features.map((fVal, idx) => (
                    <div key={idx} style={{ marginBottom: '0.35rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                        <span>PCA {idx + 1}</span>
                        <span>{fVal.toFixed(3)}</span>
                      </div>
                      <ProgressBar value={Math.abs(fVal) / 5} color="#3b82f6" height={5} />
                    </div>
                  ))}
                </div>
              </div>

              <div className="disclaimer-banner">
                <span>🔒</span>
                <div>{predictionResult.disclaimer}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: MODEL REGISTRY */}
      {activeTab === 'registry' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <h3>💡 Disease-Specific Model Registry</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1.5rem' }}>
              Persistent JSON model registry tracking model metadata and quantum circuit configurations.
            </p>

            {modelRegistry.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Disease</th>
                      <th>Model</th>
                      <th>Version</th>
                      <th>PCA Components</th>
                      <th>Qubits</th>
                      <th>Accuracy</th>
                      <th>Training Time</th>
                      <th>Registered At</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelRegistry.map((reg, i) => (
                      <tr key={i}>
                        <td>{reg.disease}</td>
                        <td><strong>{reg.model}</strong></td>
                        <td>{reg.version}</td>
                        <td>{reg.pca_components}</td>
                        <td><span className="badge badge-purple">{reg.qubits}</span></td>
                        <td>{reg.metrics?.Accuracy ? `${(reg.metrics.Accuracy * 100).toFixed(2)}%` : '-'}</td>
                        <td>{reg.training_time_seconds}s</td>
                        <td>{reg.created_at}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Registry is empty. Run training pipeline to populate registry.</p>
            )}
          </div>
        </div>
      )}

      {/* TAB 6: PLATFORM OVERVIEW */}
      {activeTab === 'overview' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="grid-4">
            <div className="glass-panel">
              <span className="badge badge-cyan" style={{ marginBottom: '0.5rem' }}>Target Disease</span>
              <h2>Breast Cancer</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Wisconsin Diagnostic Dataset</p>
            </div>
            <div className="glass-panel">
              <span className="badge badge-emerald" style={{ marginBottom: '0.5rem' }}>Dataset Size</span>
              <h2>{datasetInfo ? `${datasetInfo.samples} Samples` : '569 Samples'}</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>30 Original Features</p>
            </div>
            <div className="glass-panel">
              <span className="badge badge-purple" style={{ marginBottom: '0.5rem' }}>Dimensionality</span>
              <h2>4 PCA Components</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>79.32% Variance Retained</p>
            </div>
            <div className="glass-panel">
              <span className="badge badge-rose" style={{ marginBottom: '0.5rem' }}>Quantum Capacity</span>
              <h2>4 Qubits</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Qiskit ZZFeatureMap</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
