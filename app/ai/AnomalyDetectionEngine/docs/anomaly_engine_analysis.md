# ARGUS Anomaly Detection Engine — Deep Analysis & Build Plan

## 1. What We Have Right Now

### Project Skeleton
```
argus-server/
├── app/
│   ├── main.py                  # Bare FastAPI app (4 lines)
│   ├── ai/
│   │   ├── AnamalyDetectionEngine/
│   │   │   ├── anomaly_detector.py   ← EMPTY STUB
│   │   │   └── data/
│   │   │       ├── labeled_anomalies.csv
│   │   │       ├── train/  (82 × .npy files)
│   │   │       ├── test/   (82 × .npy files)
│   │   │       └── 2018-05-19_15.00.10/   ← cached run results
│   │   │           ├── params.log
│   │   │           ├── models/   (82 × .h5 Keras LSTM models)
│   │   │           ├── y_hat/    (82 × .npy prediction arrays)
│   │   │           └── smoothed_errors/  (82 × .npy error arrays)
│   │   └── ThreatClassificationEngine/
│   │       └── threat_classifier.py  ← EMPTY STUB
│   ├── api/deps.py + routes/
│   ├── core/ (config, database, security — empty stubs)
│   ├── models/ (6 Beanie/Pydantic model stubs)
│   └── services/ (alert_engine, blockchain, pipeline — empty stubs)
├── requirements.txt
└── README.md
```

### The Dataset — NASA SMAP & MSL Benchmark
This is the **public NASA SMAP/MSL anomaly benchmark** — the exact dataset used in the paper:  
*"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"* (Hundman et al., KDD 2018).

| Property | Detail |
|---|---|
| **Spacecraft** | SMAP (Soil Moisture Active Passive) + MSL (Mars Science Laboratory / Curiosity) |
| **Channels** | 82 telemetry streams (A-x, B-x, C-x, D-x, E-x, F-x, G-x, M-x, P-x, R-x, S-x, T-x) |
| **Data format** | [.npy](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/test/D-3.npy) — shape `(timesteps, features)` where feature[0] is the target signal, rest are commands/contextual |
| **Length** | 1,096 – 8,640 timesteps per channel |
| **Labels** | [labeled_anomalies.csv](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/labeled_anomalies.csv) — index ranges `[start, end]` per channel |
| **Anomaly types** | `point` (isolated spike/value shift) and `contextual` (normal values, abnormal in context) |
| **Train/Test split** | Pre-split; test set is ~3× larger in the raw bytes |

#### Label Stats (from CSV, 83 rows)
- **57 channels** are SMAP, **26** are MSL
- **Point anomalies**: ~65% of all labeled sequences  
- **Contextual anomalies**: ~35%
- Some channels have **multiple anomaly windows** (up to 3, e.g., G-7, P-4)

#### Cached Run Results (2018-05-19)
The [params.log](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/params.log) reveals the original model hyperparameters and a full per-channel evaluation:

| Hyperparameter | Value |
|---|---|
| `l_s` (lookback window) | 250 |
| `n_predictions` | 10 |
| `layers` | [80, 80] (2-layer LSTM) |
| `batch_size` | 70 |
| `epochs` | 35 |
| `loss_metric` | MSE |
| `smoothing_perc` | 0.05 |
| `error_buffer` | 100 |
| `threshold p` | 0.13 |
| `dropout` | 0.3 |

**Final performance across all 82 channels:**
| Metric | Value |
|---|---|
| True Positives | 84 |
| False Positives | 11 |
| False Negatives | 21 |
| **Precision** | **88.4%** |
| **Recall** | **80.0%** |

The [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) models are already trained and saved — **we can load and use them directly** without re-training.

---

## 2. How the Original TELEMANOM Algorithm Works

The architecture is a **Prediction-Error based anomaly detector** (not a classifier):

```
Raw telemetry stream
      │
      ▼
 Sliding window (l_s=250 steps)
      │
      ▼
 LSTM model → predicts next n=10 values
      │
      ▼
 Prediction Error = |y_true - y_hat| for each step
      │
      ▼
 Error Smoothing (EWMA, smoothing_perc=5%)
      │
      ▼
 Dynamic Non-parametric Thresholding
 (uses extreme value theory / percentile p=0.13)
      │
      ▼
 Anomaly sequences (start, end index pairs)
      │
      ▼
 Compare against labeled_anomalies.csv → TP/FP/FN
```

Key insight: the LSTM is trained **only on normal data** (train set has very few or no anomalies). Anomalies show up as large prediction errors.

---

## 3. What Needs to Be Built

The [anomaly_detector.py](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/anomaly_detector.py) stub is completely empty. Here is the full module we need to implement:

### Module: [AnamalyDetectionEngine/anomaly_detector.py](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/anomaly_detector.py)

```
AnomalyDetector
├── __init__(channel_id, model_dir, data_dir)
├── load_data(split="test")          → loads the .npy file
├── load_model()                     → loads the .h5 LSTM model
├── predict()                        → generates y_hat via sliding window
├── compute_errors()                 → |y_true - y_hat|, then smooth
├── compute_threshold()              → dynamic percentile threshold
├── detect_anomalies()               → returns list of (start, end) sequences
└── evaluate(labels_df)              → TP/FP/FN/Precision/Recall
```

### Module: [ThreatClassificationEngine/threat_classifier.py](file:///D:/Projects/argus_server/argus-server/app/ai/ThreatClassificationEngine/threat_classifier.py)

Once anomalies are detected, classify their **severity and type**:

```
ThreatClassifier
├── __init__(model_path)
├── extract_features(error_seq)      → statistical features from error windows
├── classify(anomaly_window)         → returns {type, severity, confidence}
└── get_threat_label()               → maps to ARGUS threat taxonomy
```

### Integration into FastAPI services

```
app/services/pipeline.py
├── run_anomaly_pipeline(channel_id, data) → calls AnomalyDetector
├── run_threat_classification(anomalies)   → calls ThreatClassifier

app/services/alert_engine.py
├── create_alert(anomaly, classification)  → writes to MongoDB
├── trigger_defense(severity)             → calls defense service

app/api/routes/
└── anomaly.py  (new)  → POST /analyze, GET /anomalies/{channel_id}
```

---

## 4. Implementation Plan (Phased)

### Phase 1 — Core Detector (use pre-trained models)

The [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) models already exist. Wire them up:

1. **[anomaly_detector.py](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/anomaly_detector.py)** — full implementation:
   - `load_data()`: `np.load(f"{data_dir}/test/{channel_id}.npy")`
   - `load_model()`: `tf.keras.models.load_model(f"{model_dir}/{channel_id}.h5")`
   - `predict()`: sliding window of `l_s=250`, predict `n=10` steps, store as pre-saved `y_hat/*.npy`
   - `compute_errors()`: absolute error + EWMA smoothing
   - `compute_threshold()`: percentile + buffer logic from original paper
   - `detect_anomalies()`: returns `[(start, end), ...]`

2. **Update [requirements.txt](file:///D:/Projects/argus_server/argus-server/requirements.txt)** to add `tensorflow`, `numpy`, `pandas`

### Phase 2 — Threat Classifier

Since this is the ARGUS *security* context, we map detected anomalies to threat types:

| Anomaly Pattern | ARGUS Threat Label |
|---|---|
| Sudden point spike in power/voltage | `POWER_SURGE` |
| Contextual drift in comms/RF channel | `SIGNAL_JAMMING` |
| Sustained shift in attitude/orientation | `SPOOFING_ATTACK` |
| Multi-channel correlated anomaly | `COORDINATED_ATTACK` |

Use a **lightweight Random Forest classifier** trained on extracted features (mean, std, peak, duration) of the error sequences. Or rule-based for the demo.

### Phase 3 — API & Alerting

- `POST /api/v1/anomaly/analyze` → accepts `{channel_id, mode: "realtime" | "batch"}`
- `GET /api/v1/anomaly/{channel_id}` → returns detected anomalies + classifications
- `WebSocket /ws/telemetry` → streaming real-time anomaly alerts

### Phase 4 — Real-time Simulation

Replay the test [.npy](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/test/D-3.npy) data as a simulated live stream (the existing telemetry simulator from the previous conversation feeds this).

---

## 5. Technology Choices

| Component | Choice | Reason |
|---|---|---|
| LSTM model loading | `tensorflow.keras` | Models are saved as [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) (Keras format) |
| Inference runtime | `onnxruntime` (already in requirements) | Convert [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) → ONNX for faster inference |
| Data processing | `numpy`, `pandas` | Already used in original pipeline |
| Threshold algorithm | Reimplemented from telemanom paper | Already validated at 88.4% precision |
| Threat classification | `scikit-learn` RandomForest | Already in requirements |
| API | `FastAPI` + `Motor`/`Beanie` | Already in stack |

> [!IMPORTANT]
> The [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) Keras models require **TensorFlow** to load, but [requirements.txt](file:///D:/Projects/argus_server/argus-server/requirements.txt) currently only has `torch`. We must add `tensorflow` OR convert all 82 models to ONNX format first for the `onnxruntime` path.

> [!TIP]
> The ONNX path is recommended for production: convert [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) → `.onnx` once with `tf2onnx`, then use `onnxruntime` (already in requirements) for inference. This avoids a heavy TF dependency at runtime.

---

## 6. Quick Reference: Data Shapes

```python
# train/A-1.npy  → shape (N_train, F)  e.g. (576128/8/8 bytes → ~72k floats)
# test/A-1.npy   → shape (N_test, F)   e.g. ~216k floats for A-1

# F (features) varies by channel:
#   - Single-feature channels (most SMAP): F = 1  (univariate)
#   - Multi-feature channels (MSL): F = 25 (multivariate — telemetry + commands)

# y_hat/A-1.npy  → shape (N_predictions,)  — predicted values
# smoothed_errors/A-1.npy → shape (N_predictions,) — EWMA-smoothed errors
# models/A-1.h5  → Keras LSTM, input (batch, l_s=250, F), output (n_predictions=10,)
```

---

## 7. Next Steps (Recommended Order)

- [ ] Implement [anomaly_detector.py](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/anomaly_detector.py) — wire up existing [.h5](file:///D:/Projects/argus_server/argus-server/app/ai/AnamalyDetectionEngine/data/2018-05-19_15.00.10/models/M-5.h5) models
- [ ] Add `tensorflow` or implement ONNX conversion script
- [ ] Implement [threat_classifier.py](file:///D:/Projects/argus_server/argus-server/app/ai/ThreatClassificationEngine/threat_classifier.py) with rule-based + ML classification
- [ ] Fill in [app/services/pipeline.py](file:///D:/Projects/argus_server/argus-server/app/services/pipeline.py) to orchestrate the two engines
- [ ] Fill in [app/services/alert_engine.py](file:///D:/Projects/argus_server/argus-server/app/services/alert_engine.py) for MongoDB persistence
- [ ] Implement `app/api/routes/anomaly.py` REST endpoints
- [ ] Hook into real-time telemetry WebSocket stream
- [ ] Write [app/core/database.py](file:///D:/Projects/argus_server/argus-server/app/core/database.py) and [app/core/config.py](file:///D:/Projects/argus_server/argus-server/app/core/config.py)
