# ARGUS AI Layer Documentation

## 1) AI Stack Overview

The ARGUS AI subsystem is implemented as a 3-layer pipeline:

1. **Anomaly Detection Layer** (`AnomalyDetectionEngine`)
2. **Threat Classification Layer** (`ThreatClassificationEngine`)
3. **Explainability Layer** (`ExplainablityEngine`)

### End-to-End Flow

1. Raw telemetry is analyzed by the anomaly detector to find abnormal windows.
2. Detected anomaly windows are converted into engineered features.
3. Threat classifier labels each anomaly (`attack` / `failure` / `unknown`) and risk.
4. Explainability engine generates top contributing features + human-readable reasoning.

---

## 2) Layer 1: Anomaly Detection Engine

### Model

- **Primary model family**: LSTM sequence prediction (TELEMANOM-style)
- **Runtime mode in current project**:
  - Uses cached pre-trained per-channel models (`.h5`) and cached predictions (`y_hat/*.npy`)
  - Or can run live inference through the helper pipeline
- **Configured loss metric**: `mse` (from `params.log`)

### Approach

1. Create lookback windows (`l_s = 250`) from telemetry.
2. Predict future points (`n_predictions = 10`) with LSTM.
3. Compute prediction error `|y_true - y_hat|`.
4. Smooth error signal.
5. Apply dynamic thresholding.
6. Convert exceedances to anomaly windows `(start, end)`.
7. Evaluate windows against `labeled_anomalies.csv` (TP/FP/FN, precision, recall).

### Why This Was Chosen

- Telemetry anomalies are temporal and context-dependent; LSTM prediction error is strong for this pattern.
- TELEMANOM is a known baseline for spacecraft telemetry.
- Per-channel modeling handles heterogeneous sensor behavior.

### Metrics: Applicability and Current Values

| Metric | Applicable? | Current value in repo | Notes |
|---|---|---|---|
| Accuracy | No (not a direct class classifier) | N/A | Not a primary metric for this detector design |
| Precision | Yes | `0.8842105263` | From cached run final totals |
| Recall | Yes | `0.8000000000` | From cached run final totals |
| F1 Score | Derivable | `0.8400` | Computed from precision/recall |
| MSE | Yes | `macro: 20.0808984703`, `micro: 6.0603526496` | Computed from cached `y_hat` vs aligned `y_true` across 82 channels |
| MAE | Yes | `macro: 0.2784395818`, `micro: 0.1600355221` | Computed from cached predictions across 82 channels |

Additional useful logged metric:
- Mean normalized prediction error (across 82 channels): `0.0594155188`

---

## 3) Layer 2: Threat Classification Engine

### Model

- **Architecture**: Hybrid ensemble
  - Rule-based heuristics (`RuleBasedHeuristics`)
  - ML model (`HistGradientBoostingClassifier`)
- **ML class**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Class mapping**: `0=normal`, `1=failure`, `2=attack` in trainer workflow

### Approach

1. Extract rich feature set from anomaly window (residual, temporal, correlation, consistency features).
2. Run rule-based detector for known signatures (GPS spoofing, sensor injection, drift, stuck sensor, etc.).
3. If model is fitted, run ML prediction + confidence.
4. Combine via ensemble decision logic and produce:
   - `threat_class`
   - `confidence`
   - `risk_score`
   - `specific_threat_type`

### Why This Was Chosen

- Rule-based logic gives deterministic, domain-safe fallback behavior.
- HistGradientBoosting is strong on tabular engineered features, fast at inference, and robust for mixed feature scales.
- Ensemble setup improves reliability when one branch is uncertain.

### Metrics: Applicability and Current Availability

| Metric | Applicable? | Status in current repo |
|---|---|---|
| Accuracy | Yes | Implemented in evaluation functions; no persisted final score artifact found |
| Precision | Yes | Implemented (macro/weighted); no persisted final score artifact found |
| Recall | Yes | Implemented (macro/weighted); no persisted final score artifact found |
| F1 Score | Yes | Implemented (macro/weighted); no persisted final score artifact found |
| MSE | Generally no | Not a primary metric for this classifier |
| MAE | Generally no | Not a primary metric for this classifier |

Notes:
- Training/evaluation code computes full classification metrics, confusion matrix, and ROC-AUC.
- If you want report-ready numeric results in this file, run a training/evaluation pass and persist outputs.

---

## 4) Layer 3: Explainability Engine

### Model / Technique

- **Not a standalone predictive ML model**.
- Uses:
  - Temporal pattern analysis (`TemporalPatternDetector`)
  - Feature attribution scoring (current implementation: value/magnitude-based heuristic)
  - Template-based natural language generation (`ExplanationTemplates`)

### Approach

1. Receive classification output + extracted features + anomaly window.
2. Identify anomaly temporal pattern (`spike`, `drift`, `persistent`, `intermittent`).
3. Rank top feature contributors.
4. Generate context-specific explanation text (attack/failure templates).
5. Output explanation confidence + processing stats.

### Why This Was Chosen

- Operators need actionable reasoning, not only class labels.
- Template-driven explainability is deterministic and stable for mission-facing UIs.
- Temporal context improves trust and root-cause interpretation.

### Metrics: Applicability

| Metric | Applicable? | Notes |
|---|---|---|
| Accuracy | Not primary | Explainability is descriptive, not a classifier target |
| Precision | Not primary | Same reason |
| Recall | Not primary | Same reason |
| F1 Score | Not primary | Same reason |
| MSE | Not primary | Not a regression target |
| MAE | Not primary | Not a regression target |

Operational metrics actually relevant here:
- `confidence_explanation`
- `processing_time_ms`
- component latency stats from `get_processing_stats()`

---

## 5) Cross-Layer Metric Recommendation

For formal reporting, use:

1. **Anomaly Layer**: `MSE`, `MAE`, `Precision`, `Recall`, `F1`
2. **Threat Classifier**: `Accuracy`, `Precision`, `Recall`, `F1` (+ confusion matrix, ROC-AUC)
3. **Explainability Layer**: latency + explanation confidence + human validation rubric

This keeps metrics aligned with each layer's true objective.

---

## 6) Source Files Referenced

- `app/ai/AnomalyDetectionEngine/anomaly_detector.py`
- `app/ai/AnomalyDetectionEngine/data/2018-05-19_15.00.10/params.log`
- `app/ai/ThreatClassificationEngine/threat_classifier.py`
- `app/ai/ThreatClassificationEngine/helpers/threat_model.py`
- `app/ai/ThreatClassificationEngine/training/classifier_trainer.py`
- `app/ai/ThreatClassificationEngine/config/threat_config.py`
- `app/ai/ExplainablityEngine/explainability_engine.py`
- `app/ai/ExplainablityEngine/explanation_templates.py`

