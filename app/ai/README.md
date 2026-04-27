# ARGUS AI Module Documentation

This folder contains the full AI subsystem used by ARGUS server for detection, classification, and explanation of telemetry anomalies.

## Module Layout

1. `AnomalyDetectionEngine/`  
   Detects anomalous windows from telemetry time series (LSTM + statistical fallback in runtime pipeline).
2. `ThreatClassificationEngine/`  
   Classifies anomaly windows into threat classes with hybrid logic (rule-based + ML).
3. `ExplainablityEngine/`  
   Produces human-readable reasoning and top contributing features.
4. `AI_LAYER_DOCUMENTATION.md`  
   Detailed deep-dive with model rationale and metric notes.

## Runtime Data Flow (Current Server Integration)

Primary integration entry point: `app/api/routes/sim_receiver.py`.

1. Simulator frames arrive over `WS /ws/sim`.
2. Server appends frames to a rolling buffer.
3. Anomaly detection runs for monitored channels (`battery_level`, `temperature`, `signal_strength`).
4. For actionable anomaly windows, threat classifier runs.
5. Threat events are pushed to dashboard stream (`WS /stream`).
6. Classified threats are also passed to blockchain logging route layer.

## Engine Roles

### 1) Anomaly Detection Engine

Key file: `AnomalyDetectionEngine/anomaly_detector.py`

1. Loads per-channel model artefacts when available.
2. Computes reconstruction/prediction error and anomaly windows.
3. Supports evaluation against labeled anomaly metadata.
4. Runtime fallback in `sim_receiver.py` uses a lightweight statistical detector when model loading fails.

### 2) Threat Classification Engine

Key file: `ThreatClassificationEngine/threat_classifier.py`

1. Extracts engineered features from anomaly windows.
2. Runs rule-based heuristics for known patterns.
3. Optionally combines ML model confidence when fitted.
4. Produces class (`attack`, `failure`, `unknown`), confidence, risk score, and reasoning.

### 3) Explainability Engine

Key file: `ExplainablityEngine/explainability_engine.py`

1. Performs temporal pattern analysis.
2. Ranks top contributing features.
3. Generates template-based explanation text for operators.

## Configuration

Threat classification config is centralized in:

1. `ThreatClassificationEngine/config/threat_config.py`

Relevant env-backed fields include:

1. `ENVIRONMENT` (default: `development`)
2. `DEBUG` (default: `false`)
3. `LOG_LEVEL` (default: `INFO`)

## Artefacts and Data

1. Pretrained and cached anomaly artefacts are under `AnomalyDetectionEngine/data/`.
2. Training scripts are under `ThreatClassificationEngine/training/`.
3. Runtime inference in current server flow is orchestrated by API route layer, not by a separate worker process.

## Operational Notes

1. Threat emission in current integration is AI-confirmed; simulator mode alone does not directly emit threat events.
2. Only selected channels are currently monitored in live route integration.
3. Explainability engine is present and importable, but route-level wiring determines when explanations are exposed to APIs/clients.

## Recommended Reading Order

1. `AI_LAYER_DOCUMENTATION.md`
2. `AnomalyDetectionEngine/anomaly_detector.py`
3. `ThreatClassificationEngine/threat_classifier.py`
4. `ExplainablityEngine/explainability_engine.py`
