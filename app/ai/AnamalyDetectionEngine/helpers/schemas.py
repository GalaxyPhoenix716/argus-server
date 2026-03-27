"""
helpers/schemas.py
------------------
Pydantic-compatible dataclass holding all artefacts produced by the anomaly
detection pipeline for a single channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnomalyResult:
    """
    All outputs from running the TELEMANOM pipeline on one telemetry channel.

    Attributes
    ----------
    channel_id       : Telemetry channel identifier, e.g. "P-1".
    anomaly_sequences: Detected (start, end) index pairs aligned to original
                       time axis (offset by l_s).
    errors_smoothed  : EWMA-smoothed prediction error array.
    threshold        : Dynamic threshold value used for detection.
    y_hat            : Model predictions (length = T - l_s).
    y_true           : Ground-truth target values (same length as y_hat).
    true_positives   : Populated by evaluator.evaluate().
    false_positives  : Populated by evaluator.evaluate().
    false_negatives  : Populated by evaluator.evaluate().
    precision        : Populated by evaluator.evaluate().
    recall           : Populated by evaluator.evaluate().
    """
    channel_id: str
    anomaly_sequences: list[tuple[int, int]]
    errors_smoothed: np.ndarray
    threshold: float
    y_hat: np.ndarray
    y_true: np.ndarray
    # --- evaluation fields (set by evaluator.evaluate) ---
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> dict:
        """Serialise scalar fields to a plain dict (safe for JSON / DataFrames)."""
        return {
            "channel_id": self.channel_id,
            "anomaly_sequences": self.anomaly_sequences,
            "threshold": float(self.threshold),
            "num_anomaly_sequences": len(self.anomaly_sequences),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }
