"""
helpers/evaluator.py
--------------------
Sequence-level evaluation of detected anomaly windows against ground truth
from the labeled_anomalies.csv file.

Functions
---------
evaluate_sequences(detected, ground_truth) → (tp, fp, fn)
evaluate(result, channel_id, labels_csv)   → AnomalyResult  (in-place update)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import _LABELS_CSV
from .schemas import AnomalyResult

logger = logging.getLogger(__name__)


def evaluate_sequences(
    detected: list[tuple[int, int]],
    ground_truth: list[tuple[int, int]],
) -> tuple[int, int, int]:
    """
    Sequence-level TP / FP / FN computation.

    Matching rule (consistent with TELEMANOM paper):
    - A detected window is a TRUE POSITIVE if it overlaps *any* ground-truth window.
    - A detected window with no overlap is a FALSE POSITIVE.
    - A ground-truth window not matched by any detection is a FALSE NEGATIVE.

    One ground-truth window can only be matched once (set-based tracking).

    Parameters
    ----------
    detected     : List of (start, end) detected anomaly windows.
    ground_truth : List of (start, end) labeled anomaly windows.

    Returns
    -------
    (true_positives, false_positives, false_negatives)
    """
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for d_start, d_end in detected:
        matched = False
        for idx, (g_start, g_end) in enumerate(ground_truth):
            if d_start <= g_end and d_end >= g_start:   # intervals overlap
                matched = True
                matched_gt.add(idx)
        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def evaluate(
    result: AnomalyResult,
    channel_id: str,
    labels_csv: Optional[Path] = None,
) -> AnomalyResult:
    """
    Populate the evaluation fields of an AnomalyResult in-place.

    Reads ground-truth anomaly windows for `channel_id` from
    `labeled_anomalies.csv`, computes TP / FP / FN, precision, and recall.

    Parameters
    ----------
    result     : AnomalyResult whose `anomaly_sequences` field will be evaluated.
    channel_id : Telemetry channel identifier.
    labels_csv : Path to labeled_anomalies.csv; defaults to the engine data/ path.

    Returns
    -------
    The same AnomalyResult with evaluation fields updated.
    """
    labels_path = labels_csv or _LABELS_CSV
    if not Path(labels_path).exists():
        logger.warning("Labels file not found: %s — skipping evaluation.", labels_path)
        return result

    df = pd.read_csv(labels_path)
    channel_rows = df[df["chan_id"] == channel_id.upper()]
    if channel_rows.empty:
        logger.warning("No labels found for channel %s.", channel_id)
        return result

    ground_truth: list[tuple[int, int]] = []
    for _, row in channel_rows.iterrows():
        seqs = ast.literal_eval(row["anomaly_sequences"])
        ground_truth.extend([(int(s), int(e)) for s, e in seqs])

    tp, fp, fn = evaluate_sequences(result.anomaly_sequences, ground_truth)

    result.true_positives = tp
    result.false_positives = fp
    result.false_negatives = fn
    result.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    result.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    logger.info(
        "%s — TP:%d  FP:%d  FN:%d  Precision:%.4f  Recall:%.4f",
        channel_id, tp, fp, fn, result.precision, result.recall,
    )
    return result
