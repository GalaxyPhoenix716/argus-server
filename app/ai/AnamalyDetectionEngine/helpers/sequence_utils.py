"""
helpers/sequence_utils.py
--------------------------
Low-level utilities for working with contiguous boolean sequences,
including extraction, buffering, merging, and anomaly window detection.

Functions
---------
count_sequences(mask)                          → int
extract_sequences(mask)                        → [(start, end), ...]
merge_sequences(sequences)                     → [(start, end), ...]
detect_anomalies(errors_smoothed, threshold, config, channel_id)
    → [(start, end), ...]   aligned to original time axis
"""

from __future__ import annotations

import logging

import numpy as np

from .config import ChannelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive sequence operations
# ---------------------------------------------------------------------------
def count_sequences(mask: np.ndarray) -> int:
    """
    Count the number of contiguous True regions in a boolean array.

    Example
    -------
    >>> count_sequences(np.array([False, True, True, False, True]))
    2
    """
    count = 0
    in_seq = False
    for v in mask:
        if v and not in_seq:
            count += 1
            in_seq = True
        elif not v:
            in_seq = False
    return count


def extract_sequences(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for each contiguous True region.

    Indices are inclusive on both ends.
    """
    sequences: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            sequences.append((start, i - 1))
            start = None
    if start is not None:
        sequences.append((start, len(mask) - 1))
    return sequences


def merge_sequences(sequences: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping or immediately adjacent (start, end) windows into the
    smallest set of non-overlapping intervals.
    """
    if not sequences:
        return []
    sequences = sorted(sequences)
    merged = [list(sequences[0])]
    for start, end in sequences[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# High-level anomaly window detection
# ---------------------------------------------------------------------------
def detect_anomalies(
    errors_smoothed: np.ndarray,
    threshold: float,
    config: ChannelConfig,
    channel_id: str = "",
) -> list[tuple[int, int]]:
    """
    Extract anomaly windows from a smoothed error signal.

    Steps:
    1. Threshold: mask = errors_smoothed > threshold
    2. Extract contiguous True regions → raw sequences
    3. Expand each sequence by ±error_buffer timesteps
    4. Merge overlapping buffered sequences
    5. Shift by l_s to align indices with the original time axis

    Parameters
    ----------
    errors_smoothed : EWMA-smoothed prediction error array (length T - l_s).
    threshold       : Detection threshold (from signal_processing.compute_threshold).
    config          : ChannelConfig; uses error_buffer and l_s.
    channel_id      : Used for log messages.

    Returns
    -------
    List of (start, end) tuples in the **original** time axis coordinate system.
    """
    above_mask = errors_smoothed > threshold
    raw = extract_sequences(above_mask)

    # Expand each window by error_buffer on both sides, clamped to array bounds
    n = len(errors_smoothed)
    buffered = [
        (max(0, s - config.error_buffer), min(n - 1, e + config.error_buffer))
        for s, e in raw
    ]
    merged = merge_sequences(buffered)

    # Re-align to original time axis (errors start at index l_s)
    aligned = [(s + config.l_s, e + config.l_s) for s, e in merged]

    logger.info(
        "%s — %d anomaly sequence(s) detected above threshold %.6f",
        channel_id, len(aligned), threshold,
    )
    return aligned
