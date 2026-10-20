"""Pairwise clustering metrics for evaluating deduplication.

Dedup quality is about pairs: a predicted pair is correct if both documents
truly are copies of each other. Counting pairs through the contingency table
of (predicted, true) labels avoids the O(N^2) enumeration of all pairs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def _pairs(counts: Counter) -> int:
    return sum(c * (c - 1) // 2 for c in counts.values())


def pairwise_scores(predicted: Sequence[int], truth: Sequence[int]) -> dict[str, float]:
    """Pairwise precision, recall and F1 of a predicted clustering.

    With no predicted pairs, precision is defined as 1.0 (nothing claimed,
    nothing wrong); with no true pairs, recall is 1.0.
    """
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same length")
    pred_pairs = _pairs(Counter(predicted))
    true_pairs = _pairs(Counter(truth))
    tp = _pairs(Counter(zip(predicted, truth, strict=True)))
    precision = tp / pred_pairs if pred_pairs else 1.0
    recall = tp / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_pairs": float(true_pairs),
        "predicted_pairs": float(pred_pairs),
    }
