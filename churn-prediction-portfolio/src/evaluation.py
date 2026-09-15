"""Metrics and operational policy; thresholds are learned on validation only."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score)


def choose_threshold(y, probability, target_recall=0.8):
    """Highest score cutoff achieving target recall on validation labels.

    Among nested flagged sets this minimises contact volume. This is an
    illustrative service target, not an estimated profit optimum.
    """
    y, p = np.asarray(y), np.asarray(probability)
    if not 0 < target_recall <= 1 or not (y == 1).any():
        raise ValueError('Recall target must be in (0,1] and labels need positives.')
    positives = np.sort(p[y == 1])[::-1]
    return float(positives[int(np.ceil(target_recall * len(positives))) - 1])


def measure(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier_score': float(brier_score_loss(y, p)),
            'accuracy': float(accuracy_score(y, prediction)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'flagged': int(prediction.sum()), 'flagged_fraction': float(prediction.mean()),
            'threshold': float(threshold),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def capacity_metrics(y, p, fraction=0.2):
    y, p = np.asarray(y), np.asarray(p)
    # Stable sorting makes ties reproducible. In production ties need a policy.
    k = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-p, kind='stable')[:k]
    caught = int(y[chosen].sum())
    return {'capacity_fraction': fraction, 'contacts': k, 'churners_captured': caught,
            'precision': caught / k, 'recall': caught / int(y.sum()),
            'lift_over_random': float((caught / k) / y.mean())}


def bootstrap_intervals(y, p, threshold, repetitions=500, seed=2026):
    """Percentile intervals conditional on this fitted model and test sample.

    These do not include uncertainty from retraining or from dataset shift.
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ['roc_auc', 'average_precision', 'precision', 'recall']}
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) != 2:
            continue
        result = measure(y[idx], p[idx], threshold)
        for k in values:
            values[k].append(result[k])
    return {k: [float(v) for v in np.quantile(a, [0.025, 0.975])] for k, a in values.items()}
