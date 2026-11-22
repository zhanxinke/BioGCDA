"""Manuscript metrics; AP and trapezoidal AUPR are separate quantities."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, auc,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, roc_auc_score)


def metrics(labels, probabilities):
    predicted = probabilities >= 0.5
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return dict(auc=float(roc_auc_score(labels, probabilities)),
                ap=float(average_precision_score(labels, probabilities)),
                aupr_trapezoid=float(auc(recall, precision)),
                accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                f1=float(f1_score(labels, predicted, zero_division=0)),
                mcc=float(matthews_corrcoef(labels, predicted)))


def report_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32).astype(np.float64)
    if len(labels) == 0 or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Predictions must be nonempty, finite, and aligned with labels.')
    predicted = scores >= 0.5
    if len(np.unique(labels)) == 2:
        result = metrics(labels, scores)
    else:
        result = dict(accuracy=float(accuracy_score(labels, predicted)),
                      precision=float(precision_score(labels, predicted, zero_division=0)),
                      f1=float(f1_score(labels, predicted, zero_division=0)),
                      mcc=float(matthews_corrcoef(labels, predicted)),
                      auc=None, ap=None, aupr_trapezoid=None)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    return result | dict(n=len(labels), correct=tp+tn, tp=tp, tn=tn, fp=fp, fn=fn, threshold=0.5)
