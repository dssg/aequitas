from typing import Literal

FAIRNESS_METRICS = ["Predictive Equality", "Equal Opportunity", "Demographic Parity"]
PERFORMANCE_METRICS = ["TPR", "FPR", "FNR", "Accuracy", "Precision"]

METRIC_NAMES = {
    "Predictive Equality": "fpr_ratio",
    "Equal Opportunity": "tpr_ratio",
    "Demographic Parity": "pprev_ratio",
    "TPR": "tpr",
    "FPR": "fpr",
    "FNR": "fnr",
    "Accuracy": "accuracy",
    "Precision": "precision",
}

FAIRNESS_METRIC = Literal[
    "Predictive Equality", "Equal Opportunity", "Demographic Parity"
]
PERFORMANCE_METRIC = Literal["TPR", "FPR", "FNR", "Accuracy", "Precision"]
