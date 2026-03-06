"""Evaluation reporting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classification_report_to_df(label, pred, mask, emo_list) -> pd.DataFrame:
    """Convert per-class report to one-row DataFrame."""
    from sklearn.metrics import classification_report

    label = label[mask.astype(bool)]
    pred = pred[mask.astype(bool)]
    report = classification_report(label, pred, target_names=emo_list, output_dict=True)

    results = {"Methods": ["Model"]}
    total_len = len(label)
    f1_weighted_parts = []

    for i, emo in enumerate(emo_list):
        f1 = report[emo]["f1-score"]
        results[f"{emo} F1"] = [round(f1 * 100, 2)]

        emo_index = np.array(label) == i
        f1_weighted_parts.append(f1 * np.sum(emo_index))

    avg_f1 = sum(f1_weighted_parts) / total_len
    results["w-F1"] = [round(avg_f1 * 100, 2)]
    return pd.DataFrame(results)
