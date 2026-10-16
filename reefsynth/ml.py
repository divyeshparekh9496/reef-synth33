"""Classical ML trained on the synthetic data.

Three models, all scikit-learn, all evaluated on *held-out farms* (split by
scenario, never by row, so the score reflects generalization to a farm the
model has never seen):

1. Anomaly detector  - IsolationForest fit only on fault-free hours.
2. Fault classifier  - HistGradientBoosting: which fault is active right now
                       (16 faults + none), including sensor-vs-process faults.
3. Early warning     - HistGradientBoosting: will corals be stressed or worse
                       in the next 24 hours?

The agents use these as fast, cheap first-pass signals and only call the
larger Nemotron models when something looks wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from .features import as_matrix, feature_columns


class ReefModels:
    def __init__(self, cols: list[str], anomaly: IsolationForest, fault: HistGradientBoostingClassifier,
                 risk: HistGradientBoostingClassifier, anomaly_threshold: float):
        self.cols = cols
        self.anomaly = anomaly
        self.fault = fault
        self.risk = risk
        self.anomaly_threshold = anomaly_threshold

    def predict(self, h: pd.DataFrame) -> pd.DataFrame:
        X = as_matrix(h, self.cols)
        Xi = np.nan_to_num(X, nan=0.0)
        out = pd.DataFrame(index=h.index)
        out["anomaly_score"] = -self.anomaly.score_samples(Xi)
        out["anomaly"] = out["anomaly_score"] > self.anomaly_threshold
        proba = self.fault.predict_proba(X)
        classes = self.fault.classes_
        top = proba.argmax(1)
        out["fault_pred"] = classes[top]
        out["fault_conf"] = proba[np.arange(len(top)), top]
        out["risk_24h"] = self.risk.predict_proba(X)[:, 1]
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ReefModels":
        return joblib.load(path)


def train(hourly: pd.DataFrame, out_dir: str | None = None, seed: int = 0) -> tuple[ReefModels, dict]:
    H = hourly.copy()
    # Skip the first day of each run: rolling features need a warm-up window.
    H = H[H["hour_idx"] >= 24]
    if "split" not in H:
        H["split"] = "train"
    # Presets are demo scenarios: keep them out of training so demos are honest.
    train_df = H[H["split"] == "train"]
    test_df = H[H["split"] == "test"]
    if test_df.empty:
        test_df = train_df

    cols = feature_columns(H)
    Xtr, Xte = as_matrix(train_df, cols), as_matrix(test_df, cols)

    normal = train_df["label_fault"] == "none"
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    iso.fit(np.nan_to_num(Xtr[normal.to_numpy()], nan=0.0))
    s_norm = -iso.score_samples(np.nan_to_num(Xtr[normal.to_numpy()], nan=0.0))
    thr = float(np.quantile(s_norm, 0.98))

    fault = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                           random_state=seed)
    fault.fit(Xtr, train_df["label_fault"])
    risk = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    risk.fit(Xtr, train_df["label_risk_24h"])

    models = ReefModels(cols, iso, fault, risk, thr)

    # ---- evaluation on held-out farms ----
    pred = models.predict(test_df)
    y_any = (test_df["label_fault"] != "none").to_numpy()
    tp = int((pred["anomaly"].to_numpy() & y_any).sum())
    metrics = {
        "train_hours": int(len(train_df)),
        "test_hours": int(len(test_df)),
        "test_farms": int(test_df["scenario"].nunique()),
        "anomaly_precision": round(tp / max(1, int(pred["anomaly"].sum())), 3),
        "anomaly_recall": round(tp / max(1, int(y_any.sum())), 3),
        "fault_macro_f1": round(float(f1_score(test_df["label_fault"], pred["fault_pred"], average="macro")), 3),
        "fault_accuracy": round(float((test_df["label_fault"].to_numpy() == pred["fault_pred"].to_numpy()).mean()), 3),
        "risk_auc": (round(float(roc_auc_score(test_df["label_risk_24h"], pred["risk_24h"])), 3)
                     if test_df["label_risk_24h"].nunique() > 1 else None),
        "fault_report": classification_report(test_df["label_fault"], pred["fault_pred"],
                                              output_dict=True, zero_division=0),
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        models.save(str(Path(out_dir) / "reef_models.joblib"))
        (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return models, metrics
