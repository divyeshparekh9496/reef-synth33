"""Turn raw 5-minute telemetry into hourly feature rows.

Only *measured* signals and observable equipment states are used as
features. Ground truth is used only for labels. That mirrors reality: an
agent on a real farm never sees the true water chemistry, only its probes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SENSORS

MEAS = [f"meas_{k}" for k in SENSORS]
EQUIP = ["heater_on", "chiller_on", "ato_active", "lights_on", "power", "doser_dkh_per_day"]
HEALTH_ORDER = ["healthy", "stressed", "bleaching", "tissue_loss"]


def hourly_features(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    d = df.copy()
    d["hour_idx"] = (d["hour"] // 1).astype(int)

    agg = {c: "mean" for c in MEAS}
    agg.update({c: "mean" for c in EQUIP})
    h = d.groupby("hour_idx").agg(agg)
    h.columns = [c.removeprefix("meas_") for c in h.columns]

    # Missingness per hour (dropouts show up here before anywhere else)
    miss = d.groupby("hour_idx")[[f"meas_{k}" for k in ("temp_c", "ph", "salinity_ppt", "do_mgl")]].apply(
        lambda g: g.isna().mean().mean()
    )
    h["missing_frac"] = miss

    # pH spread inside the hour and daily pH amplitude (photosynthesis signal)
    h["ph_std_1h"] = d.groupby("hour_idx")["meas_ph"].std()

    # Sparse lab-style measurements: carry the last reading forward
    sparse = ["alk_dkh", "nitrate_ppm", "phosphate_ppm", "calcium_ppm", "magnesium_ppm"]
    h[sparse] = h[sparse].ffill().bfill()
    dense = ["temp_c", "ph", "salinity_ppt", "do_mgl", "par"]
    h[dense] = h[dense].interpolate(limit_direction="both")

    for c in ["temp_c", "ph", "salinity_ppt", "do_mgl", "alk_dkh", "nitrate_ppm", "phosphate_ppm", "calcium_ppm"]:
        h[f"{c}_d6h"] = h[c] - h[c].shift(6)
        h[f"{c}_d24h"] = h[c] - h[c].shift(24)
    h["ph_amp_24h"] = h["ph"].rolling(24, min_periods=6).max() - h["ph"].rolling(24, min_periods=6).min()
    h["temp_max_24h"] = h["temp_c"].rolling(24, min_periods=1).max()
    h["heater_duty_24h"] = h["heater_on"].rolling(24, min_periods=1).mean()
    h["chiller_duty_24h"] = h["chiller_on"].rolling(24, min_periods=1).mean()
    h["ato_duty_24h"] = h["ato_active"].rolling(24, min_periods=1).mean()
    # Consistency check between correlated signals: pH should follow light.
    h["ph_vs_light"] = h["ph"] - h["ph"].rolling(24, min_periods=6).mean() - 0.0003 * (h["par"] - h["par"].rolling(24, min_periods=6).mean())
    h["hour_of_day"] = (h.index % 24).astype(float)

    # Labels
    h["label_fault"] = d.groupby("hour_idx")["primary_fault"].agg(lambda s: s.mode().iat[0])
    h["label_fault_kind"] = d.groupby("hour_idx")["fault_kind"].agg(lambda s: s.mode().iat[0])
    h["label_health"] = d.groupby("hour_idx")["health_class"].last()
    sev = h["label_health"].map({k: i for i, k in enumerate(HEALTH_ORDER)})
    # Early-warning target: will corals be stressed or worse `horizon` hours from now?
    future = sev[::-1].rolling(horizon_hours, min_periods=1).max()[::-1].shift(-1)
    h["label_risk_24h"] = (future.fillna(sev) >= 1).astype(int)
    h["coral_health"] = d.groupby("hour_idx")["coral_health"].last()
    h["scenario"] = df.attrs.get("scenario", {}).get("name", "")
    return h.reset_index()


def feature_columns(h: pd.DataFrame) -> list[str]:
    drop = {"hour_idx", "scenario", "coral_health", "split"} | {c for c in h.columns if c.startswith("label_")}
    return [c for c in h.columns if c not in drop]


def as_matrix(h: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return h[cols].astype(float).to_numpy()
