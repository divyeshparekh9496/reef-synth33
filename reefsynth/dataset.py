"""Batch dataset generation.

Generates many domain-randomized farms in parallel and writes:

    <out>/telemetry/<scenario>.csv.gz   raw 5-minute telemetry with truth + labels
    <out>/hourly.csv.gz                 hourly feature rows for ML
    <out>/scenarios.jsonl               the spec behind every run (reproducible)
    <out>/instruct.jsonl                instruction-tuning pairs (telemetry -> diagnosis)
    <out>/manifest.json                 counts, class balance, split

This is the job that runs on Nebius Serverless Jobs (see jobs/).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import hourly_features
from .scenarios import FAULTS, PRESETS, ScenarioSpec, random_scenario
from .simulator import simulate

ADVICE = {
    "none": "No fault. Keep the current routine.",
    "heater_stuck_on": "Cut power to the heater at the controller, confirm with a second thermometer, and replace the relay or heater.",
    "heater_failure": "Check the heater breaker and element; move a backup heater into the sump.",
    "chiller_failure": "Check chiller power and flow; reduce photoperiod and add surface fans until it is restored.",
    "heatwave": "Shorten the photoperiod, add evaporative cooling and fans, and watch the chiller duty cycle.",
    "ato_failure": "Check the ATO reservoir and float switch; top off slowly with fresh RODI water.",
    "ato_stuck_on": "Shut the ATO pump; restore salinity slowly with saltwater, no more than 0.5 ppt per day.",
    "doser_overdose": "Stop the alkalinity doser now. Do not try to correct quickly; let consumption bring alkalinity down.",
    "doser_empty": "Refill the doser and resume dosing gradually; raise alkalinity by no more than 0.5 dKH per day.",
    "skimmer_failure": "Clean and restart the skimmer; reduce feeding until nutrients return to range.",
    "overfeeding": "Cut feeding back to the normal ration and do a 10-15% water change if nitrate keeps rising.",
    "nutrient_starvation": "Reduce export (skimmer, media) and feed slightly more; target nitrate 2-10 ppm and phosphate 0.03-0.1 ppm.",
    "power_outage": "Restore power, run battery air pumps for oxygen, and do not feed until circulation is back.",
    "co2_spike": "Ventilate the room or add a CO2 scrubber on the skimmer air intake.",
    "ph_probe_drift": "Do not dose based on this pH reading. Recalibrate or replace the pH probe; other signals look normal.",
    "temp_probe_offset": "Verify temperature with a second calibrated thermometer before changing the setpoint.",
    "sensor_dropout": "Check the controller network and probe connections; treat missing readings as unknown, not zero.",
}


def _run_one(args: tuple[dict, str, bool]) -> dict:
    spec_d, out_dir, write_raw = args
    spec = ScenarioSpec.from_dict(spec_d)
    df = simulate(spec)
    h = hourly_features(df)
    if write_raw:
        tdir = Path(out_dir) / "telemetry"
        tdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(tdir / f"{spec.name}.csv.gz", index=False)
    return {"spec": spec.to_dict(), "hourly": h}


def _window_text(h: pd.DataFrame, end: int, hours: int = 24) -> str:
    w = h.iloc[max(0, end - hours + 1): end + 1]
    cols = ["temp_c", "ph", "salinity_ppt", "do_mgl", "alk_dkh", "nitrate_ppm", "phosphate_ppm", "calcium_ppm"]
    lines = []
    for c in cols:
        s = w[c]
        lines.append(f"{c}: start {s.iloc[0]:.2f}, end {s.iloc[-1]:.2f}, min {s.min():.2f}, max {s.max():.2f}")
    lines.append(f"heater duty 24h: {w['heater_on'].mean():.0%}, chiller duty 24h: {w['chiller_on'].mean():.0%}, "
                 f"ATO duty 24h: {w['ato_active'].mean():.0%}, missing readings: {w['missing_frac'].mean():.0%}")
    return "\n".join(lines)


def instruction_pairs(h: pd.DataFrame, every_hours: int = 12) -> list[dict]:
    """Correct-by-construction training pairs: answers come from ground truth."""
    out = []
    for end in range(24, len(h), every_hours):
        row = h.iloc[end]
        fault = row["label_fault"]
        kind = FAULTS[fault]["kind"] if fault in FAULTS else "none"
        answer = {
            "fault": fault,
            "fault_kind": kind,
            "coral_status": row["label_health"],
            "risk_next_24h": "elevated" if row["label_risk_24h"] else "low",
            "action": ADVICE.get(fault, ADVICE["none"]),
        }
        out.append({
            "instruction": "You are a coral farm operations agent. Diagnose the last 24 hours of telemetry. "
                           "Answer with JSON: fault, fault_kind, coral_status, risk_next_24h, action.",
            "input": _window_text(h, end),
            "output": json.dumps(answer),
            "scenario": row["scenario"],
        })
    return out


def generate(out_dir: str, n: int = 200, days: float = 14.0, seed0: int = 0,
             include_presets: bool = True, write_raw: bool = True, workers: int | None = None,
             test_fraction: float = 0.2) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [random_scenario(seed0 + i, days=days).to_dict() for i in range(n)]
    if include_presets:
        specs += [p.to_dict() for p in PRESETS.values()]

    workers = workers or min(8, os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run_one, [(s, str(out), write_raw) for s in specs], chunksize=4))

    rng = np.random.default_rng(seed0)
    names = [r["spec"]["name"] for r in results if r["spec"]["name"] not in PRESETS]
    test = set(rng.choice(names, size=max(1, int(len(names) * test_fraction)), replace=False)) if names else set()

    hourly = []
    with open(out / "scenarios.jsonl", "w") as fs, open(out / "instruct.jsonl", "w") as fi:
        for r in results:
            name = r["spec"]["name"]
            split = "preset" if name in PRESETS else ("test" if name in test else "train")
            fs.write(json.dumps({**r["spec"], "split": split}) + "\n")
            h = r["hourly"]
            h["split"] = split
            hourly.append(h)
            for pair in instruction_pairs(h):
                fi.write(json.dumps({**pair, "split": split}) + "\n")
    H = pd.concat(hourly, ignore_index=True)
    H.to_csv(out / "hourly.csv.gz", index=False)

    manifest = {
        "scenarios": len(results),
        "hourly_rows": int(len(H)),
        "raw_rows": int(sum(len(r["hourly"]) for r in results) * 12),
        "split_counts": H.drop_duplicates("scenario")["split"].value_counts().to_dict(),
        "fault_hours": H["label_fault"].value_counts().to_dict(),
        "health_hours": H["label_health"].value_counts().to_dict(),
        "days_per_scenario": days,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
