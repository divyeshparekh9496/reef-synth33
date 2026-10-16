"""End-to-end run: scenario -> synthetic farm -> detectors -> agents -> report -> evaluation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import agents, llm
from .features import hourly_features
from .ml import ReefModels
from .scenarios import FAULTS, ScenarioSpec
from .simulator import simulate, summarize


def find_incidents(h: pd.DataFrame, gap: int = 3, min_len: int = 2, max_incidents: int = 4) -> list[tuple[int, int]]:
    """Group flagged hours into incidents.

    An hour is flagged when the fault classifier is confident something is wrong or the
    anomaly detector fires. Long stretches of high coral risk with no fault signal are kept
    as their own incidents (lingering damage) only when they last 6+ hours.
    """
    warm = h["hour_idx"] >= 24
    fault_flag = (((h["fault_pred"] != "none") & (h["fault_conf"] > 0.5)) | h["anomaly"]) & warm
    risk_flag = (h["risk_24h"] > 0.6) & warm & ~fault_flag

    def spans_of(flag: pd.Series, min_n: int) -> list[list[int]]:
        out: list[list[int]] = []
        for i in h.index[flag]:
            if out and i - out[-1][1] <= gap:
                out[-1][1] = i
            else:
                out.append([i, i])
        return [s for s in out if s[1] - s[0] + 1 >= min_n]

    spans = spans_of(fault_flag, min_len) + spans_of(risk_flag, 6)
    # Keep the most serious incidents (fault spans first, then peak risk, then length) to bound model spend.
    spans.sort(key=lambda s: (-int(fault_flag.loc[s[0]:s[1]].any()),
                              -float(h.loc[s[0]:s[1], "risk_24h"].max()), -(s[1] - s[0])))
    merged: list[list[int]] = []
    for s in sorted(spans[:max_incidents]):
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(list(s))
    return [tuple(s) for s in merged]


def _truth_events(spec: ScenarioSpec) -> list[dict]:
    return [{"fault": e.fault, "kind": FAULTS[e.fault]["kind"], "start_hour": e.start_hour,
             "end_hour": e.start_hour + e.duration_hours, "severity": e.severity} for e in spec.events]


def evaluate(h: pd.DataFrame, spec: ScenarioSpec, incidents: list[dict]) -> dict:
    """Score the run against the simulator's ground truth."""
    events = _truth_events(spec)
    flagged = (((h["fault_pred"] != "none") & (h["fault_conf"] > 0.5)) | h["anomaly"]) & (h["hour_idx"] >= 24)
    detections = []
    for e in events:
        win = h[(h["hour_idx"] >= e["start_hour"]) & (h["hour_idx"] < e["end_hour"] + 24)]
        hit = win[flagged.loc[win.index]]
        detections.append({**e, "detected": not hit.empty,
                           "latency_hours": (round(max(0.0, float(hit["hour_idx"].iloc[0] - e["start_hour"])), 1) if not hit.empty
                                             else None)})
    diag_scores = []
    for inc in incidents:
        win = h[(h["hour_idx"] >= inc["start_hour"]) & (h["hour_idx"] <= inc["end_hour"])]
        faulty = win.loc[win["label_fault"] != "none", "label_fault"]
        truth = faulty.mode().iat[0] if not faulty.empty else "none"
        coral_affected = bool((win["label_health"] != "healthy").any())
        d = inc["diagnosis"]
        diag_scores.append({
            "incident": f"{inc['start_hour']}-{inc['end_hour']}",
            "truth": truth,
            "diagnosed": d["root_cause"],
            "root_cause_correct": d["root_cause"] == truth,
            "kind_correct": d["fault_kind"] == (FAULTS[truth]["kind"] if truth in FAULTS else "none"),
            "coral_affected": coral_affected,
            "source": d.get("source"),
        })
    n_det = sum(d["detected"] for d in detections)
    return {
        "events": detections,
        "detection_rate": round(n_det / len(detections), 3) if detections else None,
        "diagnoses": diag_scores,
        "diagnosis_accuracy": (round(sum(d["root_cause_correct"] for d in diag_scores) / len(diag_scores), 3)
                               if diag_scores else None),
        "false_incidents": sum(1 for d in diag_scores if d["truth"] == "none" and not d["coral_affected"]),
    }


def run(spec: ScenarioSpec, models: ReefModels, max_incidents: int = 4, notes: bool = True,
        critic: bool = True, report: bool = True) -> dict:
    t0 = time.time()
    n_calls_before = len(llm.LEDGER.calls)
    df = simulate(spec)
    h = hourly_features(df)
    h = h.join(models.predict(h))

    spans = find_incidents(h, max_incidents=max_incidents)

    def handle(span):
        s, e = span
        diag = agents.diagnose(h, s, e)
        acts = agents.propose_actions(diag)
        return {"start_hour": int(h.loc[s, "hour_idx"]), "end_hour": int(h.loc[e, "hour_idx"]),
                "peak_risk_24h": round(float(h.loc[s:e, "risk_24h"].max()), 3),
                "diagnosis": diag, "actions": acts}

    with ThreadPoolExecutor(max_workers=4) as ex:
        incidents = list(ex.map(handle, spans))
        shift_notes = []
        if notes:
            days = [(d, g) for d, g in h.groupby(h["hour_idx"] // 24)]
            shift_notes = list(ex.map(lambda x: agents.shift_note(int(x[0]), x[1]), days[:21]))

    summary = summarize(df)
    review = agents.critique(spec, summary) if critic else None
    last = h.iloc[-1]
    report_ctx = {
        "farm": spec.description or spec.name,
        "coral": spec.coral,
        "days": spec.days,
        "final_status": last["label_health"],
        "final_health": float(last["coral_health"]),
        "incidents": incidents,
    }
    report = agents.farm_report(report_ctx) if report else {"text": "", "source": None}
    evaluation = evaluate(h, spec, incidents)

    # Downsampled series for the dashboard
    series_cols = ["hour_idx", "temp_c", "ph", "salinity_ppt", "alk_dkh", "nitrate_ppm", "phosphate_ppm",
                   "do_mgl", "coral_health", "risk_24h", "anomaly_score"]
    series = h[series_cols].round(4).to_dict(orient="list")
    truth = df.iloc[::12][["hour", "true_ph", "true_temp_c", "bleaching_index"]].round(4).to_dict(orient="list")

    calls = llm.LEDGER.calls[n_calls_before:]
    return {
        "scenario": spec.to_dict(),
        "nemotron_live": llm.live(),
        "summary": summary,
        "incidents": incidents,
        "shift_notes": shift_notes,
        "report": report,
        "critique": review,
        "evaluation": evaluation,
        "series": series,
        "truth": truth,
        "model_calls": {
            "total": len(calls),
            "by_agent": pd.Series([c.get("agent", "") for c in calls]).value_counts().to_dict() if calls else {},
            "by_tier": pd.Series([c["tier"] for c in calls]).value_counts().to_dict() if calls else {},
            "models": llm.TIERS,
        },
        "seconds": round(time.time() - t0, 2),
    }
