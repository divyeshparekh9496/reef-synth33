"""The agent crew.

    ScenarioAgent   (Nemotron Super)  natural language -> validated ScenarioSpec
    CriticAgent     (Nemotron Super)  LLM-as-judge review of generated data
    WatchAgent      (Nemotron Nano)   cheap per-day shift notes from telemetry
    DiagnosisAgent  (Nemotron Ultra)  root cause, sensor-vs-process call, evidence
    ControlAgent    (Nemotron Super)  proposes actions; SafetyGuard enforces limits;
                                      every action waits for human approval
    ReportAgent     (Nemotron Ultra)  farm health report for the operator

The classical ML models (ml.py) run first on every hour. The agents are only
called for hours the models flag, which is what keeps Token Factory usage
small enough to run a farm on a hackathon credit budget.

Every agent has a deterministic fallback so the pipeline still runs offline.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field

import pandas as pd

from . import llm
from .dataset import ADVICE
from .profiles import HEALTHY_RANGES, SAFETY_LIMITS
from .scenarios import FAULTS, PRESETS, ScenarioSpec

# ---------------------------------------------------------------------------
# ScenarioAgent
# ---------------------------------------------------------------------------

SCENARIO_SYSTEM = f"""You design scenarios for a coral farm digital twin that generates synthetic training data.
Return ONLY a JSON object with these keys:
  name (kebab-case string), description (one sentence), days (number 1-30),
  coral ("sps" | "lps" | "soft"), biomass (0.3-1.5), room_temp_c (18-28), seed (integer),
  events: list of {{fault, start_hour, duration_hours, severity (0-1)}}
Allowed faults and what they do:
{json.dumps({k: v["desc"] for k, v in FAULTS.items()}, indent=1)}
Rules: start_hour + duration_hours must fit inside days*24. Leave at least 24 hours of normal
operation at the start so detectors have a baseline. Prefer realistic combinations (for example
a heatwave is far more dangerous while the chiller is down)."""


def _fallback_scenario(prompt: str) -> ScenarioSpec:
    p = prompt.lower()
    keywords = {
        "heat": "summer-heatwave", "bleach": "summer-heatwave", "chiller": "summer-heatwave",
        "dos": "doser-runaway", "alk": "doser-runaway", "probe": "lying-probe", "ph": "lying-probe",
        "sensor": "lying-probe", "starv": "silent-starvation", "nutrient": "silent-starvation",
        "power": "blackout", "outage": "blackout",
    }
    for k, name in keywords.items():
        if k in p:
            return ScenarioSpec.from_dict(PRESETS[name].to_dict())
    return ScenarioSpec.from_dict(PRESETS["baseline"].to_dict())


def scenario_from_prompt(prompt: str) -> tuple[ScenarioSpec, str]:
    """Returns (spec, source) where source is 'nemotron' or 'fallback'."""
    raw = llm.chat("super", SCENARIO_SYSTEM, prompt, max_tokens=900, agent="scenario")
    data = llm.extract_json(raw)
    if isinstance(data, dict):
        try:
            data.setdefault("seed", zlib.crc32(prompt.encode()) % 10_000)
            spec = ScenarioSpec.from_dict(data)
            spec.name = re.sub(r"[^a-z0-9-]+", "-", spec.name.lower()).strip("-") or "custom"
            return spec, "nemotron"
        except (TypeError, ValueError):
            pass
    return _fallback_scenario(prompt), "fallback"


# ---------------------------------------------------------------------------
# CriticAgent: LLM-as-judge for synthetic data quality
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are a senior aquaculture scientist reviewing SYNTHETIC coral farm data for realism.
You get the scenario and a summary of the simulated water chemistry and coral outcome.
Judge whether magnitudes, timings and coral responses are physically and biologically plausible.
Return ONLY JSON: {"plausibility": 1-5, "issues": [short strings], "keep": true|false}"""


def critique(spec: ScenarioSpec, summary: dict) -> dict:
    payload = json.dumps({"scenario": spec.to_dict(), "summary": summary})
    data = llm.extract_json(llm.chat("super", CRITIC_SYSTEM, payload, max_tokens=500, agent="critic"))
    if isinstance(data, dict) and "plausibility" in data:
        data["source"] = "nemotron"
        return data
    # Rule-based fallback: hard physical bounds
    issues = []
    r = summary["ranges_true"]
    if r["temp_c"][1] > 34 or r["temp_c"][0] < 15:
        issues.append("temperature outside physically plausible tank range")
    if r["salinity_ppt"][1] > 45 or r["salinity_ppt"][0] < 25:
        issues.append("salinity outside plausible range")
    if r["ph"][0] < 7.4 or r["ph"][1] > 8.7:
        issues.append("pH outside plausible range")
    return {"plausibility": 5 - min(4, len(issues) * 2), "issues": issues, "keep": not issues, "source": "rules"}


# ---------------------------------------------------------------------------
# WatchAgent: cheap shift notes
# ---------------------------------------------------------------------------

WATCH_SYSTEM = """You write the end-of-day log entry for a coral farm technician.
Be brief (max 3 sentences), concrete, and use the numbers given. Mention anything outside
healthy range and any equipment that ran unusually. No speculation about causes."""


def _day_stats(day: pd.DataFrame) -> dict:
    cols = ["temp_c", "ph", "salinity_ppt", "alk_dkh", "nitrate_ppm", "phosphate_ppm", "do_mgl"]
    return {
        **{c: [round(float(day[c].min()), 2), round(float(day[c].max()), 2)] for c in cols},
        "heater_duty": round(float(day["heater_on"].mean()), 2),
        "chiller_duty": round(float(day["chiller_on"].mean()), 2),
        "missing_frac": round(float(day["missing_frac"].mean()), 2),
        "max_risk_24h": round(float(day["risk_24h"].max()), 2) if "risk_24h" in day else None,
    }


def shift_note(day_idx: int, day: pd.DataFrame) -> dict:
    stats = _day_stats(day)
    text = llm.chat("nano", WATCH_SYSTEM, f"Day {day_idx + 1} stats (min,max): {json.dumps(stats)}",
                    max_tokens=200, agent="watch")
    if not text:
        out = []
        key_map = {"temp_c": "temp_c", "ph": "ph", "salinity_ppt": "salinity_ppt", "alk_dkh": "alk_dkh",
                   "nitrate_ppm": "nitrate_ppm", "phosphate_ppm": "phosphate_ppm"}
        for c, rk in key_map.items():
            lo, hi = HEALTHY_RANGES[rk]
            mn, mx = stats[c]
            if mn < lo or mx > hi:
                out.append(f"{c} ranged {mn}-{mx} (healthy {lo}-{hi})")
        text = (f"Day {day_idx + 1}: " + ("; ".join(out) if out else "all parameters in range")
                + f". Heater duty {stats['heater_duty']:.0%}, chiller duty {stats['chiller_duty']:.0%}.")
        src = "fallback"
    else:
        src = "nemotron"
    return {"day": day_idx + 1, "note": text, "stats": stats, "source": src}


# ---------------------------------------------------------------------------
# DiagnosisAgent
# ---------------------------------------------------------------------------

DIAG_SYSTEM = f"""You are the diagnosis agent for a commercial coral farm. You receive hourly telemetry
around an incident, equipment duty cycles, and the outputs of fast ML detectors (which can be wrong).
Your job:
1. Identify the most likely root cause from this list: {sorted(FAULTS)} or "none".
2. Decide whether it is a PROCESS fault (the water is really changing) or a SENSOR fault (a probe is
   lying). Check consistency between correlated signals: e.g. a real pH drop usually comes with lower
   alkalinity, CO2, or a power loss; a pH reading that drifts alone while alkalinity, light and oxygen
   look normal points to the probe. A temperature reading that jumps without the heater or chiller
   reacting points to the probe.
3. Estimate the risk to the corals over the next 24 hours.
Return ONLY JSON:
{{"root_cause": str, "fault_kind": "process"|"sensor"|"none", "confidence": 0-1,
  "evidence": [short strings citing numbers], "coral_risk": "low"|"medium"|"high",
  "explanation": "2-3 sentences for the farm operator"}}"""


def _incident_table(h: pd.DataFrame, start: int, end: int) -> str:
    lo, hi = max(0, start - 12), min(len(h), end + 6)
    cols = ["hour_idx", "temp_c", "ph", "salinity_ppt", "do_mgl", "par", "alk_dkh", "nitrate_ppm",
            "phosphate_ppm", "heater_on", "chiller_on", "ato_active", "power", "missing_frac",
            "fault_pred", "fault_conf", "risk_24h"]
    w = h.iloc[lo:hi:2][cols].copy()
    return w.round(3).to_csv(index=False)


def diagnose(h: pd.DataFrame, start: int, end: int) -> dict:
    window = h.iloc[start:end + 1]
    ml_votes = window["fault_pred"].value_counts()
    confident = window.loc[(window["fault_pred"] != "none") & (window["fault_conf"] > 0.5), "fault_pred"]
    ml_top = confident.mode().iat[0] if not confident.empty else "none"
    prompt = (f"Incident hours {int(h.iloc[start]['hour_idx'])}-{int(h.iloc[end]['hour_idx'])}.\n"
              f"ML detector votes over the incident: {ml_votes.head(4).to_dict()}\n"
              f"Telemetry (every 2nd hour, from 12h before to 6h after):\n{_incident_table(h, start, end)}")
    data = llm.extract_json(llm.chat("ultra", DIAG_SYSTEM, prompt, max_tokens=1200, agent="diagnosis"))
    if isinstance(data, dict) and data.get("root_cause"):
        data["source"] = "nemotron"
        if data["root_cause"] not in FAULTS and data["root_cause"] != "none":
            data["root_cause_raw"] = data["root_cause"]
            data["root_cause"] = ml_top
        return data
    kind = FAULTS[ml_top]["kind"] if ml_top in FAULTS else "none"
    risk = float(window["risk_24h"].max())
    return {
        "root_cause": ml_top,
        "fault_kind": kind,
        "confidence": round(float(window["fault_conf"].mean()), 2),
        "evidence": [f"ML detector voted {ml_top} in {int((window['fault_pred'] == ml_top).sum())} of {len(window)} hours"],
        "coral_risk": "high" if risk > 0.7 else ("medium" if risk > 0.4 else "low"),
        "explanation": FAULTS.get(ml_top, {}).get(
            "desc", "No equipment fault identified; elevated coral risk from earlier damage."),
        "source": "fallback",
    }


# ---------------------------------------------------------------------------
# ControlAgent + SafetyGuard
# ---------------------------------------------------------------------------

ACTIONS = {
    "set_temp_setpoint": {"param": "value_c", "desc": "Change heater setpoint (deg C)"},
    "set_doser_multiplier": {"param": "value", "desc": "Scale alkalinity/calcium dosing (1.0 = normal)"},
    "set_feeding_multiplier": {"param": "value", "desc": "Scale daily feeding (1.0 = normal)"},
    "water_change": {"param": "percent", "desc": "Do a water change of N percent"},
    "disable_equipment": {"param": "device", "desc": "Turn off heater|ato|doser"},
    "inspect_equipment": {"param": "device", "desc": "Ask a technician to inspect a device"},
    "recalibrate_probe": {"param": "probe", "desc": "Ask a technician to recalibrate ph|temp|salinity probe"},
    "alert_operator": {"param": "message", "desc": "Page the on-call operator"},
}

CONTROL_SYSTEM = f"""You are the control agent for a coral farm. Given a diagnosis, propose at most 3 actions.
Use ONLY these actions: {json.dumps(ACTIONS)}
Hard limits (anything outside is rejected automatically): {json.dumps(SAFETY_LIMITS)}
Prefer slow, reversible corrections. Never chase a reading from a probe that may be faulty.
Return ONLY a JSON list: [{{"action": str, "param": value, "reason": str}}]"""


@dataclass
class SafetyGuard:
    limits: dict = field(default_factory=lambda: dict(SAFETY_LIMITS))

    def check(self, proposal: dict, diagnosis: dict) -> dict:
        a = proposal.get("action")
        p = proposal.get("param")
        verdict = {"action": a, "param": p, "reason": proposal.get("reason", ""), "status": "pending_approval",
                   "guard": "ok"}
        if a not in ACTIONS:
            verdict.update(status="rejected", guard="unknown action")
            return verdict
        try:
            if a == "set_temp_setpoint":
                lo, hi = self.limits["temp_setpoint_c"]
                v = float(p)
                if not lo <= v <= hi:
                    verdict.update(param=min(hi, max(lo, v)), guard=f"clamped to [{lo}, {hi}]")
            elif a == "set_doser_multiplier":
                lo, hi = self.limits["doser_multiplier"]
                v = float(p)
                if not lo <= v <= hi:
                    verdict.update(param=min(hi, max(lo, v)), guard=f"clamped to [{lo}, {hi}]")
            elif a == "set_feeding_multiplier":
                v = float(p)
                hi = self.limits["max_feeding_multiplier"]
                if v > hi or v < 0:
                    verdict.update(param=min(hi, max(0.0, v)), guard=f"clamped to [0, {hi}]")
            elif a == "water_change":
                lo, hi = self.limits["water_change_pct"]
                v = float(p)
                if not lo <= v <= hi:
                    verdict.update(param=min(hi, max(lo, v)), guard=f"clamped to [{lo}, {hi}]")
        except (TypeError, ValueError):
            verdict.update(status="rejected", guard="invalid parameter")
        # Never act on chemistry when the diagnosis says the sensor is the problem.
        if diagnosis.get("fault_kind") == "sensor" and a in {"set_temp_setpoint", "set_doser_multiplier",
                                                               "water_change"}:
            verdict.update(status="rejected", guard="sensor fault suspected: no chemistry changes")
        return verdict


_FALLBACK_ACTIONS = {
    "heater_stuck_on": [("disable_equipment", "heater"), ("inspect_equipment", "heater")],
    "heater_failure": [("inspect_equipment", "heater")],
    "chiller_failure": [("inspect_equipment", "chiller"), ("set_feeding_multiplier", 0.7)],
    "heatwave": [("set_feeding_multiplier", 0.7), ("alert_operator", "Heat stress risk: add cooling")],
    "ato_failure": [("inspect_equipment", "ato")],
    "ato_stuck_on": [("disable_equipment", "ato"), ("inspect_equipment", "ato")],
    "doser_overdose": [("set_doser_multiplier", 0.0), ("inspect_equipment", "doser")],
    "doser_empty": [("inspect_equipment", "doser"), ("set_doser_multiplier", 1.2)],
    "skimmer_failure": [("inspect_equipment", "skimmer"), ("set_feeding_multiplier", 0.7)],
    "overfeeding": [("set_feeding_multiplier", 1.0), ("water_change", 10)],
    "nutrient_starvation": [("set_feeding_multiplier", 1.3)],
    "power_outage": [("alert_operator", "Power outage: start battery air pumps")],
    "co2_spike": [("alert_operator", "Ventilate room; pH depressed by CO2")],
    "ph_probe_drift": [("recalibrate_probe", "ph")],
    "temp_probe_offset": [("recalibrate_probe", "temp")],
    "sensor_dropout": [("inspect_equipment", "controller network")],
}


def propose_actions(diagnosis: dict, guard: SafetyGuard | None = None) -> list[dict]:
    guard = guard or SafetyGuard()
    data = llm.extract_json(llm.chat("super", CONTROL_SYSTEM, json.dumps(diagnosis), max_tokens=600,
                                     agent="control"))
    source = "nemotron"
    if isinstance(data, dict):
        data = data.get("actions")
    if not isinstance(data, list) or not data:
        source = "fallback"
        data = [{"action": a, "param": p, "reason": ADVICE.get(diagnosis.get("root_cause", "none"), "")}
                for a, p in _FALLBACK_ACTIONS.get(diagnosis.get("root_cause"), [])]
    out = []
    for prop in data[:3]:
        if isinstance(prop, dict):
            v = guard.check(prop, diagnosis)
            v["source"] = source
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

REPORT_SYSTEM = """You write the farm health report for a coral farm owner. Use plain language and the
facts given. Structure: one-line status, what happened (incidents in time order), coral impact,
actions waiting for approval, and what to watch next. Max 220 words. Do not invent facts."""


def farm_report(context: dict) -> dict:
    text = llm.chat("ultra", REPORT_SYSTEM, json.dumps(context, default=str), max_tokens=900, agent="report")
    if text:
        return {"text": text, "source": "nemotron"}
    incs = context.get("incidents", [])
    lines = [f"Status: {context['final_status']} (coral health {context['final_health']:.0%})."]
    for inc in incs:
        d = inc["diagnosis"]
        lines.append(f"- Hours {inc['start_hour']}-{inc['end_hour']}: {d['root_cause']} "
                     f"({d['fault_kind']}, risk {d['coral_risk']}). {d['explanation']}")
    if not incs:
        lines.append("No incidents detected.")
    pend = [a for inc in incs for a in inc["actions"] if a["status"] == "pending_approval"]
    lines.append(f"{len(pend)} action(s) waiting for approval.")
    return {"text": "\n".join(lines), "source": "fallback"}
