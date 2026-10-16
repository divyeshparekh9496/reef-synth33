"""Process-based reef tank simulator.

This is a lightweight digital twin of a closed coral-farm system. It models
the main processes that decide whether a coral farm thrives or crashes:

* heat balance (room, lights, pumps, heater and chiller with hysteresis)
* evaporation and automatic top-off (salinity)
* calcification consuming alkalinity and calcium, light- and health-dependent
* a dosing controller that auto-tunes from *measured* alkalinity (so a lying
  sensor can mislead it, just like in a real farm)
* feeding and nutrient export (nitrate, phosphate)
* carbonate chemistry proxy for pH (alkalinity, photosynthesis, CO2)
* dissolved oxygen
* coral physiology: heat-stress accumulation (degree heating hours),
  bleaching index, tissue health and a categorical health label
* a sensor layer: per-probe noise, sampling intervals, spikes, drift,
  offsets and dropouts

Every row carries ground truth, measurements and labels, which is what makes
the output useful for training and evaluating detectors and agents.

The constants are tuned to produce realistic magnitudes, not to be a
peer-reviewed model. They are documented inline so they can be calibrated
against real farm logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .profiles import CORALS, TankProfile
from .scenarios import FAULTS, ScenarioSpec

START_TIME = pd.Timestamp("2026-07-01 00:00:00")

# Natural seawater-ish salt mix used for water changes
SALT_MIX = {"alk": 8.0, "ca": 420.0, "mg": 1320.0, "no3": 0.0, "po4": 0.0}

# Sensor model: (noise sd, sample every N hours or None for every step)
SENSORS = {
    "temp_c": (0.03, None),
    "ph": (0.012, None),
    "salinity_ppt": (0.08, None),
    "do_mgl": (0.1, None),
    "par": (0.02, None),  # relative
    "alk_dkh": (0.08, 2.0),
    "nitrate_ppm": (0.3, 12.0),
    "phosphate_ppm": (0.008, 12.0),
    "calcium_ppm": (6.0, 24.0),
    "magnesium_ppm": (15.0, 24.0),
}

MEASURED_COLUMNS = [f"meas_{k}" for k in SENSORS]
TRUE_COLUMNS = [f"true_{k}" for k in SENSORS]


@dataclass
class _Mods:
    """Per-step equipment modifiers derived from active faults."""

    power: bool = True
    heater_stuck: bool = False
    heater_dead: bool = False
    chiller_dead: bool = False
    room_delta: float = 0.0
    ato_mode: str = "normal"  # normal | off | stuck
    ato_stuck_sev: float = 0.0
    doser_mult: float = 1.0
    skimmer_mult: float = 1.0
    feed_mult: float = 1.0
    export_mult: float = 1.0
    co2_ph_delta: float = 0.0


def _mods_for(spec: ScenarioSpec, hour: float) -> tuple[_Mods, list[str]]:
    m = _Mods()
    active = []
    for e in spec.events:
        if not e.active(hour):
            continue
        active.append(e.fault)
        s = e.severity
        f = e.fault
        if f == "power_outage":
            m.power = False
        elif f == "heater_stuck_on":
            m.heater_stuck = True
        elif f == "heater_failure":
            m.heater_dead = True
        elif f == "chiller_failure":
            m.chiller_dead = True
        elif f == "heatwave":
            m.room_delta += 6.0 * s
        elif f == "ato_failure":
            m.ato_mode = "off"
        elif f == "ato_stuck_on":
            m.ato_mode = "stuck"
            m.ato_stuck_sev = s
        elif f == "doser_overdose":
            m.doser_mult *= 1 + 4 * s
        elif f == "doser_empty":
            m.doser_mult = 0.0
        elif f == "skimmer_failure":
            m.skimmer_mult *= 1 - 0.8 * s
        elif f == "overfeeding":
            m.feed_mult *= 1 + 3 * s
        elif f == "nutrient_starvation":
            m.export_mult *= 1 + 3 * s
            m.feed_mult *= 1 - 0.7 * s
        elif f == "co2_spike":
            m.co2_ph_delta -= 0.35 * s
    if not m.power:
        m.ato_mode = "off"
    return m, active


def simulate(spec: ScenarioSpec, tank: TankProfile | None = None) -> pd.DataFrame:
    """Run one scenario and return a labeled time series."""
    spec.validate()
    tank = tank or TankProfile(coral=spec.coral, biomass=spec.biomass, room_temp_c=spec.room_temp_c)
    coral = CORALS[spec.coral]
    rng = np.random.default_rng(spec.seed)

    dt = spec.step_minutes / 60.0  # hours
    n = int(round(spec.days * 24 / dt))

    # Small per-farm variation so no two runs are identical
    room_base = spec.room_temp_c + rng.normal(0, 0.3)
    demand = coral.calcification_demand * spec.biomass
    nominal_doser = tank.doser_dkh_per_day * demand * rng.uniform(0.9, 1.1)

    # State
    T = tank.temp_setpoint_c + rng.normal(0, 0.1)
    S = tank.salinity_target_ppt + rng.normal(0, 0.1)
    alk = tank.alk_target_dkh + rng.normal(0, 0.15)
    ca = tank.calcium_target_ppm + rng.normal(0, 8)
    mg = tank.magnesium_target_ppm + rng.normal(0, 20)
    no3 = tank.nitrate_target_ppm * rng.uniform(0.7, 1.3)
    po4 = tank.phosphate_target_ppm * rng.uniform(0.7, 1.3)
    ph = 8.15
    do = 6.8
    health = 1.0
    bleach = 0.0
    dhh = 0.0  # degree heating hours above coral threshold
    heater_on = False
    doser_ctrl = 1.0  # dosing controller's own auto-tune multiplier
    ph_drift = 0.0
    alk_hist: list[float] = []
    alk_meas_window: list[float] = []

    steps_per_day = int(round(24 / dt))
    rows = []

    for i in range(n):
        hour = i * dt
        hod = hour % 24
        m, active = _mods_for(spec, hour)

        # --- light ---------------------------------------------------------
        if m.power and tank.light_on_hour <= hod < tank.light_off_hour:
            phase = (hod - tank.light_on_hour) / (tank.light_off_hour - tank.light_on_hour)
            par = tank.peak_par * math.sin(math.pi * phase) ** 0.8
        else:
            par = 0.0
        light_frac = par / tank.peak_par

        # --- heat balance (deg C per hour) -------------------------------------
        room = room_base + tank.room_temp_swing_c * math.sin(2 * math.pi * (hod - 9) / 24) + m.room_delta
        dT = 0.03 * (room - T)
        if m.power:
            dT += 0.05 + 0.15 * light_frac  # pumps + lights
            if m.heater_stuck:
                heater_on = True
            elif m.heater_dead:
                heater_on = False
            else:
                if T < tank.temp_setpoint_c - 0.1:
                    heater_on = True
                elif T > tank.temp_setpoint_c + 0.1:
                    heater_on = False
            chiller_on = tank.chiller_enabled and not m.chiller_dead and T > tank.chiller_setpoint_c
        else:
            heater_on = False
            chiller_on = False
        if heater_on:
            dT += 0.9 if m.heater_stuck else 0.6
        if chiller_on:
            dT -= 0.8
        T += dT * dt

        # --- salinity (evaporation + ATO) ------------------------------------------
        evap = 0.010 / 24 * (1 + 0.04 * (T - 25) + 0.3 * light_frac)  # fraction of volume per hour
        S += S * evap * dt
        ato_active = False
        if m.ato_mode == "normal" and S > tank.salinity_target_ppt + 0.02:
            S -= min(S - tank.salinity_target_ppt, 0.3 * dt)
            ato_active = True
        elif m.ato_mode == "stuck":
            S -= S * (0.025 / 24) * (0.5 + m.ato_stuck_sev) * dt
            ato_active = True

        # --- calcification and dosing ------------------------------------------------
        temp_factor = max(0.0, 1 - max(0.0, T - coral.heat_threshold_c) * 0.25 - max(0.0, 23.5 - T) * 0.15)
        alk_factor = max(0.0, min(1.3, alk / tank.alk_target_dkh))
        calc_rate = (nominal_doser / tank.doser_dkh_per_day) * (tank.doser_dkh_per_day / 24 / 0.55)
        calc = calc_rate * (0.4 + 0.6 * light_frac) * health * temp_factor * alk_factor  # dKH/h
        doser_rate = (nominal_doser / 24) * doser_ctrl * m.doser_mult if m.power else 0.0
        alk += (doser_rate - calc) * dt
        ca += (doser_rate - calc) * 7.15 * dt
        mg += (doser_rate * 0.2 - calc * 0.8) * dt
        alk = max(alk, 1.0)
        ca = max(ca, 200.0)

        # --- nutrients ------------------------------------------------------------------
        feeding = tank.feeding_hour <= hod < tank.feeding_hour + 1
        if feeding:
            no3 += 1.2 * spec.biomass * m.feed_mult * dt
            po4 += 0.04 * spec.biomass * m.feed_mult * dt
        skim = m.skimmer_mult if m.power else 0.2
        k_no3 = 0.012 * (0.4 + 0.6 * skim) * m.export_mult * (0.6 + 0.4 * health)
        k_po4 = 0.025 * (0.4 + 0.6 * skim) * m.export_mult * (0.6 + 0.4 * health)
        no3 = max(0.0, no3 - k_no3 * no3 * dt)
        po4 = max(0.0, po4 - k_po4 * po4 * dt)

        # --- pH and oxygen ---------------------------------------------------------------
        ph_eq = (8.05 + 0.03 * (alk - 8.3) + 0.14 * light_frac - 0.01 * (T - 25.6) + m.co2_ph_delta
                 - (0.15 if not m.power else 0.0))
        ph += (ph_eq - ph) * min(1.0, dt / 1.5)
        do_sat = 7.0 - 0.15 * (T - 25.0)
        do_eq = do_sat + 0.3 * light_frac if m.power else 3.0
        do += (do_eq - do) * min(1.0, dt / (1.0 if m.power else 4.0))

        # --- weekly water change (10%, day 7 of each week at 11:00) -----------------
        if abs((hour % 168.0) - 155.0) < dt / 2:
            f = 0.10
            alk += f * (SALT_MIX["alk"] - alk)
            ca += f * (SALT_MIX["ca"] - ca)
            mg += f * (SALT_MIX["mg"] - mg)
            no3 += f * (SALT_MIX["no3"] - no3)
            po4 += f * (SALT_MIX["po4"] - po4)

        # --- coral physiology ---------------------------------------------------------
        over = max(0.0, T - coral.heat_threshold_c)
        dhh = dhh + over * dt if over > 0 else dhh * (1 - 0.01 * dt)
        alk_hist.append(alk)
        if len(alk_hist) > steps_per_day:
            alk_hist.pop(0)
        alk_swing = max(alk_hist) - min(alk_hist)
        stress_parts = {
            "heat": over / 2 + dhh / 48,
            "cold": max(0.0, 23.0 - T) / 2,
            "alk": max(0.0, 6.5 - alk) / 2 + max(0.0, alk - 11.0) / 2 + max(0.0, alk_swing - 1.0),
            "salinity": max(0.0, abs(S - 35.0) - 1.5) / 1.5,
            "ph": max(0.0, 7.8 - ph) * 3,
            "nutrients": max(0.0, no3 - 25) / 25 + max(0.0, po4 - 0.25) / 0.25
            + (0.4 if (no3 < 0.5 and po4 < 0.01) else 0.0),
            "oxygen": max(0.0, 4.5 - do) / 2,
        }
        stress = sum(stress_parts.values())
        if stress > 0.1:
            health -= coral.sensitivity * 0.0015 * stress * dt
        else:
            health += 0.002 * (1 - health) * dt
        health = min(1.0, max(0.0, health))
        if stress_parts["heat"] > 0:
            bleach += 0.015 * stress_parts["heat"] * coral.sensitivity * dt
        else:
            bleach -= 0.003 * bleach * dt
        bleach = min(1.0, max(0.0, bleach))

        if health < 0.5:
            hclass = "tissue_loss"
        elif bleach >= 0.4:
            hclass = "bleaching"
        elif stress > 0.3 or health < 0.85 or bleach >= 0.15:
            hclass = "stressed"
        else:
            hclass = "healthy"

        # --- sensors --------------------------------------------------------------------
        sensor_active = [f for f in active if FAULTS[f]["kind"] == "sensor"]
        if "ph_probe_drift" in active:
            sev = next(e.severity for e in spec.events if e.fault == "ph_probe_drift" and e.active(hour))
            ph_drift -= 0.004 * sev * dt
        else:
            ph_drift = 0.0  # probe recalibrated when the fault window ends
        temp_offset = 0.0
        if "temp_probe_offset" in active:
            temp_offset = 1.5 * next(e.severity for e in spec.events if e.fault == "temp_probe_offset" and e.active(hour))

        truth = {
            "temp_c": T, "ph": ph, "salinity_ppt": S, "do_mgl": do, "par": par,
            "alk_dkh": alk, "nitrate_ppm": no3, "phosphate_ppm": po4,
            "calcium_ppm": ca, "magnesium_ppm": mg,
        }
        meas = {}
        for key, (sd, every) in SENSORS.items():
            if every is not None and (i % max(1, int(round(every / dt)))) != 0:
                meas[key] = np.nan
                continue
            v = truth[key]
            v = v * (1 + rng.normal(0, sd)) if key == "par" else v + rng.normal(0, sd)
            if key == "ph":
                v += ph_drift
            if key == "temp_c":
                v += temp_offset
            if rng.random() < 0.001:  # occasional electrical spike
                v += rng.choice([-1, 1]) * 8 * sd * (50 if key == "par" else 1)
            meas[key] = max(0.0, v)
        if "sensor_dropout" in active:
            for key in SENSORS:
                if rng.random() < 0.7:
                    meas[key] = np.nan

        # Dosing controller auto-tunes once a day from the measured alkalinity
        if not math.isnan(meas["alk_dkh"]):
            alk_meas_window.append(meas["alk_dkh"])
        if i > 0 and i % steps_per_day == 0 and alk_meas_window:
            err = (tank.alk_target_dkh - float(np.mean(alk_meas_window))) / tank.alk_target_dkh
            doser_ctrl = float(min(1.5, max(0.0, doser_ctrl + 1.0 * err)))
            alk_meas_window.clear()

        row = {
            "time": START_TIME + pd.Timedelta(hours=hour),
            "hour": round(hour, 4),
            **{f"true_{k}": truth[k] for k in SENSORS},
            **{f"meas_{k}": meas[k] for k in SENSORS},
            "heater_on": heater_on,
            "chiller_on": chiller_on,
            "ato_active": ato_active,
            "lights_on": par > 0,
            "power": m.power,
            "doser_dkh_per_day": doser_rate * 24,
            "fault": "+".join(active) if active else "none",
            "primary_fault": active[0] if active else "none",
            "fault_kind": ("sensor" if active and all(FAULTS[f]["kind"] == "sensor" for f in active)
                           else ("process" if active else "none")),
            "sensor_fault_active": bool(sensor_active),
            "coral_health": health,
            "bleaching_index": bleach,
            "degree_heating_hours": dhh,
            "stress": stress,
            **{f"stress_{k}": v for k, v in stress_parts.items()},
            "health_class": hclass,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["scenario"] = spec.to_dict()
    df.attrs["tank"] = tank.to_dict()
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Compact, model-friendly summary of a run (used in agent prompts)."""
    last = df.iloc[-1]
    out = {
        "scenario": df.attrs.get("scenario", {}).get("name"),
        "hours": float(df["hour"].iloc[-1]),
        "final": {
            "coral_health": round(float(last["coral_health"]), 3),
            "bleaching_index": round(float(last["bleaching_index"]), 3),
            "health_class": last["health_class"],
        },
        "health_class_hours": {
            k: round(float(v), 1)
            for k, v in (df["health_class"].value_counts() * (df["hour"].diff().median() or 1)).items()
        },
        "faults": sorted(set(f for v in df["fault"] for f in v.split("+") if f != "none")),
        "ranges_true": {},
    }
    for c in TRUE_COLUMNS:
        out["ranges_true"][c.removeprefix("true_")] = [round(float(df[c].min()), 3), round(float(df[c].max()), 3)]
    return out
