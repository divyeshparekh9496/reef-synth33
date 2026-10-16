"""Scenario specification: what happens to the tank, when, and how badly.

A ScenarioSpec is plain JSON so it can be written by a person, sampled at
random (domain randomization), or produced by the Nemotron scenario agent
from a natural-language request.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

# Every fault the simulator knows how to inject.
# kind: "process" faults change the real water; "sensor" faults only corrupt
# the measurements (the tank is fine, the probe is lying). Telling those two
# apart is one of the hardest things an operator does, so the data labels it.
FAULTS = {
    "heater_stuck_on": {"kind": "process", "desc": "Heater relay welded on; temperature climbs past setpoint."},
    "heater_failure": {"kind": "process", "desc": "Heater dead; tank drifts toward room temperature."},
    "chiller_failure": {"kind": "process", "desc": "Chiller off; no protection against warm room or lights."},
    "heatwave": {"kind": "process", "desc": "Room temperature rises by `severity` x 6 deg C (summer heat event)."},
    "ato_failure": {"kind": "process", "desc": "Auto top-off stops; evaporation raises salinity."},
    "ato_stuck_on": {"kind": "process", "desc": "Auto top-off runs continuously; salinity falls."},
    "doser_overdose": {"kind": "process", "desc": "Alkalinity/calcium doser runs at (1 + 4 x severity) x nominal."},
    "doser_empty": {"kind": "process", "desc": "Doser reservoir empty; alkalinity and calcium are consumed and not replaced."},
    "skimmer_failure": {"kind": "process", "desc": "Protein skimmer stops; nutrient export drops."},
    "overfeeding": {"kind": "process", "desc": "Feeding at (1 + 3 x severity) x nominal; nitrate and phosphate climb."},
    "nutrient_starvation": {"kind": "process", "desc": "Over-export; nitrate and phosphate bottom out."},
    "power_outage": {"kind": "process", "desc": "Lights, heater, chiller, ATO, doser and skimmer all off."},
    "co2_spike": {"kind": "process", "desc": "Room CO2 rises (closed windows, crowded room); pH drops."},
    "ph_probe_drift": {"kind": "sensor", "desc": "pH probe drifts low; true pH is fine."},
    "temp_probe_offset": {"kind": "sensor", "desc": "Temperature probe reads high by `severity` x 1.5 deg C."},
    "sensor_dropout": {"kind": "sensor", "desc": "Telemetry gaps: random sensors stop reporting."},
}

CORAL_TYPES = ("sps", "lps", "soft")


@dataclass
class FaultEvent:
    fault: str
    start_hour: float
    duration_hours: float
    severity: float = 0.5  # 0..1

    def active(self, hour: float) -> bool:
        return self.start_hour <= hour < self.start_hour + self.duration_hours


@dataclass
class ScenarioSpec:
    name: str = "baseline"
    description: str = ""
    days: float = 14.0
    step_minutes: float = 5.0
    seed: int = 0
    coral: str = "sps"
    biomass: float = 1.0
    room_temp_c: float = 22.5
    events: list[FaultEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        events = [FaultEvent(**e) for e in d.get("events", [])]
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "events"}
        spec = cls(**allowed, events=events)
        spec.validate()
        return spec

    def validate(self) -> None:
        """Clamp and check values so model-written specs can't break the simulator."""
        if self.coral not in CORAL_TYPES:
            raise ValueError(f"coral must be one of {CORAL_TYPES}, got {self.coral!r}")
        self.days = float(min(max(self.days, 0.5), 60.0))
        self.step_minutes = float(min(max(self.step_minutes, 1.0), 60.0))
        self.biomass = float(min(max(self.biomass, 0.1), 2.0))
        self.room_temp_c = float(min(max(self.room_temp_c, 15.0), 32.0))
        total_h = self.days * 24
        clean = []
        for e in self.events:
            if e.fault not in FAULTS:
                raise ValueError(f"unknown fault {e.fault!r}; known: {sorted(FAULTS)}")
            e.severity = float(min(max(e.severity, 0.0), 1.0))
            e.start_hour = float(min(max(e.start_hour, 0.0), total_h))
            e.duration_hours = float(min(max(e.duration_hours, 0.25), total_h - e.start_hour))
            if e.duration_hours > 0:
                clean.append(e)
        self.events = clean


def random_scenario(seed: int, days: float = 14.0, max_events: int = 3,
                    allow_faults: Optional[list[str]] = None) -> ScenarioSpec:
    """Domain randomization: sample a plausible farm and a set of faults."""
    rng = random.Random(seed)
    faults = allow_faults or list(FAULTS)
    n_events = rng.choice([0] + list(range(1, max_events + 1)))
    total_h = days * 24
    events = []
    for _ in range(n_events):
        f = rng.choice(faults)
        dur = {
            "power_outage": rng.uniform(1, 8),
            "heatwave": rng.uniform(24, 96),
            "sensor_dropout": rng.uniform(2, 24),
        }.get(f, rng.uniform(6, 72))
        start = rng.uniform(24, max(25, total_h - dur))
        events.append(FaultEvent(f, round(start, 2), round(dur, 2), round(rng.uniform(0.3, 1.0), 2)))
    spec = ScenarioSpec(
        name=f"random-{seed}",
        description="Domain-randomized scenario",
        days=days,
        seed=seed,
        coral=rng.choice(CORAL_TYPES),
        biomass=round(rng.uniform(0.5, 1.4), 2),
        room_temp_c=round(rng.uniform(20.0, 25.0), 2),
        events=events,
    )
    spec.validate()
    return spec


# Hand-written scenarios that make good demos.
PRESETS = {
    "baseline": ScenarioSpec(name="baseline", description="Healthy SPS farm, no faults.", days=7, seed=1),
    "summer-heatwave": ScenarioSpec(
        name="summer-heatwave",
        description="Heatwave hits while the chiller is down; classic bleaching setup.",
        days=14, seed=2, coral="sps",
        events=[FaultEvent("chiller_failure", 72, 200, 1.0), FaultEvent("heatwave", 96, 120, 0.8)],
    ),
    "doser-runaway": ScenarioSpec(
        name="doser-runaway",
        description="Two-part doser controller glitches and overdoses alkalinity.",
        days=10, seed=3, coral="sps",
        events=[FaultEvent("doser_overdose", 60, 36, 0.8)],
    ),
    "lying-probe": ScenarioSpec(
        name="lying-probe",
        description="pH probe drifts low. Water is fine; a naive controller would dose wrongly.",
        days=10, seed=4, coral="lps",
        events=[FaultEvent("ph_probe_drift", 48, 150, 0.9)],
    ),
    "silent-starvation": ScenarioSpec(
        name="silent-starvation",
        description="Doser runs dry and nutrients bottom out; slow decline nobody notices.",
        days=21, seed=5, coral="sps",
        events=[FaultEvent("doser_empty", 96, 150, 1.0), FaultEvent("nutrient_starvation", 48, 400, 0.7)],
    ),
    "blackout": ScenarioSpec(
        name="blackout",
        description="Six-hour power outage on a cool night.",
        days=5, seed=6, coral="sps", room_temp_c=20.0,
        events=[FaultEvent("power_outage", 50, 6, 1.0)],
    ),
}
