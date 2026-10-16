"""Tank and coral profiles.

Target ranges are typical reef-keeping values used by commercial coral farms
and hobbyists. They are configuration, not ground truth: edit them to match a
specific facility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class CoralProfile:
    name: str
    # Calcification demand relative to an SPS-dominated system (1.0)
    calcification_demand: float
    # Temperature above which heat stress accumulates (deg C)
    heat_threshold_c: float
    # How quickly stress turns into tissue damage (higher = more fragile)
    sensitivity: float


CORALS = {
    "sps": CoralProfile("sps", calcification_demand=1.0, heat_threshold_c=27.8, sensitivity=1.3),
    "lps": CoralProfile("lps", calcification_demand=0.6, heat_threshold_c=28.3, sensitivity=0.9),
    "soft": CoralProfile("soft", calcification_demand=0.2, heat_threshold_c=28.8, sensitivity=0.6),
}


@dataclass
class TankProfile:
    name: str = "commercial-sps-frag-system"
    volume_l: float = 1500.0
    coral: str = "sps"
    biomass: float = 1.0  # relative calcifying biomass (1.0 = fully stocked)

    # Targets
    temp_setpoint_c: float = 25.6
    salinity_target_ppt: float = 35.0
    alk_target_dkh: float = 8.3
    calcium_target_ppm: float = 430.0
    magnesium_target_ppm: float = 1350.0
    nitrate_target_ppm: float = 5.0
    phosphate_target_ppm: float = 0.05

    # Equipment
    heater_kw_per_100l: float = 0.012  # deg C per 5 min at full power, normalised
    chiller_enabled: bool = True
    chiller_setpoint_c: float = 26.6
    light_on_hour: float = 10.0
    light_off_hour: float = 20.0
    peak_par: float = 350.0  # umol/m2/s at frag level
    feeding_hour: float = 18.0
    room_temp_c: float = 22.5
    room_temp_swing_c: float = 1.5

    # Dosing (balanced two-part), dKH per day supplied by the doser at nominal
    doser_dkh_per_day: float = 1.6

    def to_dict(self) -> dict:
        return asdict(self)


# Safe operating envelope used by the control agent's guardrail.
# Anything outside these bounds is never proposed, regardless of model output.
SAFETY_LIMITS = {
    "temp_setpoint_c": (24.0, 27.0),
    "doser_multiplier": (0.0, 1.5),
    "max_alk_change_dkh_per_day": 0.5,
    "max_feeding_multiplier": 1.5,
    "water_change_pct": (0.0, 25.0),
}


# Healthy reference ranges (used for rule checks and reports)
HEALTHY_RANGES = {
    "temp_c": (24.5, 27.5),
    "salinity_ppt": (33.5, 36.0),
    "ph": (7.85, 8.45),
    "alk_dkh": (7.0, 9.5),
    "calcium_ppm": (390.0, 470.0),
    "magnesium_ppm": (1250.0, 1450.0),
    "nitrate_ppm": (1.0, 20.0),
    "phosphate_ppm": (0.02, 0.15),
}
