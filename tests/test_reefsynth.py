import json
import os

import pytest

os.environ.pop("NEBIUS_API_KEY", None)  # tests always run the offline path

from reefsynth import agents, llm  # noqa: E402
from reefsynth.features import hourly_features  # noqa: E402
from reefsynth.scenarios import PRESETS, FaultEvent, ScenarioSpec, random_scenario  # noqa: E402
from reefsynth.simulator import simulate  # noqa: E402


def test_baseline_stays_healthy():
    df = simulate(ScenarioSpec(days=5, seed=1))
    assert (df["health_class"] == "healthy").mean() > 0.95
    assert df["true_temp_c"].between(24.5, 27.5).all()
    assert df["true_alk_dkh"].between(7.0, 9.5).all()


def test_heatwave_without_chiller_bleaches_sps():
    df = simulate(PRESETS["summer-heatwave"])
    assert df["true_temp_c"].max() > 29.5
    assert df["bleaching_index"].max() > 0.4
    assert df["coral_health"].iloc[-1] < 0.7


def test_probe_drift_only_corrupts_measurement():
    df = simulate(PRESETS["lying-probe"])
    late = df[df["hour"].between(150, 190)]
    gap = (late["true_ph"] - late["meas_ph"]).mean()
    assert gap > 0.3  # probe reads far low
    assert (late["health_class"] == "healthy").all()  # corals are fine
    assert (late["fault_kind"] == "sensor").all()


def test_doser_overdose_raises_alk():
    df = simulate(PRESETS["doser-runaway"])
    assert df["true_alk_dkh"].max() > 11


def test_spec_validation_clamps_and_rejects():
    spec = ScenarioSpec.from_dict({"days": 999, "events": [
        {"fault": "heatwave", "start_hour": -5, "duration_hours": 10_000, "severity": 3}]})
    assert spec.days == 60
    e = spec.events[0]
    assert e.start_hour == 0 and e.severity == 1.0 and e.start_hour + e.duration_hours <= 60 * 24
    with pytest.raises(ValueError):
        ScenarioSpec.from_dict({"events": [{"fault": "alien_invasion", "start_hour": 1, "duration_hours": 1}]})


def test_random_scenarios_are_reproducible():
    a, b = random_scenario(7), random_scenario(7)
    assert a.to_dict() == b.to_dict()
    assert simulate(a)["true_ph"].sum() == pytest.approx(simulate(b)["true_ph"].sum())


def test_hourly_features_have_no_truth_leak():
    h = hourly_features(simulate(ScenarioSpec(days=3, seed=2)))
    from reefsynth.features import feature_columns

    cols = feature_columns(h)
    assert not any(c.startswith(("true_", "label_")) for c in cols)
    assert "coral_health" not in cols


def test_safety_guard_blocks_unsafe_and_sensor_driven_actions():
    g = agents.SafetyGuard()
    v = g.check({"action": "set_temp_setpoint", "param": 31}, {"fault_kind": "process"})
    assert v["param"] == 27.0 and v["status"] == "pending_approval"
    v = g.check({"action": "set_doser_multiplier", "param": 0.5}, {"fault_kind": "sensor"})
    assert v["status"] == "rejected"
    v = g.check({"action": "launch_rockets", "param": 1}, {"fault_kind": "process"})
    assert v["status"] == "rejected"


def test_offline_scenario_agent_falls_back():
    spec, src = agents.scenario_from_prompt("a heatwave on the farm")
    assert src == "fallback" and spec.name == "summer-heatwave"


def test_extract_json():
    assert llm.extract_json('sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json("<think>x</think> [1, 2]") == [1, 2]
    assert llm.extract_json("no json here") is None


def test_pipeline_end_to_end_with_bundled_models():
    from reefsynth.cli import DEFAULT_MODELS
    from reefsynth.ml import ReefModels
    from reefsynth.pipeline import run

    if not os.path.exists(DEFAULT_MODELS):
        pytest.skip("bundled models not present")
    r = run(PRESETS["doser-runaway"], ReefModels.load(DEFAULT_MODELS))
    assert r["incidents"], "doser runaway should raise an incident"
    assert r["evaluation"]["detection_rate"] == 1.0
    json.dumps(r, default=str)  # dashboard payload must serialize


def test_api_smoke():
    from fastapi.testclient import TestClient

    from reefsynth.api import app

    c = TestClient(app)
    st = c.get("/api/status").json()
    assert "summer-heatwave" in st["presets"]
    r = c.post("/api/run", json={"preset": "blackout"})
    assert r.status_code == 200 and "series" in r.json()
    csv = c.get("/api/telemetry.csv?preset=baseline")
    assert csv.status_code == 200 and csv.text.startswith("time,hour,")
