"""Command line interface.

    reefsynth generate --out data --n 300            # synthesize farms (Serverless Job)
    reefsynth train --data data --out models         # train detectors on synthetic data
    reefsynth run --preset summer-heatwave           # full agent pipeline on one farm
    reefsynth run --prompt "heatwave on an LPS farm" # Nemotron writes the scenario
    reefsynth eval --n 20                            # score agents on fresh unseen farms
    reefsynth serve --port 8000                      # dashboard + API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODELS = str(Path(__file__).resolve().parent.parent / "models" / "reef_models.joblib")


def _load_models(path: str):
    from .ml import ReefModels

    if not Path(path).exists():
        sys.exit(f"No models at {path}. Run `reefsynth generate` and `reefsynth train` first.")
    return ReefModels.load(path)


def cmd_generate(a):
    from .dataset import generate

    m = generate(a.out, n=a.n, days=a.days, seed0=a.seed, write_raw=not a.no_raw, workers=a.workers)
    print(json.dumps(m, indent=2))


def cmd_train(a):
    import pandas as pd

    from .ml import train

    H = pd.read_csv(Path(a.data) / "hourly.csv.gz")
    _, metrics = train(H, a.out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "fault_report"}, indent=2))


def cmd_run(a):
    from . import agents, pipeline
    from .scenarios import PRESETS, ScenarioSpec

    if a.prompt:
        spec, src = agents.scenario_from_prompt(a.prompt)
        print(f"# scenario from {src}:\n{spec.to_json()}\n", file=sys.stderr)
    elif a.spec:
        spec = ScenarioSpec.from_dict(json.loads(Path(a.spec).read_text()))
    else:
        spec = ScenarioSpec.from_dict(PRESETS[a.preset].to_dict())
    r = pipeline.run(spec, _load_models(a.models), max_incidents=a.max_incidents)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2, default=str))
    print(r["report"]["text"])
    print("\n# evaluation:", json.dumps({k: v for k, v in r["evaluation"].items() if k != "events"}, default=str))
    print("# model calls:", json.dumps(r["model_calls"]["by_tier"]))


def cmd_eval(a):
    """Score the full crew on farms generated fresh (seeds never used in training)."""
    from . import llm, pipeline
    from .scenarios import random_scenario

    models = _load_models(a.models)
    rows = []
    for i in range(a.n):
        spec = random_scenario(a.seed + i, days=a.days, max_events=2)
        r = pipeline.run(spec, models, max_incidents=3, notes=False, critic=False, report=False)
        ev = r["evaluation"]
        rows.append(ev)
        print(f"{spec.name}: faults={[e['fault'] for e in ev['events']]} det={ev['detection_rate']} "
              f"diag={ev['diagnosis_accuracy']}", file=sys.stderr)
    det = [e for r in rows for e in r["events"]]
    dia = [d for r in rows for d in r["diagnoses"] if d["truth"] != "none"]
    lat = [e["latency_hours"] for e in det if e["latency_hours"] is not None]
    out = {
        "farms": a.n,
        "nemotron_live": llm.live(),
        "fault_events": len(det),
        "detection_rate": round(sum(e["detected"] for e in det) / max(1, len(det)), 3),
        "median_latency_hours": sorted(lat)[len(lat) // 2] if lat else None,
        "diagnosis_accuracy_on_real_faults": round(sum(d["root_cause_correct"] for d in dia) / max(1, len(dia)), 3),
        "sensor_vs_process_accuracy": round(sum(d["kind_correct"] for d in dia) / max(1, len(dia)), 3),
        "false_incidents": sum(r["false_incidents"] for r in rows),
        "model_calls": llm.LEDGER.summary(),
    }
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_serve(a):
    import uvicorn

    uvicorn.run("reefsynth.api:app", host=a.host, port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="reefsynth", description="Synthetic coral farm data + Nemotron agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="synthesize a labeled dataset")
    g.add_argument("--out", default="data")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--days", type=float, default=14)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--no-raw", action="store_true", help="skip per-farm 5-minute telemetry files")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train detectors on a generated dataset")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="models")
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("run", help="run the agent pipeline on one farm")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--preset", default="summer-heatwave")
    src.add_argument("--prompt")
    src.add_argument("--spec", help="path to a scenario JSON")
    r.add_argument("--models", default=DEFAULT_MODELS)
    r.add_argument("--max-incidents", type=int, default=4)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="score the crew on fresh unseen farms")
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--days", type=float, default=10)
    e.add_argument("--seed", type=int, default=100_000)
    e.add_argument("--models", default=DEFAULT_MODELS)
    e.add_argument("--out")
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="dashboard + API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
