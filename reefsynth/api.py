"""FastAPI app: the ReefSynth dashboard and JSON API.

Deploy as a Nebius Serverless Endpoint (see README) or run locally:

    uvicorn reefsynth.api:app --port 8000
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import __version__, agents, llm, pipeline
from .ml import ReefModels
from .scenarios import FAULTS, PRESETS, ScenarioSpec
from .simulator import simulate

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("REEF_MODELS", ROOT.parent / "models" / "reef_models.joblib"))
METRICS_PATH = MODEL_PATH.parent / "metrics.json"

app = FastAPI(title="ReefSynth", version=__version__)
_models: ReefModels | None = None
_lock = threading.Lock()
_last_spec: ScenarioSpec | None = None


def models() -> ReefModels:
    global _models
    with _lock:
        if _models is None:
            if not MODEL_PATH.exists():
                # First boot without bundled models: build a small dataset and train (about 30s).
                from .dataset import generate
                from .ml import train
                import pandas as pd

                data_dir = MODEL_PATH.parent / "bootstrap-data"
                generate(str(data_dir), n=60, days=10, write_raw=False)
                train(pd.read_csv(data_dir / "hourly.csv.gz"), str(MODEL_PATH.parent))
            try:
                _models = ReefModels.load(str(MODEL_PATH))
            except Exception:
                # Bundled models pickled with a different scikit-learn: retrain locally.
                from .dataset import generate
                from .ml import train
                import pandas as pd

                data_dir = MODEL_PATH.parent / "bootstrap-data"
                generate(str(data_dir), n=60, days=10, write_raw=False)
                _models, _ = train(pd.read_csv(data_dir / "hourly.csv.gz"), str(MODEL_PATH.parent))
    return _models


class ScenarioRequest(BaseModel):
    prompt: str


class RunRequest(BaseModel):
    preset: str | None = None
    prompt: str | None = None
    spec: dict | None = None
    max_incidents: int = 4


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    if metrics:
        metrics = {k: v for k, v in metrics.items() if k != "fault_report"}
    return {
        "version": __version__,
        "nemotron_live": llm.live(),
        "models": llm.TIERS,
        "presets": {k: v.description for k, v in PRESETS.items()},
        "faults": {k: v for k, v in FAULTS.items()},
        "ml_metrics": metrics,
    }


@app.post("/api/scenario")
def scenario(req: ScenarioRequest):
    spec, source = agents.scenario_from_prompt(req.prompt)
    return {"spec": spec.to_dict(), "source": source}


@app.post("/api/run")
def run(req: RunRequest):
    global _last_spec
    try:
        if req.spec:
            spec = ScenarioSpec.from_dict(req.spec)
            source = "user"
        elif req.prompt:
            spec, source = agents.scenario_from_prompt(req.prompt)
        else:
            name = req.preset or "summer-heatwave"
            if name not in PRESETS:
                raise HTTPException(404, f"unknown preset {name}")
            spec, source = ScenarioSpec.from_dict(PRESETS[name].to_dict()), "preset"
    except ValueError as e:
        raise HTTPException(400, str(e))
    _last_spec = spec
    result = pipeline.run(spec, models(), max_incidents=max(1, min(req.max_incidents, 6)))
    result["scenario_source"] = source
    return result


@app.get("/api/telemetry.csv")
def telemetry(preset: str | None = None):
    """Download the full 5-minute synthetic telemetry (truth, measurements, labels)."""
    if preset:
        if preset not in PRESETS:
            raise HTTPException(404, f"unknown preset {preset}")
        spec = ScenarioSpec.from_dict(PRESETS[preset].to_dict())
    elif _last_spec is not None:
        spec = _last_spec
    else:
        spec = ScenarioSpec.from_dict(PRESETS["summer-heatwave"].to_dict())
    df = simulate(spec)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{spec.name}.csv"'})


@app.get("/api/ledger")
def ledger():
    return {"live": llm.live(), "by_tier": llm.LEDGER.summary(), "calls": len(llm.LEDGER.calls)}


@app.get("/healthz")
def healthz():
    return {"ok": True}
