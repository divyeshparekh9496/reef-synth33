#!/usr/bin/env bash
# Batch job for Nebius Serverless Jobs:
#   1. synthesize N labeled coral farms in parallel
#   2. train the detectors on them
#   3. score the full agent crew (Nemotron on Token Factory) on fresh unseen farms
#
# Env:
#   N_FARMS        number of farms to synthesize (default 400)
#   DAYS           days per farm (default 14)
#   EVAL_FARMS     farms for the agent evaluation (default 20)
#   OUT            output dir (default /data/reefsynth; mount an Object Storage bucket here)
#   NEBIUS_API_KEY Token Factory key (agents fall back to rules without it)
set -euo pipefail

N_FARMS="${N_FARMS:-400}"
DAYS="${DAYS:-14}"
EVAL_FARMS="${EVAL_FARMS:-20}"
OUT="${OUT:-/data/reefsynth}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$OUT/$RUN_ID"
mkdir -p "$DEST"

echo "[reefsynth] generating $N_FARMS farms x $DAYS days -> $DEST/data"
reefsynth generate --out "$DEST/data" --n "$N_FARMS" --days "$DAYS"

echo "[reefsynth] training detectors"
reefsynth train --data "$DEST/data" --out "$DEST/models"

echo "[reefsynth] evaluating agent crew on $EVAL_FARMS unseen farms"
reefsynth eval --n "$EVAL_FARMS" --models "$DEST/models/reef_models.joblib" --out "$DEST/eval.json"

echo "[reefsynth] done: $DEST"
ls -la "$DEST" "$DEST/data" | head -40
