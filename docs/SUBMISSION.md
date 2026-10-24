# Devpost submission kit

Everything below maps to a field on the Nebius x NVIDIA Global AI Hackathon submission form. Fill the bracketed parts before submitting.

## Track

Physical AI. This submission has no physical hardware, so the video shows the key application modules in action, as the rules allow.

## Project description

**ReefSynth: synthetic coral-farm data and a Nemotron agent crew that keeps corals alive.**

**The problem.** Commercial coral farms and reef-restoration nurseries lose stock to failures that build slowly: a heater relay that sticks on, a doser that runs dry, a pH probe that drifts, a heatwave while the chiller is down. AI monitoring could catch these early, but there is no labeled failure data, because no farm can kill its own corals to produce it, and real logs rarely record the true cause.

**What we built.**

1. A process-based digital twin of a coral farm (heat balance, evaporation and top-off, calcification consuming alkalinity and calcium, a dosing controller, nutrients, pH and oxygen, and coral physiology with degree heating hours and bleaching) plus a realistic sensor layer. It generates fully labeled telemetry for 16 equipment, environmental and sensor faults.
2. Detectors trained on that synthetic data and tested on unseen farms: an anomaly detector, a 17-class fault classifier (accuracy 0.90) and a 24-hour coral-risk model (AUC 0.91).
3. A crew of NVIDIA Nemotron agents on Nebius Token Factory:
   - Nemotron 3 Super writes scenarios from plain English and reviews synthetic data as an LLM judge.
   - Nemotron 3 Nano writes cheap daily shift notes.
   - Nemotron 3 Ultra diagnoses incidents, including the hard call of whether the water is really changing or a probe is lying, and writes the owner's report.
   - Nemotron 3 Super proposes actions, which a deterministic safety guard clamps or blocks before they reach a human for approval.
4. Every run is scored against the simulator's ground truth: which faults were caught, how fast, and whether the diagnosis was right.

**Why it matters.** It gives coral farms a monitoring system trained on thousands of simulated failures they never had to suffer, and it is honest about what it can and cannot see.

## How NVIDIA models and Nebius were used

- NVIDIA Nemotron 3 Nano, Super and Ultra, all through the Nebius Token Factory OpenAI-compatible API, each routed to the job that fits its size and cost (see the README table).
- Cheap scikit-learn detectors score every hour; Token Factory is called only for flagged incidents, which keeps a 14-day farm to a handful of Ultra calls.
- Nebius Serverless Jobs run batch synthesis, training and evaluation.
- Nebius Serverless Endpoints host the dashboard and API.
- [Optional] LangSmith tracing of every agent call.

## What's new during the submission period

The entire project was created during the submission period (first commit October 16, 2026).

## Links

- Demo: [Nebius Serverless Endpoint URL]
- Code: https://github.com/divyeshparekh9496/reef-synth33
- Video: [YouTube URL]

## Demo video script (under 3 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00-0:20 | Coral farm photo or diagram, then the dashboard | Coral farms lose stock to slow failures. AI could catch them, but there is no labeled failure data. We generate it. |
| 0:20-0:50 | Type a scenario prompt, click run, show the spec | Nemotron Super turns plain English into a scenario for our digital twin, which simulates two weeks of chemistry and coral physiology in under a second. |
| 0:50-1:30 | Charts with the shaded fault windows, then the incident card | The detectors flag the problem. Nemotron Ultra explains the root cause with evidence and the 24-hour risk to the corals. |
| 1:30-2:00 | `lying-probe` preset: sensor-fault badge and the recalibration action | Here the pH probe is lying while the water is fine. The agent calls it a sensor fault and asks for a recalibration; the safety guard would block any chemistry change based on that probe. |
| 2:00-2:30 | Ground-truth scoring panel, then the Nemotron calls table | Every run is scored against ground truth. Nano handles daily notes, Super handles structure, Ultra only runs on incidents. |
| 2:30-2:55 | Terminal: `reefsynth eval`, then the Serverless Job logs | The same container runs as a Nebius Serverless Job to synthesize hundreds of farms and score the crew on farms it has never seen. |

Do not use copyrighted music or third-party footage. Record your own screen and voice.

## Feedback on Nebius Token Factory, AI Cloud and NVIDIA tools

Write this as you build. Prompts to answer:

- Token Factory: onboarding time, finding current model IDs, latency and reliability per tier, rate limits hit, error messages, pricing clarity, anything you wished the API returned.
- Nemotron 3 Nano, Super, Ultra: JSON-format reliability, reasoning-trace handling (`<think>` blocks), quality differences between tiers on the diagnosis task, cost versus quality.
- Serverless Endpoints and Jobs: CLI experience, image pull from GHCR, passing secrets, cold start time, logs.
- What would have made the build faster.
