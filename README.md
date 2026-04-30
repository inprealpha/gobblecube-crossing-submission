# Gobblecube Crossing Submission

This repository contains my submission for the **Gobblecube AI Builder
Take-Home**, and I chose the **Crossing Challenge**.

The local starter baseline I measured was **0.8311** on the 5k Dev
sample, and the current best version in this repo scores **0.7102**.

## What This Submission Does

The task is to predict two things from about one second of recent
pedestrian motion:

1. Whether the pedestrian will start crossing within the next 2 seconds.
2. Where the pedestrian's bounding box will be at 0.5, 1.0, 1.5, and
   2.0 seconds into the future.

The final system keeps the starter repo's basic shape, but improves both
parts of the prediction:

- The **intent** model uses a richer set of motion features and an
  XGBoost classifier.
- The **trajectory** model learns corrections on top of the simple
  constant-velocity baseline using an ExtraTrees regressor.

## Current Result

- Challenge chosen: **Crossing Challenge**
- Baseline measured locally: **0.8311**
- Current best measured locally: **0.7102**
- Validation command: `python grade.py` from the repo root
- Tests: `python -m pytest tests/` and `python tests/smoke.py`

Lower is better for this challenge.

## Where To Start

- Main technical writeup: `crossing-challenge-starter/README.md`
- Plain-language explanation: `docs/plain-language-walkthrough.md`
- Iteration story: `docs/iteration-story.md`
- Repo map: `docs/repo-map.md`
- Submission checklist: `docs/submission-checklist.md`
- Agent/tooling note: `AGENTS.md`

## Repo Layout

- `crossing-challenge-starter/`
  This is the actual submission work.
- `docs/`
  Human-friendly explanations of what was done and why.

## Reproduce The Current Local Result

From the repo root:

```bash
crossing-challenge-starter/.venv/bin/python grade.py
```

Or, from `crossing-challenge-starter/`:

```bash
python baseline.py
python grade.py
python -m pytest tests/
python tests/smoke.py
```

Build the submission image from the repo root:

```bash
docker build -t my-crossing .
```

## Notes

- The local workspace did not have Docker installed, so I could not run a
  final `docker build` verification here. The root `Dockerfile` is the
  intended submission build path.
- The submission code itself does not call external services at inference
  time. It loads only local files, especially `model.pkl`.
