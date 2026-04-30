# Repo Map

This document explains what each folder is for.

## Root folder

- `README.md`
  Submission landing page. This is the best place for a reviewer to
  start.
- `Dockerfile`, `predict.py`, `grade.py`, `requirements.txt`, `model.pkl`
  Root-level submission surface. These files let the grader build and
  run from the repository root.
- `AGENTS.md`
  Short note about the agent workflow and repo-specific instructions.
- `docs/`
  Human-friendly supporting documents.
- `crossing-challenge-starter/`
  The actual submission work for the chosen challenge.

## docs/

- `docs/plain-language-walkthrough.md`
  Simple explanation for a non-technical reader.
- `docs/iteration-story.md`
  Step-by-step explanation of how the work evolved.
- `docs/repo-map.md`
  This file.
- `docs/submission-checklist.md`
  Final review checklist before sending the repo.

## crossing-challenge-starter/

This is the important folder for the chosen submission.

- `crossing-challenge-starter/README.md`
  Main technical writeup, experiment log, current score, and validation
  notes.
- `crossing-challenge-starter/predict.py`
  The runtime entry point used by the grader. This is the file that
  serves predictions one request at a time.
- `crossing-challenge-starter/baseline.py`
  Training script that rebuilds the current `model.pkl`.
- `crossing-challenge-starter/model.pkl`
  Trained model weights used by `predict.py`.
- `crossing-challenge-starter/grade.py`
  Local grading harness.
- `crossing-challenge-starter/Dockerfile`
  Packaging setup for submission.
- `crossing-challenge-starter/requirements.txt`
  Python dependencies.

## crossing-challenge-starter/data/

- `train.parquet`
  Training windows used to fit the model.
- `dev.parquet`
  Development set used for local scoring.
- `schema.md`
  Explains what each data column means.
- `build_tracklets.py`, `build_windows.py`
  Original helper scripts from the starter repo. These were not the main
  focus of the submission work.

## crossing-challenge-starter/tests/

- `test_predict.py`
  Contract tests for prediction shape and safety.
- `smoke.py`
  Minimal smoke test for the prediction interface.
