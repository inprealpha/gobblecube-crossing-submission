# Submission Checklist

Use this before sending the repo URL to Gobblecube.

## Score and validation

- Run `python grade.py` from the repo root and confirm the score is still
  around `0.7102`.
- Run `python baseline.py` inside `crossing-challenge-starter/` only if
  you want to rebuild `model.pkl` from scratch.
- If you rebuild inside `crossing-challenge-starter/`, copy the refreshed
  `crossing-challenge-starter/model.pkl` to the root `model.pkl` before
  building Docker.
- Run `python -m pytest tests/`.
- Run `python tests/smoke.py`.

## Packaging

- Run `docker build -t my-crossing .` from the repo root.
- Run the container on the grader path if Docker is available.
- Confirm the image size is still within the challenge limit.

## Repo presentation

- Make sure the root `README.md` clearly says this is the **Crossing**
  submission.
- Make sure `crossing-challenge-starter/README.md` still shows the
  current best score and experiment log.
- Keep `AGENTS.md` and the supporting docs in the repo because the
  challenge asked for the agent/markdown context.

## Final manual sanity check

- `predict.py` should not call external services.
- `model.pkl` should exist and load locally.
- The repo should be public.
- The GitHub repo URL should open cleanly.
- The commit history should still show the actual iteration process.

## Suggested order right before submission

1. Final local tests.
2. Final Docker check on a machine with Docker installed.
3. Push latest commits.
4. Open the GitHub repo in a browser and read it like a reviewer.
5. Send the repo URL.
