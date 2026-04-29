# Agent Instructions Used

This repo was worked on with Codex using these project-specific rules:

- Prefer current documentation via `ctx7` for library/framework/API/CLI
  questions. This challenge work did not require library-doc lookup; the
  implementation used local starter code and local experimentation.
- Use subagents for parallel exploration and context isolation. Subagents
  separately analyzed ETA, Crossing, the Auto Research workflow, trajectory
  experiments, and intent experiments.
- Keep the README updated as an experiment log and commit each meaningful
  change so the git history shows the actual iteration path.
- Do not use external APIs at inference time. The final `predict.py` loads
  local `model.pkl` only.
