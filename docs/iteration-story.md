# Iteration Story

This document explains the project as a step-by-step story.

## 1. Starting from the original repo

The original GitHub repository offered multiple challenge options.
I inspected the available paths before choosing one. The Crossing
challenge was the better choice for fast local iteration because the
data was already in the repo and the feedback loop was fast.

## 2. Measuring before changing anything

The first rule was: do not guess whether the baseline is good or bad.
Measure it.

So the first useful experiment was not new code. It was simply running
the baseline local grader and tests.

Measured local baseline:

- `Score: 0.8311`
- `intent_term: 0.856`
- `traj_term: 0.806`

That told us something important right away.

The trajectory side was weaker than the intent side.

## 3. Why trajectory came first

The baseline trajectory logic was a hand-written assumption that recent
motion continues forward at roughly the same speed.

That is a decent starting guess, but people do not move like simple
physics objects. They hesitate, speed up, slow down, and change their
mind.

So the most logical first move was not to rebuild everything. It was to
keep the simple baseline guess and learn a correction on top of it.

## 4. First successful jump: residual trajectory model

I added a trajectory model that predicts the error in the baseline
future-position guess.

That means the model does not start from zero. It starts from the simple
constant-velocity guess and then adjusts it.

Why this was a good idea:

- It preserved a simple, stable baseline.
- It focused learning on the part that was actually wrong.
- It matched the scoring setup, which cares about the predicted box
  centers.

This produced the first major improvement.

- E1 score: `0.7275`

## 5. Then we cleaned up the trajectory model

The first improved trajectory model helped a lot, but it was bigger and
slower than necessary.

So the next question was not “Can we make it fancier?”

It was:

"Can we get almost the same accuracy with a smaller and faster model?"

That is a more mature engineering question.

After benchmarking several tree counts and leaf sizes, I kept a smaller
ExtraTrees trajectory regressor.

- E2 score: `0.7224`

This model was lighter, faster, and still strong.

## 6. Once trajectory improved, intent became the next bottleneck

At that point the trajectory term had improved a lot, so the intent side
became the next obvious target.

The starter intent model used a smaller feature set. I expanded it using
the richer motion features that had already proven useful for trajectory.

This is another important pattern:

- do not invent features twice
- reuse useful signals across related tasks

## 7. Final promoted improvement: better intent model

I trained a stronger XGBoost classifier on the combined feature set.

That improved the intent side while keeping inference lightweight.

It also created an opportunity for another cleanup: reuse the same
trajectory feature computation at inference time instead of recomputing
similar information twice.

That helped both clarity and speed.

- E3 score: `0.7102`
- `intent_term: 0.831`
- `traj_term: 0.589`

## 8. What parallel subagents were used for

I used subagents as disposable helpers for focused questions.

Examples:

- compare challenge options before choosing
- summarize the Auto Research process from Karpathy's repo
- benchmark trajectory model families and sizes
- benchmark intent model options

This kept the main thread cleaner and sped up the search.

## 9. Why the git history matters

The recruiters explicitly said they care about the git log.

So the work was not compressed into one polished final commit. The
history shows a believable progression:

- baseline measurement
- first useful improvement
- speed and size tuning
- intent improvement
- docs and submission cleanup

That is exactly the signal they said they want.

## 10. What the current repo is meant to communicate

The repo should tell a reviewer three things quickly.

1. I chose a challenge deliberately.
2. I improved it through measured iteration.
3. I can explain the work clearly to both technical and non-technical
   readers.

That is why the repo now includes both technical writeups and plain
language docs.
