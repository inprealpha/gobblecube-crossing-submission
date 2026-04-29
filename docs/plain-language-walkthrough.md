# Plain-Language Walkthrough

This document explains the project in simple language.

## What problem were we solving?

Imagine a small delivery robot or self-driving vehicle approaching a
crosswalk. It sees a person near the road. It has to answer two
questions very quickly:

1. Is this person about to start crossing?
2. Where will this person be over the next 2 seconds?

If the system is too nervous, the vehicle hesitates all the time.
If the system is too careless, it becomes unsafe.

So the goal is not just to guess. The goal is to make a better decision
than a very simple starting system.

## What does the score mean?

The official local score combines two pieces:

- **Intent score**
  How well we predict whether the person will cross soon.
- **Trajectory score**
  How close our predicted future positions are to the real future
  positions.

Lower is better.

You can think of it like a golf score. A lower number means fewer
mistakes.

## What was the starting point?

The starter code already had a working but simple baseline.

- For crossing intent, it used a tree-based model with a small number of
  hand-made features.
- For future motion, it used a very simple rule: assume the pedestrian
  keeps moving at roughly the same recent speed and direction.

That second idea is useful as a first guess, but real people do not move
in such a neat way. They slow down, hesitate, turn, or change their
mind.

That is why the trajectory part was the biggest early weakness.

## Why did we choose this challenge?

I chose this challenge for a practical reason: it gave faster learning
loops.

- The training and dev data were already inside the repo.
- The grader was quick to run locally.
- The score was split into two parts, which made it easier to see where
  the model was weak.

That made it a good fit for an Auto Research style process: test one
idea, measure it, keep or reject it, then move to the next idea.

## What is “Auto Research” in plain English?

It is a disciplined trial-and-measure loop.

Instead of trying to jump straight to a perfect solution, we did this:

1. Start from a working baseline.
2. Measure it.
3. Change one important thing.
4. Measure again.
5. Keep the change only if the result is actually better.

This is similar to improving a recipe.

- First you taste the original.
- Then you change one ingredient.
- Then you taste again.
- If it got worse, you do not pretend it got better.

That was the spirit of this work.

## What did we improve first?

We improved the **trajectory** prediction first.

Why?

Because the baseline intent model was acceptable, but the simple future
motion rule was clearly weak. So the highest-value question was:

"Can we keep the simple baseline motion guess, but learn a correction on
top of it?"

That is exactly what we did.

## What does “learn a correction” mean?

Start with the simple guess:

- "The person will keep moving like they were moving in the last few
  frames."

Then add a second model that says:

- "That first guess is usually too far left here."
- "It usually overshoots at 2 seconds."
- "When the ego vehicle is moving this way, pedestrians tend to hesitate."

So instead of throwing away the baseline motion idea, we used it as a
base prediction and learned the error on top of it.

That is often a strong practical strategy.

## What models did we end up using?

The final system uses two tree-based models.

### Intent model

This is an **XGBoost classifier**.

In simple language, think of it as a large collection of small decision
rules. Each rule asks short yes/no style questions about the recent
motion and scene context, such as whether the person is moving upward in
the frame, how steady the motion is, and whether the ego vehicle is
moving.

### Trajectory model

This is an **ExtraTrees regressor**.

Again, in simple language, it is another ensemble of decision trees, but
this time it predicts numbers rather than a yes/no label. It learns how
to shift the simple future-position guess so the answer lands closer to
the real future position.

If you studied random forests or decision trees before, the family will
feel familiar. These are all close relatives.

## What kind of input information did we use?

We did not use raw video pixels. The challenge already gave a processed
history for each example.

The important inputs were things like:

- Recent pedestrian bounding boxes
- Recent movement direction and speed
- Recent acceleration or hesitation
- Size and shape changes in the box
- Ego vehicle speed and yaw history
- Simple scene labels such as time of day, weather, and location type

In other words, we used recent motion and context to make a forecast.

## What were the main stages of improvement?

### Stage E0: baseline

We first measured the untouched starter code.

- Baseline local score: `0.8311`

### Stage E1: smarter trajectory

We kept the intent model and improved future motion by learning a
trajectory correction on top of the constant-velocity baseline.

- Score improved to `0.7275`

This was the biggest jump, because it fixed the biggest weakness first.

### Stage E2: smaller and cleaner trajectory model

The first improved trajectory model worked, but it was heavier than it
needed to be. So we tuned it to keep most of the gain while making it
smaller and faster.

- Score improved further to `0.7224`

### Stage E3: richer intent model

Once trajectory was in much better shape, the next obvious place to work
was the intent prediction.

We added a stronger XGBoost model using a richer feature set built from
the same recent motion signals.

- Final current score: `0.7102`

## Why does this make logical sense?

Because the work followed a clear order.

1. Find the weakest part of the system.
2. Improve that part first.
3. Once it is no longer the main weakness, move to the next bottleneck.

That is exactly what happened.

- First bottleneck: future position prediction
- Second bottleneck: crossing intent calibration

This is a very general way to reason about complex problems, even
outside engineering.

## What did not help much?

Some ideas looked reasonable but were not strong enough to keep.

- Simple global corrections to the baseline motion guess did not help.
- Some alternative model families were not better after measurement.
- Some larger models were not worth the extra runtime or size.

That is important. A serious process includes dead ends, not just wins.

## Where did the work happen in the repo?

- `crossing-challenge-starter/predict.py`
  The final inference logic.
- `crossing-challenge-starter/baseline.py`
  The training script that rebuilds `model.pkl`.
- `crossing-challenge-starter/model.pkl`
  The trained model file used at inference time.
- `crossing-challenge-starter/README.md`
  Technical writeup and experiment log.
- `docs/`
  Human-friendly explanation documents.

## What should a non-technical reviewer take away?

The important point is not just that the score improved.

The important point is that the work was done in a sensible order:

- understand the baseline
- identify the biggest weakness
- improve one thing at a time
- measure honestly
- keep the history visible

That is the real story of the project.
