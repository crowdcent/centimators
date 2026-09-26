# Protocol: Jev + DSPyMator follow-up experiments (pre-registered 2026-09-26)

Written and committed before any of these runs. Results that contradict a hypothesis are reported as they are.

## Fixed across all experiments

- Tasks, 500-row test sets, 4,000-row training pools, stratified labeled subsets `tasks.subsample(train, n, seed)` and seeds 0/1/2: unchanged from the main ablation (`results/abl/`).
- The test set is used only to score a finished model, never to choose a setting.
- Primary metric: macro-F1. Secondary: log loss, ECE. Comparisons use a 95% paired bootstrap over test rows (2,000 resamples), averaging over seeds.
- Every configuration that runs is reported, including failures.

## E1: does GEPA keep improving past 256 labels?

- Jev + GEPA, the same configuration as the ablation (Brier metric, GPT-6 Sol reflection, default minibatch), at n = 1,024 and 4,000, 3 seeds each (at 4,000 the seed changes the GEPA search and validation split, not the labeled set).
- Validation split: stratified, min(n/2, 256) rows; the rest drive reflection. Metric-call budget: 300 + 2·min(n, 1,024). At n ≤ 256 these equal the ablation's settings. The cap keeps compute bounded, and it is a limit of this test.
- H1: F1 at 1,024 is at least F1 at 256 on the tasks that were still rising (financial tweets, satire). Plateau on AG News and SST-2.

## E2: tuning GEPA's own settings with scikit-learn

- `GridSearchCV(DSPyMator(optimizer=DSPyOptimizer(dspy.GEPA, ...), validation_data=0.5), cv=StratifiedKFold(3), scoring="f1_macro")` on the n = 256 labeled set, seeds 0/1/2.
- Grid: `optimizer__reflection_minibatch_size` ∈ {3, 8} × `optimizer__max_metric_calls` ∈ {500, 1,500}.
- The selected setting is refit on all 256 labels (`refit=True`) and scored once on the test set.
- H2: the CV-selected setting beats the fixed default (ablation `jev_gepa` at 256) on mean F1. No prediction on size of effect.
- Amendment (2026-09-26 15:40 UTC, before any E2 result existed): E2 runs on seed 0 only. Each grid is 13 GEPA fits and the runs are limited to the VM's 2 CPUs; three seeds would take ~6 hours. The comparison is against the default's seed-0 cell, with the paired bootstrap over test rows as the only uncertainty estimate.

## E3: stacking Jev with embeddings (scikit-learn composition)

- Features: bge-base embeddings, and log of Jev zero-shot class probabilities. Jev has seen no labels, so its training-row probabilities carry no label leakage.
- Models, all `LogisticRegressionCV` (C chosen by CV inside the labeled set) at n ∈ {16, 64, 256, 1,024, 4,000}:
  - `jev_lr`: Jev probabilities only (a learned recalibration of Jev)
  - `stack_lr`: embeddings plus Jev probabilities
- Jev probabilities are computed on the VM (API calls); the fits run on CrowdCent Cloud.
- H3: `stack_lr` beats both `embed_lr` and Jev + GEPA at n ≥ 1,024. H3b: `jev_lr` lowers log loss vs raw Jev at every n ≥ 64.
