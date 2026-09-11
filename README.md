# DAgger on the Stanford OCR dataset

This project implements the OCR sequence-labelling experiment from Ross,
Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning* (AISTATS 2011).

Each word is a trajectory. At character position `t`, the policy observes the
128 image pixels and a 26-dimensional one-hot encoding of its previously
executed letter prediction. DAgger labels every state visited during a complete
left-to-right rollout with the ground-truth letter, aggregates those examples,
and retrains a linear classifier using scikit-learn's `SGDClassifier` with
`loss="log_loss"`, `alpha=1e-4`, `average=True`, `max_iter=1000`, and
`tol=1e-3`. The classifier uses the DAgger random seed (default 0).
Both supervised baselines use the same classifier. Each DAgger iteration
fits a fresh classifier on the full aggregated dataset. These are starting
parameters, not tuned settings; runtime and accuracy should be measured
against the previous SVC and LinearSVC implementations.

The generated plot compares DAgger with structured behavior cloning (the
iteration-1 policy trained on expert trajectories) and an independent
no-structure supervised model that uses only the 128 image pixels. Ten-fold
runs show mean accuracy with 95% confidence bands.

Run one held-out fold with the paper's 20 iterations and
`beta_i = I(i=1)` schedule:

```powershell
python dagger.py --test-fold 9
```

Run the paper's full ten-fold protocol:

```powershell
python dagger.py --all-folds
```

For a quick non-plotting check:

```powershell
python dagger.py --iterations 1 --test-fold 9 --no-plot
```

The full 20-iteration, ten-fold experiment can still be computationally
expensive as the aggregated training dataset grows. The scores in
`update_results_graph.py` are historical results from the original SVC run;
rerun training to obtain results for `SGDClassifier`.


## Reproducible baseline experiments (step 1)

Every run now writes a unique timestamped folder under `results/`:

- `config.json`: seed, exact classifier parameters, training/evaluation folds,
  dataset and source hashes, library versions, completion status, and total time.
- `metrics.csv`: no-structure baseline, structured BC (iteration 1), and each
  later DAgger iteration, including accuracy, imitation error, dataset size,
  training, rollout, evaluation, and elapsed seconds. Accuracy is a fraction.

Rows are saved as iterations finish, so completed measurements survive an
interruption. A non-complete config indicates an unfinished run. Training time
includes feature stacking; rollout time includes aggregation. Elapsed time
includes setup and logging but excludes the interactive plot. Models are not saved.
All-fold runs save a separate folder for every held-out fold. The BC row is
also DAgger's iteration-1 result; it is not a second independent fit.

Run matched baselines and five DAgger iterations:

```powershell
python dagger.py --iterations 5 --test-fold 9 --seed 0 --alpha 0.0001 --no-plot
```

Tune alpha without using final test fold 9 (train on folds 0-7, validate on 8):

```powershell
python tune_alpha.py --iterations 5 --seeds 0 1 2
```

The tuning script compares `1e-5`, `1e-4`, and `1e-3`, selecting by mean
final-iteration free-running validation accuracy. Its `tuning_summary.json`
is saved inside the final candidate run folder. Use the selected alpha for
all three methods; this is a shared DAgger-selected setting, not independent
optimization of each baseline. Keep the iteration count fixed across candidates.
For an individual validation run:

```powershell
python dagger.py --iterations 5 --validation-fold 8 --test-fold 9 --alpha 0.0001 --no-plot
```

After selecting settings, omit `--validation-fold` to retrain on folds 0-8
and evaluate on fold 9. Repeat with `--seed 0`, `--seed 1`, and `--seed 2`.
Do not choose settings from test scores. For unbiased tuning during ten-fold
cross-validation, repeat validation selection within each outer training split;
`--all-folds` itself evaluates a fixed alpha and does not do nested tuning.
Label budgeting and selective querying are not implemented in this step.
