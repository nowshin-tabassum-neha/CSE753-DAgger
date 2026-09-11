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
See the querying strategies below for budgeted experiments.


## Expert-query accounting

New runs include four label-use columns in `metrics.csv` and print query totals:

- `initial_labels`: training character labels in the initial expert dataset.
- `queries_this_iteration`: expert requests during the current rollout (zero for both baselines).
- `cumulative_queries`: additional requests across completed rollouts in this run.
- `total_labels_used`: initial labels plus cumulative queries.

These count simulated expert requests, not unique characters. Repeated visits
count again. Both baselines share the initial labels; their totals should not
be added together. Test/validation labels used for scoring are excluded. When
beta selects an expert action, the already queried label is reused without a
second charge. Counters reset for every run, including every cross-validation fold.

Standard DAgger still queries every visited training state. For fold 9, iteration
1 uses 47,010 initial labels and zero additional queries; iteration 5 reaches
188,040 additional queries and 235,050 total labels. Budgets and selective querying are available through the options below. Existing saved results are left unchanged;
new columns appear in new runs. No retraining of the existing baselines is required.


## Label-budgeted querying (step 3)

`--query-budget` limits **additional** labels across the whole run, excluding
initial demonstrations. Omit it for unlimited queries; zero retains BC with no
additional labels. The budget resets for each seed/run and each outer test fold.

- `--query-strategy standard`: query every visited state until the budget ends.
- `--query-strategy uncertainty`: query when `1 - max(predict_proba(state))`
  is at least `--uncertainty-threshold` (default 0.5). This requires a classifier
  with probabilities, such as the current SGD log-loss model.
- `--query-strategy periodic`: query every `--query-period`th training state
  (default 10), starting at state k. The count continues across words and rollouts.

Selection occurs before reading the expert target. Only queried states are
added. All training words are still rolled out after the budget is exhausted;
if an iteration collects no new labels, its previous policy is reused and
reevaluated instead of refitted. Initial training labels and held-out scoring
labels are still loaded normally. Query restriction applies to new rollout labels.

Budgeted or selective runs require `--beta-decay 0` (the default). Nonzero mixing
remains available only for unlimited standard querying. Word order is unchanged
across strategies. Standard querying may spend its entire budget early; the
uncertainty threshold or periodic interval can leave budget unused. Compare
actual cumulative queries, not only the allowed budget. Thresholds are heuristic
probability scores, not calibrated guarantees; choose them on validation data.

Example matched five-iteration experiments (seed 0, 10,000 additional labels):

```powershell
python label_budget_dagger.py --alpha 0.00001 --iterations 5 --seed 0 --query-strategy standard --query-budget 10000 --output-dir results/budget_standard
python label_budget_dagger.py --alpha 0.00001 --iterations 5 --seed 0 --query-strategy uncertainty --uncertainty-threshold 0.5 --query-budget 10000 --output-dir results/budget_uncertainty
python label_budget_dagger.py --alpha 0.00001 --iterations 5 --seed 0 --query-strategy periodic --query-period 10 --query-budget 10000 --output-dir results/budget_periodic
```

These display a plot after each run; add `--no-plot` to disable it. Repeat the
same commands with seeds 1 and 2 when ready. Config files record strategy,
budget, period, and threshold. Metrics include remaining budget and visited
state counts as well as expert queries. Blank budget fields mean unlimited.
Validate the implementation with `python -m unittest test_baselines test_queries`.


## Separate study scripts

`dagger.py` runs BC, independent-character supervision and unrestricted DAgger.
`label_budget_dagger.py` adds budgets and standard/uncertainty/periodic querying.
`ablation_retention.py` runs the retention ablation; `tune_alpha.py` tunes the baseline.
Budget flags now belong only to `label_budget_dagger.py`. Examples:

```powershell
python label_budget_dagger.py --alpha 0.00001 --iterations 5 --seed 0 --query-strategy uncertainty --query-budget 10000 --output-dir results/budget_uncertainty
python dagger.py --alpha 0.00001 --iterations 5 --seed 0
```

Existing result files need no rerun. Both scripts save metrics and display plots
unless `--no-plot` is supplied. Baseline query counts still record all expert
labels; selective decisions and budget enforcement live in the budget script.
