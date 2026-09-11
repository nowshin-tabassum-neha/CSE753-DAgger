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
