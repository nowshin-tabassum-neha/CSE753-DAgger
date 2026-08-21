# DAgger on the Stanford OCR dataset

This project implements the OCR sequence-labelling experiment from Ross,
Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning* (AISTATS 2011).

Each word is a trajectory. At character position `t`, the policy observes the
128 image pixels and a 26-dimensional one-hot encoding of its previously
executed letter prediction. DAgger labels every state visited during a complete
left-to-right rollout with the ground-truth letter, aggregates those examples,
and retrains a linear SVM.

The generated plot compares DAgger with its structured behavior-cloning
baseline (the iteration-1 policy trained on expert trajectories). Ten-fold runs
show mean accuracy with 95% confidence bands.

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

The linear kernel SVM and full 20-iteration, ten-fold experiment are
computationally expensive.
