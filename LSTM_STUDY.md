# LSTM memory study

`lstm_policy.py` implements a unidirectional LSTM. Each input contains 128
observed pixels and the 26-dimensional previous-action encoding. Hidden and
cell state carry information across characters within a word, and reset between
words. The output is 26 unnormalized class scores used with cross-entropy loss.

The study uses the existing OCR folds and corruption code. BC trains with the
previous correct letter; DAgger collects full sequences with the learner's own
previous predictions. Evaluation is always free-running. Padding is excluded
from the loss. Training batches contain whole words, shuffled only at word level.
Each aggregation round starts a fresh model and trains on all collected sequences
for the same number of epochs. Hidden states are recomputed, not stored in the data.

## Environment and verification

A separate `.venv_lstm` environment uses the existing scientific packages plus
CPU PyTorch. Use its interpreter explicitly:

```powershell
.\.venv_lstm\Scripts\python.exe -m unittest test_lstm
```

To recreate the environment:

```powershell
python -m venv --system-site-packages .venv_lstm
.\.venv_lstm\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Experiments

First check clean observations with BC only:

```powershell
.\.venv_lstm\Scripts\python.exe lstm_policy.py --rounds 0 --epochs 10 --seed 0 --no-plot
```

Compare BC with four aggregation rounds under 30% masking:

```powershell
.\.venv_lstm\Scripts\python.exe lstm_policy.py --rounds 4 --epochs 10 --mask-rate 0.3 --seed 0 --no-plot
```

Use `--noise-std 0.2` for Gaussian noise, or omit both corruption flags for clean
observations. Corruption is fixed per character; `--observation-seed` is separate
from the model seed. The default architecture is 64 hidden units, Adam learning
rate 0.001, batch size 64, and 2 CPU threads. These are untuned starting settings.
Use `--validation-fold 8 --test-fold 9` for tuning while excluding final test data.

Results under `results/lstm/<timestamp>/` include configuration, per-round
accuracy, exact-word accuracy, labels used, runtime, and an accuracy plot.
`bc.pt` and `dagger_latest.pt` contain model weights and basic architecture and
corruption metadata. They support later inference, not interrupted-training resume.
`--no-plot` saves the plot without opening a window. Round 0 is BC.

This study uses full querying and retains all data. A matched feedforward neural
baseline is still needed to isolate the effect of recurrent memory. Corruption
alone does not establish that memory helps. Compare clean/corrupted BC and DAgger
under fixed settings and multiple seeds before drawing conclusions.
