"""Word-memory LSTM policy and full-query BC/DAgger study on OCR observations."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from partial_observation import PartialObservationDAgger


class LSTMPolicy(nn.Module):
    """154 inputs -> recurrent word memory -> 26 letter logits.

    State is explicit: pass None at every new word/batch. The model does not
    retain hidden state between calls unless the caller supplies it.
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(154, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 26)

    def forward(self, features, hidden=None):
        output, hidden = self.lstm(features, hidden)
        return self.head(output), hidden


def expert_episodes(words, labels):
    episodes = []
    for images, targets in zip(words, labels):
        x = np.zeros((len(images), 154), dtype=np.float32)
        x[:, :128] = images
        for t in range(1, len(images)):
            x[t, 128 + targets[t-1]] = 1
        episodes.append((torch.from_numpy(x), torch.tensor(targets, dtype=torch.long)))
    return episodes


def fit_policy(episodes, *, hidden_size=64, epochs=10, batch_size=64,
               learning_rate=1e-3, seed=0, device="cpu"):
    """Fit a fresh model on full word sequences, masking padding in the loss."""
    torch.manual_seed(seed)
    model = LSTMPolicy(hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(episodes), generator=generator).tolist()
        total_loss, total_labels = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [episodes[i] for i in order[start:start+batch_size]]
            x = pad_sequence([e[0] for e in batch], batch_first=True).to(device)
            y = pad_sequence([e[1] for e in batch], batch_first=True, padding_value=-100).to(device)
            optimizer.zero_grad()
            logits, _ = model(x)  # Fresh hidden state per word; no carry across batches.
            loss = nn.functional.cross_entropy(logits.reshape(-1,26), y.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((y != -100).sum())
            total_loss += loss.item()*count
            total_labels += count
        print(f"  epoch {epoch+1}/{epochs}: loss={total_loss/total_labels:.4f}", flush=True)
    return model


@torch.no_grad()
def rollout(model, words, labels, *, batch_size=64, device="cpu", collect=False):
    """Free-running decoding, batched across words. Ground truth never drives actions.

    Collected features store executed previous predictions. Hidden states are
    recomputed from complete sequences during fitting, never stored as targets.
    """
    model.eval()
    episodes = []
    correct = total = exact = 0
    for start in range(0, len(words), batch_size):
        images = words[start:start+batch_size]
        targets = labels[start:start+batch_size]
        lengths = [len(word) for word in images]
        features = np.zeros((len(images), max(lengths),154), dtype=np.float32)
        for i, word in enumerate(images):
            features[i,:len(word),:128] = word
        pixels = torch.from_numpy(features).to(device)
        hidden = None
        previous = None
        predictions = []
        for t in range(max(lengths)):
            state = pixels[:,t:t+1,:].clone()
            if previous is not None:
                state[:,0,128:] = nn.functional.one_hot(previous,26).float()
            logits, hidden = model(state,hidden)
            previous = logits[:,0].argmax(-1)
            predictions.append(previous.cpu().numpy())
            if collect:
                features[:,t,:] = state[:,0].cpu().numpy()
        predicted = np.stack(predictions,axis=1)
        for i, target in enumerate(targets):
            matches = predicted[i,:lengths[i]] == np.asarray(target)
            correct += int(matches.sum())
            total += lengths[i]
            exact += int(matches.all())
            if collect:
                episodes.append((torch.from_numpy(features[i,:lengths[i]].copy()),
                                 torch.tensor(target,dtype=torch.long)))
    if not total:
        raise ValueError("No characters to evaluate.")
    return episodes, {"accuracy":correct/total, "imitation_error":1-correct/total,
                      "exact_word_accuracy":exact/len(words), "characters":total,
                      "words":len(words)}


def run_study(args):
    if args.rounds < 0 or min(args.epochs,args.batch_size,args.hidden_size,args.threads) < 1:
        raise ValueError("Rounds must be nonnegative; epochs, batch size, hidden size and threads positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if args.validation_fold == args.test_fold:
        raise ValueError("Validation and test folds must differ.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    data = PartialObservationDAgger(args.ocr_path, mask_rate=args.mask_rate,
                                   noise_std=args.noise_std, observation_seed=args.observation_seed)
    data.process_ocr()
    evaluation_fold = args.test_fold if args.validation_fold is None else args.validation_fold
    excluded = {args.test_fold, evaluation_fold}
    train_ids = [i for i,f in enumerate(data.words_fold) if f not in excluded]
    test_ids = [i for i,f in enumerate(data.words_fold) if f == evaluation_fold]
    if not train_ids or not test_ids: raise ValueError("Empty training or evaluation split.")
    train_words = [data.words[i] for i in train_ids]
    train_labels = [data.sequences[i] for i in train_ids]
    test_words = [data.words[i] for i in test_ids]
    test_labels = [data.sequences[i] for i in test_ids]
    episodes = expert_episodes(train_words,train_labels)
    initial_labels = sum(len(y) for _,y in episodes)
    folder = Path(args.output_dir)/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    config = {**vars(args), "device_used":device, "torch_version":torch.__version__,
              "status":"running", "train_folds":sorted(set(data.words_fold[i] for i in train_ids)),
              "evaluation_fold":evaluation_fold,
              "evaluation_role":"test" if args.validation_fold is None else "validation",
              "initial_labels":initial_labels, "observation_sha256":data.observation_sha256,
              "dataset_sha256":hashlib.sha256(Path(data.ocr_path).read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training":"fresh initialization per round, full sequences, all data retained",
              "querying":"unrestricted; learner-only rollout after initial BC",
              "corruption":"fixed per character; Gaussian noise, clip, then zero masking"}
    def save_config():
        (folder/"config.json").write_text(json.dumps(config,indent=2,default=str))
    save_config()
    print(f"Results: {folder}; device: {device}",flush=True)
    records=[]
    for round_index in range(args.rounds+1):
        rollout_seconds=0.0
        queries=0
        if round_index:
            t=perf_counter()
            new,_=rollout(model,train_words,train_labels,batch_size=args.batch_size,device=device,collect=True)
            episodes.extend(new)
            queries=sum(len(y) for _,y in new)
            rollout_seconds=perf_counter()-t
        print(f"Aggregation round {round_index}/{args.rounds}",flush=True)
        t=perf_counter()
        model=fit_policy(episodes,hidden_size=args.hidden_size,epochs=args.epochs,
                         batch_size=args.batch_size,learning_rate=args.learning_rate,
                         seed=args.seed,device=device)
        training_seconds=perf_counter()-t
        t=perf_counter()
        _,metrics=rollout(model,test_words,test_labels,batch_size=args.batch_size,device=device)
        row=dict(round=round_index,method="bc" if round_index==0 else "dagger",**metrics,
                 dataset_size=sum(len(y) for _,y in episodes),initial_labels=initial_labels,
                 queries_this_round=queries,cumulative_queries=round_index*initial_labels,
                 training_seconds=training_seconds,rollout_seconds=rollout_seconds,
                 evaluation_seconds=perf_counter()-t,elapsed_seconds=perf_counter()-started)
        with (folder/"metrics.csv").open("a",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(row))
            if not records: writer.writeheader()
            writer.writerow(row)
        records.append(row)
        checkpoint={"model_state":{k:v.cpu() for k,v in model.state_dict().items()},
                    "hidden_size":args.hidden_size,"round":round_index,
                    "mask_rate":args.mask_rate,"noise_std":args.noise_std,
                    "observation_seed":args.observation_seed}
        torch.save(checkpoint,folder/("bc.pt" if round_index==0 else "dagger_latest.pt"))
        print(f"Accuracy={metrics['accuracy']:.4f}; exact words={metrics['exact_word_accuracy']:.4f}",flush=True)
    config.update(status="complete",total_seconds=perf_counter()-started)
    save_config()
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.plot([r['round'] for r in records],[100*r['accuracy'] for r in records],'o-',label='LSTM DAgger')
    ax.axhline(100*records[0]['accuracy'],ls='--',label='LSTM BC')
    ax.set(xlabel='Aggregation round (0 = BC)',ylabel='Character accuracy (%)',
           title=f'LSTM OCR: mask={args.mask_rate}, noise={args.noise_std}, seed={args.seed}')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(folder/'accuracy.png',dpi=200)
    if not args.no_plot: plt.show()
    plt.close(fig)
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--hidden-size',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--learning-rate',type=float,default=1e-3)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--test-fold',type=int,default=9,choices=range(10))
    parser.add_argument('--validation-fold',type=int,choices=range(10))
    parser.add_argument('--mask-rate',type=float,default=0)
    parser.add_argument('--noise-std',type=float,default=0)
    parser.add_argument('--observation-seed',type=int,default=0)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--ocr-path',default='letter.data')
    parser.add_argument('--output-dir',default='results/lstm')
    parser.add_argument('--no-plot',action='store_true')
    run_study(parser.parse_args())


if __name__=='__main__': main()
