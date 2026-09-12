"""Ablation D: free-running character accuracy at each position in a word."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dagger import DAgger


def evaluate_positions(learner, policy, test_fold=9):
    """Evaluate complete held-out words, using only previous predictions as context."""
    correct, counts = [], []
    for images, labels, fold in zip(learner.words, learner.sequences, learner.words_fold):
        if fold != test_fold:
            continue
        previous = None
        for position, (image, target) in enumerate(zip(images, labels)):
            prediction = int(policy.predict(learner.make_state(image, previous))[0])
            if position == len(counts):
                counts.append(0)
                correct.append(0)
            counts[position] += 1
            correct[position] += int(prediction == target)
            previous = prediction
    if not counts:
        raise ValueError(f"Test fold {test_fold} contains no characters.")
    rows = [dict(position=i+1, correct=c, count=n, accuracy=c/n)
            for i, (c, n) in enumerate(zip(correct, counts))]
    return rows, sum(correct)/sum(counts)


class PositionDAgger(DAgger):
    def run(self, N=11, **kwargs):
        self.position_records = []
        self._evaluation_iteration = 0
        return super().run(N=N, **kwargs)

    def _experiment_config(self):
        import hashlib
        return {**super()._experiment_config(), "study": "ablation_D_position",
                "position_indexing": "1-based; only words reaching a position contribute",
                "position_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def evaluate_policy(self, policy, test_fold=9):
        rows, accuracy = evaluate_positions(self, policy, test_fold)
        self._evaluation_iteration += 1
        iteration = self._evaluation_iteration
        records = [dict(iteration=iteration, aggregation_round=iteration-1,
                        method="structured_bc" if iteration == 1 else "dagger",
                        seed=self.random_state, test_fold=test_fold, **row) for row in rows]
        path = self.last_run_dir / "position_metrics.csv"
        with path.open("w" if iteration == 1 else "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            if iteration == 1:
                writer.writeheader()
            writer.writerows(records)
        self.position_records.extend(records)
        return accuracy


def plot_positions(records, output_dir, show=True):
    last = max(r["iteration"] for r in records)
    fig, (ax, support) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    selected = [(1, "BC", "#64748b")]
    if last > 1:
        selected.append((last, f"DAgger ({last-1} aggregation rounds)", "#2563eb"))
    for iteration, label, color in selected:
        rows = [r for r in records if r["iteration"] == iteration]
        x = [r["position"] for r in rows]
        ax.plot(x, [100*r["accuracy"] for r in rows], "o-", label=label, color=color)
    bc = [r for r in records if r["iteration"] == 1]
    support.bar([r["position"] for r in bc], [r["count"] for r in bc], color="#94a3b8")
    ax.set_ylabel("Character accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Ablation D: free-running accuracy by position\n"
                 f"Test fold {records[0]['test_fold']}, seed {records[0]['seed']}")
    ax.legend()
    support.set_ylabel("Test words")
    support.set_xlabel("Character position (1-based)")
    support.set_xticks([r["position"] for r in bc])
    for axis in (ax, support):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir)/f"accuracy_by_position.{extension}", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10, help="Aggregation rounds after initial BC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--test-fold", type=int, choices=range(10), default=9)
    parser.add_argument("--output-dir", default="results/ablation_position")
    parser.add_argument("--no-plot", action="store_true", help="Save plots without opening a window.")
    args = parser.parse_args()
    if args.rounds < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Rounds must be nonnegative and seeds must be unique.")
    for seed in args.seeds:
        learner = PositionDAgger(alpha=args.alpha, random_state=seed)
        learner.run(N=args.rounds+1, test_fold=args.test_fold, plot=False,
                    output_dir=args.output_dir)
        plot_positions(learner.position_records, learner.last_run_dir, show=not args.no_plot)
        print(f"Position metrics and plots saved to {learner.last_run_dir}")


if __name__ == "__main__":
    main()
