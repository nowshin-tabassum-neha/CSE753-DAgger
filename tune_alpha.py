"""Select a shared SGD alpha on validation data; never evaluate the test fold."""
import argparse
import json
from pathlib import Path
from dagger import DAgger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=9, choices=range(10))
    parser.add_argument("--validation-fold", type=int, default=8, choices=range(10))
    parser.add_argument("--output-dir", default="results/tuning")
    args = parser.parse_args()
    if args.test_fold == args.validation_fold:
        parser.error("Validation and test folds must differ.")
    candidates = []
    for alpha in args.alphas:
        runs = []
        for seed in args.seeds:
            learner = DAgger(alpha=alpha, random_state=seed)
            scores, _ = learner.run(
                N=args.iterations, test_fold=args.validation_fold,
                excluded_folds=(args.test_fold,), evaluation_role="validation",
                plot=False, output_dir=args.output_dir,
            )
            runs.append({"seed": seed, "accuracy": float(scores[-1]),
                         "run_dir": str(learner.last_run_dir)})
        candidates.append({"alpha": alpha, "runs": runs,
                           "mean_validation_accuracy": sum(r["accuracy"] for r in runs)/len(runs)})
    best = max(candidates, key=lambda c: c["mean_validation_accuracy"])
    summary = {"selection_metric": "mean final-iteration free-running validation accuracy",
               "iterations": args.iterations, "test_fold": args.test_fold,
               "validation_fold": args.validation_fold, "selected_alpha": best["alpha"],
               "candidates": candidates}
    # Store beside the final candidate, avoiding overwriting previous tuning summaries.
    output = learner.last_run_dir / "tuning_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Selected alpha: {best['alpha']}; summary: {output}")


if __name__ == "__main__":
    main()
