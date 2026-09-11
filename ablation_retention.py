"""Ablation B: retain all, recent, or only the latest DAgger rollout batch."""

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

from dagger import DAgger


class RetentionDAgger(DAgger):
    """Always retain initial BC data; vary only the retained rollout batches."""

    def __init__(self, *args, retention="all", keep_last=3, **kwargs):
        super().__init__(*args, **kwargs)
        if retention not in ("all", "recent", "latest"):
            raise ValueError("retention must be all, recent, or latest.")
        if not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be a positive integer.")
        self.retention = retention
        self.keep_last = keep_last
        self._batch_sizes = deque()
        self._initial_size = None

    def run(self, N=11, **kwargs):
        # Isolate retention: full querying and learner-controlled rollouts only.
        if (kwargs.get("query_strategy", "standard") != "standard"
                or kwargs.get("query_budget") is not None
                or kwargs.get("beta_decay", 0) != 0):
            raise ValueError("Ablation B requires unlimited standard queries and beta_decay=0.")
        self._batch_sizes.clear()
        self._initial_size = None
        plot = kwargs.pop("plot", True)
        result = super().run(N=N, plot=False, **kwargs)
        self._save_study_metadata()
        if plot:
            self.plot_scores(result[0], supervised_score=self.last_structured_bc_score,
                             no_structure_score=self.last_no_structure_score,
                             test_fold=kwargs.get("test_fold", 9))
        return result

    def _fit_no_structure_policy(self, states, expert_actions):
        # The parent has created config.json by this point, before any training.
        path = self.last_run_dir / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config.update(
            study="ablation_B_retention",
            retention=self.retention,
            retained_rollout_limit=(None if self.retention == "all" else
                                    1 if self.retention == "latest" else self.keep_last),
            initial_data_policy="always retained",
            aggregation_rounds=config["iterations"] - 1,
            retention_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._study_config = config
        return super()._fit_no_structure_policy(states, expert_actions)

    def aggregate_dataset(self, policy, aggregated_data, test_fold=9, **kwargs):
        if self._initial_size is None:
            self._initial_size = len(aggregated_data[0])
        before = len(aggregated_data[0])
        super().aggregate_dataset(policy, aggregated_data, test_fold, **kwargs)
        self._batch_sizes.append(len(aggregated_data[0]) - before)
        limit = (None if self.retention == "all" else
                 1 if self.retention == "latest" else self.keep_last)
        while limit is not None and len(self._batch_sizes) > limit:
            discarded = self._batch_sizes.popleft()
            # Initial examples form the unchanged prefix; remove oldest rollout.
            for column in aggregated_data:
                del column[self._initial_size:self._initial_size + discarded]
        print(f"Retention={self.retention}: {len(self._batch_sizes)} rollout batches, "
              f"{len(aggregated_data[0])} retained examples")
        # Parent query counters are intentionally untouched by data removal.
        return aggregated_data

    def _save_study_metadata(self):
        # Parent rewrites its original config at completion; restore study fields.
        path = self.last_run_dir / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        for key in ("study", "retention", "retained_rollout_limit",
                    "initial_data_policy", "aggregation_rounds", "retention_source_sha256"):
            config[key] = self._study_config[key]
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention", choices=["all", "recent", "latest", "compare"], default="all")
    parser.add_argument("--keep-last", type=int, default=3,
                        help="Number of rollout batches retained in recent mode (default 3).")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Aggregation rounds AFTER BC; 10 means 11 code iterations.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--test-fold", type=int, choices=range(10), default=9)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablation_retention"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.rounds < 0 or args.keep_last < 1:
        parser.error("rounds must be nonnegative and keep-last must be positive.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique.")
    modes = ["all", "recent", "latest"] if args.retention == "compare" else [args.retention]
    for mode in modes:
        for seed in args.seeds:
            learner = RetentionDAgger(retention=mode, keep_last=args.keep_last,
                                      alpha=args.alpha, random_state=seed)
            folder = f"recent_{args.keep_last}" if mode == "recent" else mode
            print(f"\nAblation B: {mode}, seed {seed}, {args.rounds} aggregation rounds")
            learner.run(N=args.rounds + 1, test_fold=args.test_fold,
                        output_dir=args.output_dir / folder, plot=not args.no_plot)


if __name__ == "__main__":
    main()
