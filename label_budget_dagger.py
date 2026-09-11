"""Label-budgeted DAgger, using the baseline data loading, training and evaluation."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from sklearn.base import ClassifierMixin
from dagger import DAgger, build_parser


class LabelBudgetDAgger(DAgger):
    def run(self, N=20, *, query_strategy="standard", query_budget=None,
            query_period=10, uncertainty_threshold=0.5, **kwargs):
        self._validate_queries(query_strategy, query_budget, query_period,
                               uncertainty_threshold, kwargs.get("beta_decay", 0))
        if query_strategy == "uncertainty" and not callable(getattr(self.policy_factory(), "predict_proba", None)):
            raise ValueError("Uncertainty querying requires predict_proba.")
        self._query_options = dict(query_strategy=query_strategy, query_budget=query_budget,
                                   query_period=query_period, uncertainty_threshold=uncertainty_threshold)
        return super().run(N=N, **kwargs)

    def _experiment_config(self):
        return {**self._query_options,
                "query_order": "dataset order; periodic index continues across rollouts",
                "label_budget_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _experiment_metrics(self, cumulative_queries):
        budget = self._query_options["query_budget"]
        return {"query_strategy": self._query_options["query_strategy"], "query_budget": budget,
                "remaining_queries": None if budget is None else budget-cumulative_queries}

    def _collect_rollout(self, policy, data, test_fold, beta, excluded_folds,
                         cumulative_queries, visited_states):
        options = dict(self._query_options)
        if options["query_budget"] is not None:
            options["query_budget"] -= cumulative_queries
        return self.aggregate_dataset(policy, data, test_fold, beta=beta,
                                      excluded_folds=excluded_folds,
                                      visited_offset=visited_states, **options)

    @staticmethod
    def _validate_queries(strategy, budget, period, threshold, beta):
        if strategy not in ("standard", "uncertainty", "periodic"):
            raise ValueError("Unknown query strategy.")
        if budget is not None and (not isinstance(budget, int) or budget < 0):
            raise ValueError("Query budget must be a nonnegative integer or None.")
        if not isinstance(period, int) or period < 1:
            raise ValueError("Query period must be a positive integer.")
        if not 0 <= threshold <= 1:
            raise ValueError("Uncertainty threshold must be in [0, 1].")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        if beta != 0 and (strategy != "standard" or budget is not None):
            raise ValueError("Budgeted/selective querying requires beta_decay=0 to avoid uncounted expert actions.")

    def aggregate_dataset(
        self, policy: ClassifierMixin, aggregated_data: list[list],
        test_fold: int = 9, *, beta: float = 0.0,
        excluded_folds: tuple[int, ...] = (),
        query_strategy: str = "standard", query_budget: int | None = None,
        query_period: int = 10, uncertainty_threshold: float = 0.5,
        visited_offset: int = 0,
    ) -> list[list]:
        """Collect selected expert labels; query_budget is the remaining allowance.

        Periodic queries select every kth training state across rollouts (1-based).
        Decisions use learner observations/probabilities before accessing labels.
        All words are rolled out even when no queries remain.
        """
        self._validate_queries(query_strategy, query_budget, query_period,
                               uncertainty_threshold, beta)
        if query_strategy == "uncertainty" and not callable(getattr(policy, "predict_proba", None)):
            raise ValueError("Uncertainty querying requires predict_proba (SGD loss='log_loss').")
        self.last_rollout_queries = 0
        self.last_rollout_visited = 0
        new_states, new_actions = [], []
        for word_index, images in enumerate(self.words):
            if self.words_fold[word_index] in (test_fold, *excluded_folds):
                continue
            previous_action = None
            for position, image in enumerate(images):
                state = self.make_state(image, previous_action)
                self.last_rollout_visited += 1
                available = query_budget is None or self.last_rollout_queries < query_budget
                query = False
                if available:
                    if query_strategy == "standard":
                        query = True
                    elif query_strategy == "periodic":
                        query = (visited_offset + self.last_rollout_visited) % query_period == 0
                    else:
                        probabilities = np.asarray(policy.predict_proba(state)[0], dtype=float)
                        if (not np.all(np.isfinite(probabilities)) or
                                np.any(probabilities < 0) or np.any(probabilities > 1) or
                                not np.isclose(probabilities.sum(), 1)):
                            raise ValueError("predict_proba must return finite normalized probabilities.")
                        query = 1.0 - float(probabilities.max()) >= uncertainty_threshold
                if query:
                    expert_action = self.sequences[word_index][position]
                    new_states.append(state)
                    new_actions.append(expert_action)
                    self.last_rollout_queries += 1
                # Mixture actions are supported only for unrestricted standard DAgger.
                if self.rng.random() < beta:
                    previous_action = expert_action
                else:
                    previous_action = int(policy.predict(state)[0])
        aggregated_data[0].extend(new_states)
        aggregated_data[1].extend(new_actions)
        print(f"Visited {self.last_rollout_visited} states; queried {self.last_rollout_queries}; "
              f"dataset now has {len(aggregated_data[0])} examples")
        return aggregated_data


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--query-strategy", choices=["standard", "uncertainty", "periodic"], default="standard")
    parser.add_argument("--query-budget", type=int, help="Maximum additional expert queries across the entire run; initial labels excluded.")
    parser.add_argument("--query-period", type=int, default=10, help="Query every kth training state across rollouts.")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.5, help="Query if 1-max(predict_proba) >= threshold.")
    parser.set_defaults(output_dir="results/label_budget")
    args = parser.parse_args()
    if args.all_folds and args.validation_fold is not None:
        parser.error("--all-folds cannot be combined with --validation-fold.")
    if args.validation_fold == args.test_fold:
        parser.error("Validation and test folds must differ.")
    learner = LabelBudgetDAgger(random_state=args.seed, alpha=args.alpha)
    options = dict(N=args.iterations, beta_decay=args.beta_decay, plot=not args.no_plot,
                   output_dir=args.output_dir, query_strategy=args.query_strategy,
                   query_budget=args.query_budget, query_period=args.query_period,
                   uncertainty_threshold=args.uncertainty_threshold)
    if args.all_folds:
        learner.run_cross_validation(**options)
    else:
        learner.run(test_fold=args.test_fold if args.validation_fold is None else args.validation_fold,
                    excluded_folds=() if args.validation_fold is None else (args.test_fold,),
                    evaluation_role="test" if args.validation_fold is None else "validation", **options)


if __name__ == "__main__":
    main()
