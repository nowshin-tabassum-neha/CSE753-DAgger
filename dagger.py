"""DAgger for the Stanford OCR sequence-labelling task.

This implementation follows Ross, Gordon, and Bagnell (2011): each word is
an episode, a state contains the current 8x16 character image and a one-hot
encoding of the previously executed character prediction, and the expert
action is the ground-truth character at the current position.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import sklearn
import os.path
import string
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


LETTER_TO_ID = dict(zip(string.ascii_lowercase, range(26)))
NUM_PIXELS = 128
NUM_LETTERS = 26
NUM_FEATURES = NUM_PIXELS + NUM_LETTERS


class DAgger:
    """Dataset Aggregation for left-to-right handwritten-word recognition."""

    def __init__(
        self,
        ocr_path: str = "letter.data",
        *,
        random_state: int = 0,
        alpha: float = 1e-4,
        policy_factory: Callable[[], ClassifierMixin] | None = None,
    ) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        self.alpha = alpha
        self.ocr_path = os.path.join("Dataset", str(ocr_path))
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.policy_factory = policy_factory or self._sgd_classifier

        # These three lists share the same word index.
        self.words: list[list[np.ndarray]] = []
        self.sequences: list[list[int]] = []
        self.words_fold: list[int] = []

        # Expert-trajectory (behavior-cloning) examples, separated by fold.
        self.dataset: dict[int, list[sparse.csr_matrix]] = {
            fold: [] for fold in range(10)
        }
        self.labels: dict[int, list[int]] = {fold: [] for fold in range(10)}
        self._initial_dataset_built = False
        self.last_rollout_queries = 0
        self.last_structured_bc_score: float | None = None
        self.last_no_structure_score: float | None = None
        self.last_no_structure_policy: ClassifierMixin | None = None

    def _sgd_classifier(self) -> SGDClassifier:
        """Return a reproducible SGD logistic classifier for DAgger and baselines."""
        return SGDClassifier(
            loss="log_loss",
            alpha=self.alpha,
            average=True,
            max_iter=1000,
            tol=1e-3,
            random_state=self.random_state,
        )

    @staticmethod
    def make_state(
        image: np.ndarray, previous_letter: int | None
    ) -> sparse.csr_matrix:
        """Encode s_t = (current pixels, previously executed prediction).

        The first character has no previous letter, so its 26 context features
        are all zero. Later characters have one active context feature.
        """
        pixels = np.asarray(image, dtype=np.float64).reshape(-1)
        if pixels.size != NUM_PIXELS:
            raise ValueError(
                f"Expected {NUM_PIXELS} pixels per character, got {pixels.size}."
            )

        context = np.zeros(NUM_LETTERS, dtype=np.float64)
        if previous_letter is not None:
            if not 0 <= int(previous_letter) < NUM_LETTERS:
                raise ValueError(f"Invalid previous-letter ID: {previous_letter}")
            context[int(previous_letter)] = 1.0

        features = np.concatenate((pixels, context)).reshape(1, NUM_FEATURES)
        return sparse.csr_matrix(features)

    def process_ocr(self) -> None:
        """Read ``letter.data`` and group character rows into word episodes."""
        if self.words:
            return

        print("Processing OCR dataset")
        current_images: list[np.ndarray] = []
        current_labels: list[int] = []

        with open(self.ocr_path, "r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                fields = raw_line.rstrip("\r\n").split("\t")
                if fields and fields[-1] == "":
                    fields.pop()
                if len(fields) != 6 + NUM_PIXELS:
                    raise ValueError(
                        f"Line {line_number}: expected {6 + NUM_PIXELS} fields, "
                        f"got {len(fields)}."
                    )

                label = LETTER_TO_ID[fields[1]]
                image = np.asarray(fields[6:], dtype=np.float64)
                current_labels.append(label)
                current_images.append(image)

                # Column 2 is the next-character ID; -1 terminates the word.
                if int(fields[2]) == -1:
                    self.words.append(current_images)
                    self.sequences.append(current_labels)
                    self.words_fold.append(int(fields[5]))
                    current_images = []
                    current_labels = []

        if current_images or current_labels:
            raise ValueError("The OCR file ends before the final word terminator.")

    def build_initial_dataset(self) -> None:
        """Collect D_1 from expert-controlled trajectories (beta_1 = 1)."""
        if self._initial_dataset_built:
            return

        self.process_ocr()
        print("Building initial expert dataset")

        for word_index, images in enumerate(self.words):
            fold = self.words_fold[word_index]
            labels = self.sequences[word_index]
            previous_expert_letter: int | None = None

            for image, expert_letter in zip(images, labels):
                state = self.make_state(image, previous_expert_letter)
                self.dataset[fold].append(state)
                self.labels[fold].append(expert_letter)
                previous_expert_letter = expert_letter

        self._initial_dataset_built = True

    def aggregate_dataset(self, policy, aggregated_data, test_fold=9, *, beta=0.0, excluded_folds=()):
        """Collect all training states for standard unrestricted DAgger."""
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        self.last_rollout_queries = self.last_rollout_visited = 0
        for word_index, images in enumerate(self.words):
            if self.words_fold[word_index] in (test_fold, *excluded_folds):
                continue
            previous_action = None
            for position, image in enumerate(images):
                state = self.make_state(image, previous_action)
                expert_action = self.sequences[word_index][position]
                aggregated_data[0].append(state)
                aggregated_data[1].append(expert_action)
                self.last_rollout_queries += 1
                self.last_rollout_visited += 1
                previous_action = (expert_action if self.rng.random() < beta
                                   else int(policy.predict(state)[0]))
        print(f"Visited {self.last_rollout_visited} states; queried {self.last_rollout_queries}; "
              f"dataset now has {len(aggregated_data[0])} examples")
        return aggregated_data

    def _experiment_config(self):
        return {"query_strategy": "standard", "query_budget": None}

    def _experiment_metrics(self, cumulative_queries):
        return {}

    def _collect_rollout(self, policy, data, test_fold, beta, excluded_folds,
                         cumulative_queries, visited_states):
        return self.aggregate_dataset(policy, data, test_fold, beta=beta, excluded_folds=excluded_folds)

    def _fit_policy(
        self, states: list[sparse.csr_matrix], expert_actions: list[int]
    ) -> ClassifierMixin:
        policy = self.policy_factory()
        policy.fit(sparse.vstack(states, format="csr"), np.asarray(expert_actions))
        return policy

    def _fit_no_structure_policy(
        self, states: list[sparse.csr_matrix], expert_actions: list[int]
    ) -> ClassifierMixin:
        """Train an independent-character classifier using only the 128 pixels."""
        pixel_features = sparse.vstack(states, format="csr")[:, :NUM_PIXELS]
        policy = self.policy_factory()
        policy.fit(pixel_features, np.asarray(expert_actions))
        return policy

    def evaluate_no_structure_policy(
        self, policy: ClassifierMixin, test_fold: int = 9
    ) -> float:
        """Evaluate characters independently, without previous-letter context."""
        test_states = self.dataset[test_fold]
        if not test_states:
            raise ValueError(f"Test fold {test_fold} contains no characters.")
        pixel_features = sparse.vstack(test_states, format="csr")[:, :NUM_PIXELS]
        predictions = policy.predict(pixel_features)
        return float(accuracy_score(self.labels[test_fold], predictions))

    def evaluate_policy(
        self, policy: ClassifierMixin, test_fold: int = 9
    ) -> float:
        """Evaluate by greedy free-running decoding on complete test words."""
        predictions: list[int] = []
        ground_truth: list[int] = []

        for word_index, images in enumerate(self.words):
            if self.words_fold[word_index] != test_fold:
                continue

            previous_prediction: int | None = None
            for image, expert_action in zip(
                images, self.sequences[word_index]
            ):
                state = self.make_state(image, previous_prediction)
                prediction = int(policy.predict(state)[0])
                predictions.append(prediction)
                ground_truth.append(expert_action)
                previous_prediction = prediction

        if not ground_truth:
            raise ValueError(f"Test fold {test_fold} contains no characters.")
        return float(accuracy_score(ground_truth, predictions))

    def run(
        self,
        N: int = 20,
        *,
        test_fold: int = 9,
        beta_decay: float = 0.0,
        plot: bool = True,
        output_dir: str | Path = "results",
        excluded_folds: tuple[int, ...] = (),
        evaluation_role: str = "test",
    ) -> tuple[np.ndarray, list[ClassifierMixin]]:
        """Run matched baselines and DAgger; checkpoint metrics after each fit.

        Iteration 1 is structured BC. Later rollouts use beta_decay**(i-1).
        excluded_folds are never used for training or aggregation, allowing
        a final test fold to remain untouched during validation runs.
        """
        if N < 1 or test_fold not in range(10):
            raise ValueError("N must be positive and test_fold must be in 0..9.")
        if not 0 <= beta_decay <= 1:
            raise ValueError("beta_decay must be between 0 and 1.")
        if any(f not in range(10) for f in excluded_folds):
            raise ValueError("Excluded folds must be in 0..9.")
        train_folds = [f for f in range(10) if f not in (test_fold, *excluded_folds)]
        if not train_folds:
            raise ValueError("At least one training fold is required.")
        started = perf_counter()
        self.build_initial_dataset()
        self.rng = np.random.default_rng(self.random_state)
        states = [s for f in train_folds for s in self.dataset[f]]
        actions = [y for f in train_folds for y in self.labels[f]]
        if not states or not self.dataset[test_fold]:
            raise ValueError("Training and evaluation folds must contain characters.")
        aggregated_data = [list(states), list(actions)]
        visited_states = 0
        self.last_rollout_visited = 0
        initial_labels = len(actions)
        cumulative_queries = 0
        queries_this_iteration = 0
        self.last_rollout_queries = 0
        run_dir = Path(output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        self.last_run_dir = run_dir
        config = {
            "status": "running", "seed": self.random_state,
            "iterations": N, "beta_decay": beta_decay,
            "train_folds": train_folds, "evaluation_fold": test_fold,
            "evaluation_role": evaluation_role, "excluded_folds": list(excluded_folds),
            "initial_dataset_size": len(states),
            "label_accounting": "Simulated expert requests, including repeated characters; initial training labels shared by baselines. Evaluation labels excluded. Mixed expert actions reuse the queried label.",
            "evaluation_characters": len(self.labels[test_fold]),
            "classifier": type(self.policy_factory()).__name__,
            "classifier_params": self.policy_factory().get_params(),
            "dataset_path": str(Path(self.ocr_path).resolve()),
            "dataset_sha256": hashlib.sha256(Path(self.ocr_path).read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "timing_note": "Training includes feature stacking; rollout includes aggregation. Elapsed includes setup and logging, excludes plotting.",
        }
        config.update(self._experiment_config())
        def save_config():
            (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        save_config()
        print(f"Training folds: {train_folds}; {evaluation_role} fold: {test_fold}")
        print(f"Initial dataset: {len(states)} examples; results: {run_dir}")
        records = []
        def record(method, iteration, beta, score, training, rollout, evaluation):
            row = dict(method=method, iteration=iteration, beta=beta,
                       accuracy=score, imitation_error=1-score,
                       dataset_size=len(aggregated_data[0]), seed=self.random_state,
                       visited_states_this_iteration=self.last_rollout_visited,
                       cumulative_visited_states=visited_states,
                       initial_labels=initial_labels,
                       queries_this_iteration=queries_this_iteration,
                       cumulative_queries=cumulative_queries,
                       total_labels_used=initial_labels+cumulative_queries,
                       evaluation_fold=test_fold, training_seconds=training,
                       rollout_seconds=rollout, evaluation_seconds=evaluation,
                       elapsed_seconds=perf_counter()-started)
            row.update(self._experiment_metrics(cumulative_queries))
            records.append(row)
            with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                if len(records) == 1:
                    writer.writeheader()
                writer.writerow(row)
            print(f"{method} iteration {iteration}: accuracy={score:.4f}, train={training:.2f}s, rollout={rollout:.2f}s, "
                  f"new queries={queries_this_iteration}, cumulative queries={cumulative_queries}, "
                  f"total labels={initial_labels+cumulative_queries}")
        t = perf_counter()
        baseline = self._fit_no_structure_policy(states, actions)
        training = perf_counter()-t
        t = perf_counter()
        baseline_score = self.evaluate_no_structure_policy(baseline, test_fold)
        record("no_structure", 0, 1.0, baseline_score, training, 0.0, perf_counter()-t)
        self.last_no_structure_policy = baseline
        self.last_no_structure_score = baseline_score
        policies, scores = [], []
        for iteration in range(1, N+1):
            beta = 1.0 if iteration == 1 else beta_decay ** (iteration-1)
            rollout = 0.0
            queries_this_iteration = 0
            if iteration > 1:
                t = perf_counter()
                self._collect_rollout(policies[-1], aggregated_data, test_fold,
                                      beta, excluded_folds, cumulative_queries, visited_states)
                rollout = perf_counter()-t
                queries_this_iteration = self.last_rollout_queries
                cumulative_queries += queries_this_iteration
                visited_states += self.last_rollout_visited
            t = perf_counter()
            policy = (self._fit_policy(*aggregated_data)
                      if iteration == 1 or queries_this_iteration else policies[-1])
            training = perf_counter()-t
            t = perf_counter()
            score = self.evaluate_policy(policy, test_fold)
            evaluation = perf_counter()-t
            policies.append(policy)
            scores.append(score)
            record("structured_bc" if iteration == 1 else "dagger", iteration,
                   beta, score, training, rollout, evaluation)
        self.last_structured_bc_score = scores[0]
        self.last_run_records = records
        config.update(status="complete", total_seconds=perf_counter()-started)
        save_config()
        final_scores = np.asarray(scores)
        if plot:
            self.plot_scores(final_scores, supervised_score=scores[0],
                             no_structure_score=baseline_score, test_fold=test_fold)
        return final_scores, policies

    @staticmethod
    def plot_scores(
        scores: np.ndarray,
        *,
        supervised_score: float,
        no_structure_score: float,
        test_fold: int | None = None,
        dagger_confidence: np.ndarray | None = None,
        supervised_confidence: float | None = None,
        no_structure_confidence: float | None = None,
    ) -> None:
        """Plot DAgger, structured BC, and no-structure supervision."""
        iterations = np.arange(1, len(scores) + 1)
        supervised_baseline = np.full(len(scores), supervised_score)
        no_structure_baseline = np.full(len(scores), no_structure_score)

        _, axis = plt.subplots()
        axis.plot(
            iterations, scores, color="C0", marker="o", label="DAgger"
        )
        if dagger_confidence is not None:
            axis.fill_between(
                iterations,
                scores - dagger_confidence,
                scores + dagger_confidence,
                color="C0",
                alpha=0.2,
                label="DAgger 95% CI",
            )
        axis.plot(
            iterations,
            supervised_baseline,
            color="C1",
            linestyle="--",
            label="Structured BC",
        )
        axis.plot(
            iterations,
            no_structure_baseline,
            color="C2",
            linestyle=":",
            label="No-structure supervised",
        )
        if supervised_confidence is not None:
            axis.fill_between(
                iterations,
                supervised_baseline - supervised_confidence,
                supervised_baseline + supervised_confidence,
                color="C1",
                alpha=0.12,
            )
        if no_structure_confidence is not None:
            axis.fill_between(
                iterations,
                no_structure_baseline - no_structure_confidence,
                no_structure_baseline + no_structure_confidence,
                color="C2",
                alpha=0.12,
            )

        y_label = (
            "Mean character accuracy"
            if test_fold is None
            else "Character accuracy"
        )
        axis.set_ylabel(y_label)
        axis.set_xlabel("DAgger iteration")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_xticks(iterations)
        axis.grid(alpha=0.2)
        title_suffix = "" if test_fold is None else f" (test fold {test_fold})"
        axis.set_title(f"Stanford OCR: DAgger vs. supervised baselines{title_suffix}")
        axis.legend()
        plt.tight_layout()
        plt.show()

    def run_cross_validation(
        self,
        N: int = 20,
        *,
        beta_decay: float = 0.0,
        plot: bool = True,
        output_dir: str | Path = "results",
        **run_options,
    ) -> np.ndarray:
        """Run the paper's large-data protocol, holding out every fold once."""
        fold_scores = []
        structured_bc_scores = []
        no_structure_scores = []
        for test_fold in range(10):
            print(f"\n=== Test fold {test_fold} ===")
            scores, _ = self.run(
                N=N,
                test_fold=test_fold,
                beta_decay=beta_decay,
                plot=False,
                output_dir=output_dir,
                **run_options,
            )
            fold_scores.append(scores)
            structured_bc_scores.append(scores[0])
            if self.last_no_structure_score is None:
                raise RuntimeError("No-structure baseline was not evaluated.")
            no_structure_scores.append(self.last_no_structure_score)

        all_scores = np.vstack(fold_scores)
        mean_scores = all_scores.mean(axis=0)
        structured_bc_scores_array = np.asarray(structured_bc_scores)
        no_structure_scores_array = np.asarray(no_structure_scores)
        confidence_scale = 1.96 / np.sqrt(all_scores.shape[0])
        dagger_confidence = all_scores.std(axis=0, ddof=1) * confidence_scale
        structured_confidence = (
            structured_bc_scores_array.std(ddof=1) * confidence_scale
        )
        no_structure_confidence = (
            no_structure_scores_array.std(ddof=1) * confidence_scale
        )

        print(f"Mean final accuracy across folds: {mean_scores[-1]:.4f}")
        print(
            "Mean structured BC accuracy across folds: "
            f"{structured_bc_scores_array.mean():.4f}"
        )
        print(
            "Mean no-structure accuracy across folds: "
            f"{no_structure_scores_array.mean():.4f}"
        )
        if plot:
            self.plot_scores(
                mean_scores,
                supervised_score=float(structured_bc_scores_array.mean()),
                no_structure_score=float(no_structure_scores_array.mean()),
                dagger_confidence=dagger_confidence,
                supervised_confidence=float(structured_confidence),
                no_structure_confidence=float(no_structure_confidence),
            )
        return all_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DAgger on the Stanford OCR handwriting dataset."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--test-fold", type=int, default=9, choices=range(10))
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Repeat training with each of the ten folds held out (paper protocol).",
    )
    parser.add_argument(
        "--beta-decay",
        type=float,
        default=0.0,
        help=(
            "Use beta_i=p^(i-1). The default 0 implements beta_i=I(i=1), "
            "as used for the paper's OCR DAgger result."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--validation-fold", type=int, choices=range(10),
                        help="Evaluate on this fold while reserving --test-fold entirely.")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_folds and args.validation_fold is not None:
        raise SystemExit("--all-folds cannot be combined with --validation-fold.")
    if args.validation_fold == args.test_fold:
        raise SystemExit("Validation and test folds must differ.")
    dagger = DAgger(random_state=args.seed, alpha=args.alpha)
    if args.all_folds:
        dagger.run_cross_validation(
            N=args.iterations,
            beta_decay=args.beta_decay,
            plot=not args.no_plot,
            output_dir=args.output_dir,
        )
    else:
        scores, trained_policies = dagger.run(
            N=args.iterations,
            test_fold=args.test_fold if args.validation_fold is None else args.validation_fold,
            excluded_folds=() if args.validation_fold is None else (args.test_fold,),
            evaluation_role="test" if args.validation_fold is None else "validation",
            beta_decay=args.beta_decay,
            plot=not args.no_plot,
            output_dir=args.output_dir,
        )
