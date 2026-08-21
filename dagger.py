"""DAgger for the Stanford OCR sequence-labelling task.

This implementation follows Ross, Gordon, and Bagnell (2011): each word is
an episode, a state contains the current 8x16 character image and a one-hot
encoding of the previously executed character prediction, and the expert
action is the ground-truth character at the current position.
"""

from __future__ import annotations

import argparse
import os.path
import string
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn import svm
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
        policy_factory: Callable[[], ClassifierMixin] | None = None,
    ) -> None:
        self.ocr_path = os.path.join("Dataset", str(ocr_path))
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.policy_factory = policy_factory or self._paper_svm

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

    @staticmethod
    def _paper_svm() -> svm.SVC:
        """Return a linear one-vs-one SVM, matching the paper's base learner."""
        return svm.SVC(
            C=10,
            kernel="linear",
            decision_function_shape="ovo",
            cache_size=2000,
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

    def aggregate_dataset(
        self,
        policy: ClassifierMixin,
        aggregated_data: list[list],
        test_fold: int = 9,
        *,
        beta: float = 0.0,
    ) -> list[list]:
        """Roll out the mixture policy and aggregate expert-labelled states.

        At every visited state, the ground-truth OCR label supplies pi*(s).
        The action that affects the next state is selected from the expert with
        probability ``beta`` and from the learner with probability ``1-beta``.
        """
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be between 0 and 1.")

        new_states: list[sparse.csr_matrix] = []
        new_expert_actions: list[int] = []

        for word_index, images in enumerate(self.words):
            if self.words_fold[word_index] == test_fold:
                continue

            expert_actions = self.sequences[word_index]
            previous_executed_action: int | None = None

            # This is a complete left-to-right trajectory. Crucially, the
            # selected action at t becomes the context in state t+1.
            for image, expert_action in zip(images, expert_actions):
                state = self.make_state(image, previous_executed_action)
                new_states.append(state)
                new_expert_actions.append(expert_action)

                if self.rng.random() < beta:
                    executed_action = expert_action
                else:
                    executed_action = int(policy.predict(state)[0])
                previous_executed_action = executed_action

        aggregated_data[0].extend(new_states)
        aggregated_data[1].extend(new_expert_actions)
        print(
            f"Aggregated {len(new_states)} states; "
            f"dataset now has {len(aggregated_data[0])} examples"
        )
        return aggregated_data

    def _fit_policy(
        self, states: list[sparse.csr_matrix], expert_actions: list[int]
    ) -> ClassifierMixin:
        policy = self.policy_factory()
        policy.fit(sparse.vstack(states, format="csr"), np.asarray(expert_actions))
        return policy

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
    ) -> tuple[np.ndarray, list[ClassifierMixin]]:
        """Run DAgger on one held-out fold.

        ``beta_decay=0`` gives the paper's parameter-free schedule
        beta_i = I(i=1): the first dataset is expert-controlled and every later
        rollout is learner-controlled. For p in (0, 1], iteration i instead
        uses beta_i = p**(i-1).
        """
        if N < 1:
            raise ValueError("N must be at least 1.")
        if test_fold not in range(10):
            raise ValueError("test_fold must be in 0..9.")
        if not 0.0 <= beta_decay <= 1.0:
            raise ValueError("beta_decay must be between 0 and 1.")

        self.build_initial_dataset()
        self.rng = np.random.default_rng(self.random_state)

        train_states = sum(
            (
                self.dataset[fold]
                for fold in range(10)
                if fold != test_fold
            ),
            [],
        )
        train_actions = sum(
            (
                self.labels[fold]
                for fold in range(10)
                if fold != test_fold
            ),
            [],
        )
        aggregated_data: list[list] = [list(train_states), list(train_actions)]

        print(
            f"Training on folds other than {test_fold}; "
            f"initial dataset has {len(train_states)} examples"
        )

        policies: list[ClassifierMixin] = []
        scores: list[float] = []

        # Iteration 1: train on expert trajectories (behavior cloning).
        policy = self._fit_policy(aggregated_data[0], aggregated_data[1])
        policies.append(policy)
        score = self.evaluate_policy(policy, test_fold)
        scores.append(score)
        print(f"Iteration 1/{N}: free-running character accuracy = {score:.4f}")

        # Iterations 2..N: collect under pi_i, aggregate, and retrain.
        for iteration in range(2, N + 1):
            beta = beta_decay ** (iteration - 1)
            self.aggregate_dataset(
                policy,
                aggregated_data,
                test_fold,
                beta=beta,
            )
            policy = self._fit_policy(aggregated_data[0], aggregated_data[1])
            policies.append(policy)
            score = self.evaluate_policy(policy, test_fold)
            scores.append(score)
            print(
                f"Iteration {iteration}/{N}: beta={beta:.4f}, "
                f"free-running character accuracy = {score:.4f}"
            )

        final_scores = np.asarray(scores)
        if plot:
            self.plot_scores(final_scores, test_fold=test_fold)
        return final_scores, policies

    @staticmethod
    def plot_scores(scores: np.ndarray, *, test_fold: int | None = None) -> None:
        """Plot DAgger against its behavior-cloning starting policy."""
        iterations = np.arange(1, len(scores) + 1)
        supervised_baseline = np.full(len(scores), scores[0])

        plt.plot(iterations, scores, marker="o", label="DAgger")
        plt.plot(
            iterations,
            supervised_baseline,
            linestyle="--",
            label="Supervised (iteration 1)",
        )
        plt.ylabel("Free-running character accuracy")
        plt.xlabel("Training iteration")
        title_suffix = "" if test_fold is None else f" (test fold {test_fold})"
        plt.title(f"Stanford OCR DAgger{title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_cross_validation(
        self,
        N: int = 20,
        *,
        beta_decay: float = 0.0,
        plot: bool = True,
    ) -> np.ndarray:
        """Run the paper's large-data protocol, holding out every fold once."""
        fold_scores = []
        for test_fold in range(10):
            print(f"\n=== Test fold {test_fold} ===")
            scores, _ = self.run(
                N=N,
                test_fold=test_fold,
                beta_decay=beta_decay,
                plot=False,
            )
            fold_scores.append(scores)

        all_scores = np.vstack(fold_scores)
        mean_scores = all_scores.mean(axis=0)
        print(f"Mean final accuracy across folds: {mean_scores[-1]:.4f}")
        if plot:
            self.plot_scores(mean_scores)
        return all_scores


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dagger = DAgger()
    if args.all_folds:
        dagger.run_cross_validation(
            N=args.iterations,
            beta_decay=args.beta_decay,
            plot=not args.no_plot,
        )
    else:
        scores, trained_policies = dagger.run(
            N=args.iterations,
            test_fold=args.test_fold,
            beta_decay=args.beta_decay,
            plot=not args.no_plot,
        )
