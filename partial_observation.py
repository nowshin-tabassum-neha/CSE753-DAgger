"""Reproducible masked/noisy OCR observations for BC and DAgger experiments.

This is an observation-corruption study using the existing SGD policies.
It does not yet implement a recurrent policy or demonstrate a memory benefit.
"""

import hashlib
from pathlib import Path

import numpy as np

from dagger import DAgger, build_parser


def corrupt_image(image, *, mask_rate=0.0, noise_std=0.0, seed=0, image_id=0):
    """Add Gaussian noise, clip to [0,1], then hide random pixels with zeros.

    Independent streams make masks nested as mask_rate increases and keep
    corruption reproducible regardless of policy seed or evaluation order.
    """
    if not np.isfinite(mask_rate) or not 0 <= mask_rate <= 1:
        raise ValueError("mask_rate must be in [0,1].")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative.")
    if seed < 0 or image_id < 0:
        raise ValueError("Corruption seed and image ID must be nonnegative.")
    pixels = np.asarray(image, dtype=np.float64).copy()
    if not np.all(np.isfinite(pixels)) or np.any(pixels < 0) or np.any(pixels > 1):
        raise ValueError("OCR pixels must be finite values in [0,1].")
    if noise_std:
        rng = np.random.default_rng(np.random.SeedSequence([seed, image_id, 0]))
        pixels = np.clip(pixels + rng.normal(0, noise_std, pixels.shape), 0, 1)
    if mask_rate:
        rng = np.random.default_rng(np.random.SeedSequence([seed, image_id, 1]))
        pixels[rng.random(pixels.shape) < mask_rate] = 0
    return pixels


class PartialObservationDAgger(DAgger):
    def __init__(self, *args, mask_rate=0.0, noise_std=0.0, observation_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        corrupt_image(np.zeros(128), mask_rate=mask_rate, noise_std=noise_std, seed=observation_seed)
        self.mask_rate = mask_rate
        self.noise_std = noise_std
        self.observation_seed = observation_seed
        self._observations_ready = False

    def process_ocr(self):
        super().process_ocr()
        if self._observations_ready:
            return
        image_id = 0
        digest = hashlib.sha256()
        for word in self.words:
            for position, image in enumerate(word):
                observed = corrupt_image(image, mask_rate=self.mask_rate,
                                         noise_std=self.noise_std,
                                         seed=self.observation_seed, image_id=image_id)
                word[position] = observed
                digest.update(observed.astype("<f8").tobytes())
                image_id += 1
        self.observation_sha256 = digest.hexdigest()
        self._observations_ready = True

    def _experiment_config(self):
        return {**super()._experiment_config(), "study": "partial_observation_ocr",
                "mask_rate": self.mask_rate, "noise_std": self.noise_std,
                "observation_seed": self.observation_seed,
                "observation_sha256": self.observation_sha256,
                "corruption_scope": "all training and evaluation images",
                "corruption_schedule": "fixed per character across iterations and methods",
                "corruption_order": "Gaussian noise, clip to [0,1], random zero masking",
                "mask_indicator": False,
                "memory_policy": False,
                "observation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--mask-rate", type=float, default=0.0,
                        help="Independent probability of hiding each pixel with zero (0..1).")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise standard deviation on the 0..1 pixel scale.")
    parser.add_argument("--observation-seed", type=int, default=0,
                        help="Corruption seed, separate from classifier --seed.")
    parser.set_defaults(iterations=5, alpha=1e-5, output_dir="results/partial_observation")
    args = parser.parse_args()
    if args.all_folds and args.validation_fold is not None:
        parser.error("--all-folds cannot be combined with --validation-fold.")
    if args.validation_fold == args.test_fold:
        parser.error("Validation and test folds must differ.")
    learner = PartialObservationDAgger(alpha=args.alpha, random_state=args.seed,
                                       mask_rate=args.mask_rate, noise_std=args.noise_std,
                                       observation_seed=args.observation_seed)
    options = dict(N=args.iterations, beta_decay=args.beta_decay,
                   plot=not args.no_plot, output_dir=args.output_dir)
    if args.all_folds:
        learner.run_cross_validation(**options)
    else:
        learner.run(test_fold=args.test_fold if args.validation_fold is None else args.validation_fold,
                    excluded_folds=() if args.validation_fold is None else (args.test_fold,),
                    evaluation_role="test" if args.validation_fold is None else "validation", **options)


if __name__ == "__main__":
    main()
