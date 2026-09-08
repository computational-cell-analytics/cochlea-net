"""Evaluate the OTOF expression classifiers with stratified five-fold cross-validation."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DATASETS = {
    "LaVision-OTOF23R": {
        "positive": [
            2040, 1951, 1952, 1949, 1935, 1936, 1885, 1894, 1915, 5110, 3103, 3104,
            3316, 3196, 3204, 2859, 2611, 2604, 469, 102, 106, 82, 161,
        ],
        "negative": [
            2038, 2036, 2037, 1946, 1882, 1934, 1900, 1913, 2046, 3755, 4728, 3250,
            3256, 3269, 3290, 3308, 3074, 2340, 264, 92, 3217, 2610, 762, 2606,
        ],
    },
    "LaVision-OTOF25R": {
        "positive": [
            4804, 4811, 4784, 4838, 5005, 5001, 5223, 3883, 3874, 4204, 4489, 4358,
            4224, 3911, 3264, 2621, 2533, 2544, 2545, 4431, 4433, 4427, 4439, 4493,
            4369, 4405, 3265, 3265, 3883, 5005,
        ],
        "negative": [
            4815, 4820, 4837, 4997, 5003, 5004, 3588, 3869, 4187, 4333, 4463, 4491,
            4490, 4359, 3438, 3442, 2540, 1305, 2798, 2799, 2599, 2506,
        ],
    },
}

FEATURE_COLUMNS = [
    "intensity-p1",
    "intensity-p10",
    "intensity-p25",
    "median-intensity",
    "intensity-p75",
    "intensity-p90",
    "intensity-p99",
    "mean-intensity",
    "std-intensity",
]

N_SPLITS = 5


@dataclass
class Evaluation:
    name: str
    n_positive: int
    n_negative: int
    accuracies: np.ndarray
    correct: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.n_positive + self.n_negative


def load_annotated_features(data_root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the saved features and select the manually annotated IHCs."""
    table_path = data_root / dataset / "expression_classification.tsv"
    if not table_path.is_file():
        raise FileNotFoundError(f"Feature table not found: {table_path}")

    table = pd.read_csv(table_path, sep="\t")
    required_columns = {"label_id", *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(f"{table_path} is missing columns: {', '.join(missing_columns)}")

    positive_ids = np.unique(DATASETS[dataset]["positive"])
    negative_ids = np.unique(DATASETS[dataset]["negative"])
    overlap = np.intersect1d(positive_ids, negative_ids)
    if overlap.size:
        raise ValueError(f"{dataset} has label IDs in both classes: {overlap.tolist()}")

    label_ids = table["label_id"].to_numpy(dtype=int)
    annotated_ids = np.concatenate([positive_ids, negative_ids])
    missing_ids = np.setdiff1d(annotated_ids, label_ids)
    if missing_ids.size:
        raise ValueError(f"{dataset} is missing annotated label IDs: {missing_ids.tolist()}")

    annotated = np.isin(label_ids, annotated_ids)
    features = table.loc[annotated, FEATURE_COLUMNS]
    if features.isna().any().any():
        raise ValueError(f"{dataset} has missing values in its annotated features")

    selected_ids = label_ids[annotated]
    labels = np.where(np.isin(selected_ids, positive_ids), 0, 1)
    return features.to_numpy(), labels


def evaluate_dataset(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    repeats: int,
    seed: int,
    jobs: int,
) -> Evaluation:
    """Run repeated stratified five-fold evaluation for one dataset."""
    class_counts = np.bincount(labels, minlength=2)
    if np.min(class_counts) < N_SPLITS:
        raise ValueError(
            f"{name} needs at least {N_SPLITS} annotations per class; found {class_counts.tolist()}"
        )

    accuracies = np.empty(repeats, dtype=float)
    correct = np.empty(repeats, dtype=int)
    for repeat in range(repeats):
        random_state = seed + repeat
        folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=random_state,
            n_jobs=1,
        )
        predictions = cross_val_predict(classifier, features, labels, cv=folds, n_jobs=jobs)
        correct[repeat] = np.count_nonzero(predictions == labels)
        accuracies[repeat] = correct[repeat] / len(labels)

    return Evaluation(
        name=name,
        n_positive=int(class_counts[0]),
        n_negative=int(class_counts[1]),
        accuracies=accuracies,
        correct=correct,
    )


def format_accuracy(values: np.ndarray) -> str:
    """Format the mean, sample standard deviation, and range as percentages."""
    if len(values) == 1:
        return f"{values[0]:.1%}"
    return (
        f"{values.mean():.1%} +/- {values.std(ddof=1):.1%} "
        f"(range {values.min():.1%}-{values.max():.1%})"
    )


def run_evaluation(data_root: Path, repeats: int, seed: int, jobs: int) -> None:
    """Evaluate both classifiers and print per-dataset and pooled accuracy."""
    evaluations = []
    for name in DATASETS:
        features, labels = load_annotated_features(data_root, name)
        evaluations.append(evaluate_dataset(name, features, labels, repeats, seed, jobs))

    print(f"Stratified {N_SPLITS}-fold cross-validation ({repeats} repeat(s), seed {seed})")
    for result in evaluations:
        print(
            f"{result.name}: n={result.n_samples} "
            f"(positive={result.n_positive}, negative={result.n_negative}), "
            f"accuracy={format_accuracy(result.accuracies)}"
        )

    total_samples = sum(result.n_samples for result in evaluations)
    pooled_correct = sum((result.correct for result in evaluations), start=np.zeros(repeats, dtype=int))
    pooled_accuracy = pooled_correct / total_samples
    print(f"Aggregate (sample-weighted): n={total_samples}, accuracy={format_accuracy(pooled_accuracy)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the OTOF23R and OTOF25R expression classifiers with repeated "
            "stratified five-fold cross-validation."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing the per-dataset expression_classification.tsv files.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of shuffled five-fold evaluations. Default: 10.",
    )
    parser.add_argument("--seed", type=int, default=0, help="First random seed. Default: 0.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel cross-validation jobs. Use -1 for all CPUs. Default: 1.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    run_evaluation(args.data_root, args.repeats, args.seed, args.jobs)


if __name__ == "__main__":
    main()
