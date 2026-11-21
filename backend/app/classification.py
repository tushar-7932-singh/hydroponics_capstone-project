"""CPU classification experiment using the frozen CBIR feature index."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.cbir import (
    CLASS_LABELS,
    DEFAULT_INDEX_PATH,
    FEATURE_DIMENSION,
    PROJECT_ROOT,
    load_index,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "fnnpk-linear-classifier.npz"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fnnpk-classification-evaluation.json"
)
DEFAULT_FOLDS = 4
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 150
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 0.0001


@dataclass(frozen=True)
class LinearFeatureClassifier:
    classes: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    index_sha256: str | None = None

    def predict(self, features: np.ndarray) -> list[str]:
        """Predict class labels from one or more 576-dimensional feature rows."""
        values = _validate_features(features)
        scaled = (values - self.feature_mean) / self.feature_scale
        logits = scaled @ self.weights.T + self.bias
        return [self.classes[position] for position in np.argmax(logits, axis=1)]


def _validate_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"features must have shape (sample_count, {FEATURE_DIMENSION})."
        )
    if values.shape[0] == 0:
        raise ValueError("features must contain at least one sample.")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values.")
    return values


def _validate_labels(labels: Sequence[str], sample_count: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.str_)
    if values.ndim != 1 or len(values) != sample_count:
        raise ValueError("labels must be one-dimensional and match the feature rows.")
    unknown = sorted(set(values.tolist()).difference(CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unknown class labels: {', '.join(unknown)}.")
    missing = [label for label in CLASS_LABELS if label not in values]
    if missing:
        raise ValueError(f"Training data is missing classes: {', '.join(missing)}.")
    return values


def feature_group_ids(features: np.ndarray) -> np.ndarray:
    """Assign identical feature rows to the same group for leakage-safe splits."""
    values = _validate_features(features)
    group_by_digest: dict[bytes, int] = {}
    group_ids = np.empty(len(values), dtype=np.int64)
    for position, row in enumerate(values):
        digest = hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
        if digest not in group_by_digest:
            group_by_digest[digest] = len(group_by_digest)
        group_ids[position] = group_by_digest[digest]
    return group_ids


def stratified_group_folds(
    labels: Sequence[str],
    groups: np.ndarray,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reproducible folds, stratified by label and kept whole by group."""
    label_values = np.asarray(labels, dtype=np.str_)
    group_values = np.asarray(groups)
    if label_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional.")
    if len(label_values) == 0 or len(label_values) != len(group_values):
        raise ValueError("labels and groups must have the same non-zero length.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    group_labels: dict[object, str] = {}
    group_positions: dict[object, list[int]] = {}
    for position, (label, group) in enumerate(zip(label_values, group_values)):
        group_key = group.item() if isinstance(group, np.generic) else group
        previous_label = group_labels.setdefault(group_key, str(label))
        if previous_label != label:
            raise ValueError("A feature group contains samples from different classes.")
        group_positions.setdefault(group_key, []).append(position)

    class_groups: dict[str, list[object]] = {label: [] for label in CLASS_LABELS}
    for group, label in group_labels.items():
        if label not in class_groups:
            raise ValueError(f"Unknown class label: {label}.")
        class_groups[label].append(group)

    rng = np.random.default_rng(seed)
    fold_groups: list[set[object]] = [set() for _ in range(n_splits)]
    for label in CLASS_LABELS:
        groups_for_class = class_groups[label]
        if len(groups_for_class) < n_splits:
            raise ValueError(
                f"Class {label} has {len(groups_for_class)} independent groups; "
                f"{n_splits} folds require at least {n_splits}."
            )
        shuffled = [groups_for_class[position] for position in rng.permutation(len(groups_for_class))]
        shuffled.sort(key=lambda group: -len(group_positions[group]))
        fold_sample_counts = [0] * n_splits
        for group in shuffled:
            target_fold = min(
                range(n_splits),
                key=lambda fold: (fold_sample_counts[fold], fold),
            )
            fold_groups[target_fold].add(group)
            fold_sample_counts[target_fold] += len(group_positions[group])

    all_positions = np.arange(len(label_values), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_groups in fold_groups:
        test_positions = np.asarray(
            [
                position
                for group in test_groups
                for position in group_positions[group]
            ],
            dtype=np.int64,
        )
        test_positions.sort()
        test_mask = np.zeros(len(label_values), dtype=bool)
        test_mask[test_positions] = True
        train_positions = all_positions[~test_mask]
        folds.append((train_positions, test_positions))

    test_appearances = np.concatenate([test for _, test in folds])
    if not np.array_equal(np.sort(test_appearances), all_positions):
        raise RuntimeError("Cross-validation folds did not assign every sample once.")
    return folds


def _fit_classifier(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> LinearFeatureClassifier:
    import torch
    from torch import nn

    values = _validate_features(features)
    label_values = _validate_labels(labels, len(values))
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")

    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    standardized = ((values - mean) / scale).astype(np.float32)
    class_to_position = {label: position for position, label in enumerate(CLASS_LABELS)}
    target_values = np.asarray(
        [class_to_position[label] for label in label_values],
        dtype=np.int64,
    )
    class_counts = np.bincount(target_values, minlength=len(CLASS_LABELS))
    class_weights = len(target_values) / (len(CLASS_LABELS) * class_counts)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Linear(FEATURE_DIMENSION, len(CLASS_LABELS), device="cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device="cpu")
        )
        input_tensor = torch.from_numpy(standardized)
        target_tensor = torch.from_numpy(target_values)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(input_tensor), target_tensor)
            loss.backward()
            optimizer.step()

    return LinearFeatureClassifier(
        classes=CLASS_LABELS,
        feature_mean=mean,
        feature_scale=scale,
        weights=model.weight.detach().cpu().numpy().copy(),
        bias=model.bias.detach().cpu().numpy().copy(),
    )


def calculate_classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, object]:
    truth = np.asarray(true_labels, dtype=np.str_)
    predictions = np.asarray(predicted_labels, dtype=np.str_)
    if truth.ndim != 1 or predictions.ndim != 1 or len(truth) != len(predictions):
        raise ValueError("true_labels and predicted_labels must be equal-length vectors.")
    if len(truth) == 0:
        raise ValueError("At least one prediction is required to calculate metrics.")
    invalid = sorted(set(truth.tolist() + predictions.tolist()).difference(CLASS_LABELS))
    if invalid:
        raise ValueError(f"Unknown class labels: {', '.join(invalid)}.")

    confusion = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    label_to_position = {label: position for position, label in enumerate(CLASS_LABELS)}
    for actual, predicted in zip(truth, predictions):
        confusion[label_to_position[str(actual)], label_to_position[str(predicted)]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for position, label in enumerate(CLASS_LABELS):
        true_positive = int(confusion[position, position])
        false_positive = int(confusion[:, position].sum() - true_positive)
        false_negative = int(confusion[position, :].sum() - true_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int(confusion[position, :].sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = float(np.trace(confusion) / confusion.sum())
    macro_precision = float(np.mean([values["precision"] for values in per_class.values()]))
    macro_recall = float(np.mean([values["recall"] for values in per_class.values()]))
    macro_f1 = float(np.mean([values["f1"] for values in per_class.values()]))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {
            "labels_in_order": list(CLASS_LABELS),
            "rows_true_columns_predicted": confusion.tolist(),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_classifier(classifier: LinearFeatureClassifier, path: Path, index_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "numpy-npz-linear-pytorch-state",
        "input_feature_dimension": FEATURE_DIMENSION,
        "output_class_count": len(CLASS_LABELS),
        "class_mapping": {str(position): label for position, label in enumerate(CLASS_LABELS)},
        "architecture": "torch.nn.Linear(576, 4)",
        "feature_scaling": "per-feature standardization fit on training data",
        "index_sha256": index_hash,
    }
    with path.open("wb") as target:
        np.savez_compressed(
            target,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            classes=np.asarray(classifier.classes, dtype=np.str_),
            feature_mean=classifier.feature_mean,
            feature_scale=classifier.feature_scale,
            weights=classifier.weights,
            bias=classifier.bias,
        )


def load_classifier(path: Path) -> LinearFeatureClassifier:
    """Load the array-only classifier artifact without unpickling."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            classes = tuple(str(label) for label in archive["classes"].tolist())
            feature_mean = np.asarray(archive["feature_mean"], dtype=np.float32).copy()
            feature_scale = np.asarray(archive["feature_scale"], dtype=np.float32).copy()
            weights = np.asarray(archive["weights"], dtype=np.float32).copy()
            bias = np.asarray(archive["bias"], dtype=np.float32).copy()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read classifier artifact: {exc}") from exc

    if metadata.get("input_feature_dimension") != FEATURE_DIMENSION:
        raise ValueError("Classifier artifact has an incompatible feature dimension.")
    index_sha256 = metadata.get("index_sha256")
    if (
        not isinstance(index_sha256, str)
        or len(index_sha256) != 64
        or any(character not in "0123456789abcdef" for character in index_sha256)
    ):
        raise ValueError("Classifier artifact has an invalid feature-index SHA-256.")
    if classes != CLASS_LABELS:
        raise ValueError("Classifier artifact has an incompatible class mapping.")
    if (
        feature_mean.shape != (FEATURE_DIMENSION,)
        or feature_scale.shape != (FEATURE_DIMENSION,)
        or weights.shape != (len(CLASS_LABELS), FEATURE_DIMENSION)
        or bias.shape != (len(CLASS_LABELS),)
    ):
        raise ValueError("Classifier artifact contains invalid parameter shapes.")
    if (
        not np.isfinite(feature_mean).all()
        or not np.isfinite(feature_scale).all()
        or np.any(feature_scale <= 0)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Classifier artifact contains invalid parameter values.")
    return LinearFeatureClassifier(
        classes=classes,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        bias=bias,
        index_sha256=index_sha256,
    )


def evaluate_and_train(
    index_path: Path = DEFAULT_INDEX_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, object]:
    """Run stratified group cross-validation, then save a final all-data model."""
    import torch

    index_path = Path(index_path).resolve()
    model_path = Path(model_path)
    report_path = Path(report_path)
    if not index_path.is_file():
        raise FileNotFoundError("Feature index does not exist.")
    index = load_index(index_path)
    features = _validate_features(index.features)
    labels = _validate_labels(index.labels.tolist(), len(features))
    groups = feature_group_ids(features)
    group_labels: dict[int, str] = {}
    for label, group in zip(labels, groups):
        previous = group_labels.setdefault(int(group), str(label))
        if previous != label:
            raise ValueError("Identical feature vectors have conflicting class labels.")

    folds = stratified_group_folds(labels.tolist(), groups, n_splits=n_splits, seed=seed)
    label_width = max(len(label) for label in CLASS_LABELS)
    out_of_fold_predictions = np.empty(len(labels), dtype=f"<U{label_width}")
    fold_summaries: list[dict[str, object]] = []
    for fold_number, (train_positions, test_positions) in enumerate(folds, start=1):
        train_groups = set(groups[train_positions].tolist())
        test_groups = set(groups[test_positions].tolist())
        if train_groups.intersection(test_groups):
            raise RuntimeError("Duplicate feature group leaked across a fold.")
        classifier = _fit_classifier(
            features[train_positions],
            labels[train_positions].tolist(),
            seed=seed + fold_number,
            epochs=epochs,
            learning_rate=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        predictions = classifier.predict(features[test_positions])
        out_of_fold_predictions[test_positions] = predictions
        fold_summaries.append(
            {
                "fold": fold_number,
                "train_sample_count": int(len(train_positions)),
                "test_sample_count": int(len(test_positions)),
                "test_class_counts": {
                    label: int(np.count_nonzero(labels[test_positions] == label))
                    for label in CLASS_LABELS
                },
                "duplicate_group_overlap": 0,
            }
        )

    metrics = calculate_classification_metrics(labels.tolist(), out_of_fold_predictions.tolist())
    final_classifier = _fit_classifier(
        features,
        labels.tolist(),
        seed=seed,
        epochs=epochs,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    index_hash = _sha256_file(index_path)
    _save_classifier(final_classifier, model_path, index_hash)

    class_counts = {
        label: int(np.count_nonzero(labels == label))
        for label in CLASS_LABELS
    }
    group_sizes = Counter(groups.tolist())
    repeated_group_count = sum(size > 1 for size in group_sizes.values())
    repeated_group_sample_count = sum(
        size for size in group_sizes.values() if size > 1
    )
    report: dict[str, object] = {
        "experiment": "Frozen MobileNetV3-Small feature linear classification",
        "target_label_source": "Dataset-provided folder labels; not independently validated diagnoses.",
        "index_path": str(index_path),
        "index_sha256": index_hash,
        "input_feature_dimension": FEATURE_DIMENSION,
        "sample_count": int(len(features)),
        "class_mapping": {str(position): label for position, label in enumerate(CLASS_LABELS)},
        "class_counts": class_counts,
        "index_model": {
            "model_id": index.model_id,
            "weight_id": index.weight_id,
            "preprocessing": index.preprocessing,
        },
        "feature_group_count": int(len(group_sizes)),
        "repeated_feature_group_count": int(repeated_group_count),
        "samples_in_repeated_feature_groups": int(repeated_group_sample_count),
        "split": {
            "method": "4-fold deterministic stratified group cross-validation",
            "stratification": "greedy class-wise assignment balancing fold sample counts",
            "grouping": "identical float32 feature vectors share a group",
            "seed": seed,
            "fold_count": n_splits,
            "all_samples_tested_once": True,
            "duplicate_groups_split_across_train_and_test": False,
            "folds": fold_summaries,
        },
        "training": {
            "framework": "PyTorch CPU",
            "architecture": "single linear layer, 576 inputs and four logits",
            "loss": "class-weighted cross entropy; weight = N / (4 * class_count)",
            "optimizer": "AdamW",
            "learning_rate": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "epochs": epochs,
            "feature_scaling": "standardization; mean and standard deviation fitted on each fold's training data only",
            "final_model_scaling": "fitted on all samples after evaluation for the saved model",
            "class_weighted_loss": True,
        },
        "metrics": metrics,
        "metric_definitions": {
            "accuracy": "correct predictions divided by all out-of-fold predictions",
            "balanced_accuracy": "mean recall across the four classes",
            "macro_precision": "unweighted mean of per-class precision",
            "macro_recall": "unweighted mean of per-class recall",
            "macro_f1": "unweighted mean of per-class F1",
            "zero_division": "0.0 when precision, recall, or F1 denominator is zero",
            "confusion_matrix": "rows are true labels and columns are predicted labels, in labels_in_order",
        },
        "saved_model": str(model_path.resolve()),
        "limitations": [
            "The FN class has only 12 samples, so each test fold contains about three FN samples; per-class estimates are highly uncertain.",
            "The same small dataset supplies all cross-validation folds; external generalization is not established.",
            "Labels come from source documentation and are not independently validated diagnoses.",
            "Exact duplicate copies are grouped by their identical feature vectors, but other related plants or capture sessions may still cross folds because group identifiers are unavailable.",
            "Predictions and any future confidence scores are not diagnostic certainty.",
        ],
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_available": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CPU linear classifier on frozen CBIR features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run group cross-validation and save model.")
    train_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    train_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        report = evaluate_and_train(
            args.index,
            args.model,
            args.report,
            n_splits=args.folds,
            seed=args.seed,
            epochs=args.epochs,
        )
        print(json.dumps(report["metrics"], indent=2))
        print(f"Saved model: {Path(args.model).resolve()}")
        print(f"Saved evaluation: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
