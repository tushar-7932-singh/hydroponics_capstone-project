"""Train a separate manifest-driven classifier for the LettuceDeF dataset."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from app.cbir import (
    CLASS_LABELS,
    FEATURE_DIMENSION,
    MODEL_ID,
    PREPROCESSING,
    PROJECT_ROOT,
    WEIGHT_ID,
)
from app.classification import calculate_classification_metrics
from app.lettucedef_index import map_manifest_class

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lettucedef-mobilenet-v3-small.npz"
)
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "raw" / "LettuceDeF" / "manifest_protocol_B.csv"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lettucedef-linear-classifier.npz"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lettucedef-classification-evaluation.json"
)
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 0.0001
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_DATASET_NAME = "LettuceDeF"
PROTOCOL_B_NAME = "Protocol B"
PROTOCOL_B_MANIFEST_PATH = "data/raw/LettuceDeF/manifest_protocol_B.csv"
FEATURE_DTYPE = "float32"


@dataclass(frozen=True)
class AlignedDataset:
    features: np.ndarray
    labels: np.ndarray
    paths: tuple[str, ...]
    source_classes: np.ndarray
    splits: np.ndarray
    manifest_sha256: str
    source_manifest_path: str
    index_sha256: str
    model_id: str
    weight_id: str
    feature_dimension: int
    feature_dtype: str
    preprocessing: str


@dataclass(frozen=True)
class LinearFeatureClassifier:
    classes: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    index_sha256: str

    def predict(self, features: np.ndarray) -> list[str]:
        values = _validate_features(features)
        scaled = (values - self.feature_mean) / self.feature_scale
        logits = scaled @ self.weights.T + self.bias
        return [self.classes[position] for position in np.argmax(logits, axis=1)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _manifest_path_key(source_class: str, filename: str, row_number: int) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(
            f"Invalid file_name in manifest row {row_number}: {filename!r}."
        )
    return PurePosixPath(source_class, filename).as_posix()


def load_and_align_dataset(
    index_path: Path,
    manifest_path: Path,
) -> AlignedDataset:
    """Validate and align index rows to manifest records by source image path."""
    import csv

    index_path = Path(index_path).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if manifest_path.name != Path(PROTOCOL_B_MANIFEST_PATH).name:
        raise ValueError(
            "Protocol B is required: supply the manifest named "
            "'manifest_protocol_B.csv', not "
            f"'{manifest_path.name}'."
        )
    manifest_sha256 = _sha256_file(manifest_path)
    index_sha256 = _sha256_file(index_path)

    manifest_rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"file_name", "class", "split", "sha256"}
        if reader.fieldnames is None:
            raise ValueError("Manifest is empty or has no header.")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
                + "."
            )
        for row_number, row in enumerate(reader, start=2):
            source_class = (row.get("class") or "").strip()
            label = map_manifest_class(source_class)
            path_key = _manifest_path_key(
                source_class,
                (row.get("file_name") or "").strip(),
                row_number,
            )
            if path_key in manifest_rows:
                raise ValueError(f"Manifest contains duplicate image path '{path_key}'.")
            split = (row.get("split") or "").strip().lower()
            if split not in SPLIT_NAMES:
                raise ValueError(
                    f"Invalid split {split!r} in manifest row {row_number}; "
                    "expected train, val, or test."
                )
            checksum = (row.get("sha256") or "").strip().lower()
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError(
                    f"Invalid SHA-256 value in manifest row {row_number}."
                )
            manifest_rows[path_key] = {
                "label": label,
                "source_class": source_class,
                "split": split,
                "sha256": checksum,
            }

    if not manifest_rows:
        raise ValueError("Manifest contains no image rows.")

    try:
        with np.load(index_path, allow_pickle=False) as archive:
            required_arrays = {
                "features",
                "labels",
                "paths",
                "source_relative_paths",
                "manifest_classes",
                "split_assignments",
                "image_sha256",
                "metadata",
                "model_id",
                "weight_id",
                "preprocessing",
                "feature_dimension",
            }
            missing_arrays = required_arrays.difference(archive.files)
            if missing_arrays:
                raise ValueError(
                    "Feature index is missing required arrays: "
                    + ", ".join(sorted(missing_arrays))
                    + "."
                )
            features = np.asarray(archive["features"])
            labels = np.asarray(archive["labels"]).copy()
            project_paths = np.asarray(archive["paths"]).copy()
            relative_paths = np.asarray(archive["source_relative_paths"]).copy()
            source_classes = np.asarray(archive["manifest_classes"]).copy()
            split_assignments = np.asarray(archive["split_assignments"]).copy()
            image_sha256 = np.asarray(archive["image_sha256"]).copy()
            model_id = str(archive["model_id"].item())
            weight_id = str(archive["weight_id"].item())
            preprocessing = str(archive["preprocessing"].item())
            feature_dimension = int(archive["feature_dimension"].item())
            metadata = json.loads(str(archive["metadata"].item()))
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read feature index '{index_path}': {exc}") from exc

    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset_name") != EXPECTED_DATASET_NAME
    ):
        raise ValueError("Feature index metadata does not identify LettuceDeF.")
    if metadata.get("source_manifest_path") != PROTOCOL_B_MANIFEST_PATH:
        raise ValueError(
            "Protocol B is required: the feature index must record "
            f"'{PROTOCOL_B_MANIFEST_PATH}' as its source manifest. "
            "Rebuild the LettuceDeF index from manifest_protocol_B.csv."
        )
    if metadata.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError(
            "Protocol B manifest SHA-256 mismatch: the supplied manifest does not "
            "match the manifest recorded by the feature index. Supply the matching "
            "Protocol B CSV and rebuild the index if its source manifest changed."
        )
    expected_index_metadata = {
        "model_id": MODEL_ID,
        "weight_id": WEIGHT_ID,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": FEATURE_DTYPE,
        "preprocessing": PREPROCESSING,
    }
    recorded_index_metadata = {
        key: metadata.get(key) for key in expected_index_metadata
    }
    for key, expected_value in expected_index_metadata.items():
        if recorded_index_metadata[key] != expected_value:
            raise ValueError(
                f"Feature index metadata '{key}' is incompatible: expected "
                f"{expected_value!r}, got {recorded_index_metadata[key]!r}."
            )
    top_level_index_metadata = {
        "model_id": model_id,
        "weight_id": weight_id,
        "feature_dimension": feature_dimension,
        "preprocessing": preprocessing,
    }
    for key, expected_value in expected_index_metadata.items():
        if key == "feature_dtype":
            continue
        if top_level_index_metadata[key] != expected_value:
            raise ValueError(
                f"Feature index array '{key}' is incompatible: expected "
                f"{expected_value!r}, got {top_level_index_metadata[key]!r}."
            )
    if features.dtype != np.float32:
        raise ValueError("Feature index rows must have dtype float32.")
    features = _validate_features(features)
    array_fields = {
        "labels": labels,
        "paths": project_paths,
        "source_relative_paths": relative_paths,
        "manifest_classes": source_classes,
        "split_assignments": split_assignments,
        "image_sha256": image_sha256,
    }
    for name, values in array_fields.items():
        if values.ndim != 1 or len(values) != len(features):
            raise ValueError(
                f"Feature index array '{name}' must be one-dimensional and "
                "match the feature row count."
            )

    index_positions: dict[str, int] = {}
    for position, value in enumerate(relative_paths.tolist()):
        path_key = str(value).replace("\\", "/")
        if path_key in index_positions:
            raise ValueError(f"Feature index contains duplicate image path '{path_key}'.")
        index_positions[path_key] = position

    missing_paths = sorted(set(manifest_rows).difference(index_positions))
    unexpected_paths = sorted(set(index_positions).difference(manifest_rows))
    if missing_paths or unexpected_paths:
        details = []
        if missing_paths:
            details.append(f"missing feature rows for {len(missing_paths)} manifest paths")
        if unexpected_paths:
            details.append(f"{len(unexpected_paths)} feature paths absent from manifest")
        raise ValueError("Manifest/index path mismatch: " + "; ".join(details) + ".")

    ordered_positions: list[int] = []
    ordered_labels: list[str] = []
    ordered_source_classes: list[str] = []
    ordered_splits: list[str] = []
    ordered_project_paths: list[str] = []
    for path_key, manifest_entry in manifest_rows.items():
        position = index_positions[path_key]
        expected_project_path = (
            PurePosixPath(
                "data",
                "raw",
                "LettuceDeF",
                "extracted",
                "images",
                path_key,
            ).as_posix()
        )
        actual_label = str(labels[position])
        actual_class = str(source_classes[position])
        actual_split = str(split_assignments[position]).lower()
        actual_checksum = str(image_sha256[position]).lower()
        actual_path = str(project_paths[position]).replace("\\", "/")
        if actual_label != manifest_entry["label"]:
            raise ValueError(f"Label mismatch for manifest image '{path_key}'.")
        if actual_class != manifest_entry["source_class"]:
            raise ValueError(f"Original class mismatch for manifest image '{path_key}'.")
        if actual_split != manifest_entry["split"]:
            raise ValueError(f"Split mismatch for manifest image '{path_key}'.")
        if actual_checksum != manifest_entry["sha256"]:
            raise ValueError(f"Image SHA-256 mismatch for manifest image '{path_key}'.")
        if actual_path != expected_project_path:
            raise ValueError(f"Project-relative path mismatch for '{path_key}'.")
        ordered_positions.append(position)
        ordered_labels.append(manifest_entry["label"])
        ordered_source_classes.append(manifest_entry["source_class"])
        ordered_splits.append(manifest_entry["split"])
        ordered_project_paths.append(expected_project_path)

    return AlignedDataset(
        features=features[np.asarray(ordered_positions, dtype=np.int64)].copy(),
        labels=np.asarray(ordered_labels, dtype=np.str_),
        paths=tuple(ordered_project_paths),
        source_classes=np.asarray(ordered_source_classes, dtype=np.str_),
        splits=np.asarray(ordered_splits, dtype=np.str_),
        manifest_sha256=manifest_sha256,
        source_manifest_path=PROTOCOL_B_MANIFEST_PATH,
        index_sha256=index_sha256,
        model_id=model_id,
        weight_id=weight_id,
        feature_dimension=feature_dimension,
        feature_dtype=FEATURE_DTYPE,
        preprocessing=preprocessing,
    )


def _fit_training_preprocessing(
    features: np.ndarray,
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Fit scaler and inverse-frequency class weights on training rows only."""
    values = _validate_features(features)
    label_values = np.asarray(labels, dtype=np.str_)
    if label_values.ndim != 1 or len(label_values) != len(values):
        raise ValueError("Training labels must match the feature rows.")
    unknown = sorted(set(label_values.tolist()).difference(CLASS_LABELS))
    missing = [label for label in CLASS_LABELS if label not in label_values]
    if unknown:
        raise ValueError(f"Unknown training classes: {', '.join(unknown)}.")
    if missing:
        raise ValueError(
            "Training split is missing classes: " + ", ".join(missing) + "."
        )

    feature_mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-8] = 1.0
    counts = {
        label: int(np.count_nonzero(label_values == label))
        for label in CLASS_LABELS
    }
    class_weights = np.asarray(
        [
            len(label_values) / (len(CLASS_LABELS) * counts[label])
            for label in CLASS_LABELS
        ],
        dtype=np.float32,
    )
    return feature_mean, feature_scale, class_weights, counts


def _target_positions(labels: Sequence[str]) -> np.ndarray:
    label_to_position = {label: position for position, label in enumerate(CLASS_LABELS)}
    return np.asarray(
        [label_to_position[str(label)] for label in labels],
        dtype=np.int64,
    )


def _fit_with_validation(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    validation_features: np.ndarray,
    validation_labels: Sequence[str],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    index_sha256: str,
) -> tuple[
    LinearFeatureClassifier,
    int,
    list[dict[str, object]],
    dict[str, int],
    np.ndarray,
]:
    import torch
    from torch import nn

    feature_mean, feature_scale, class_weights, class_counts = (
        _fit_training_preprocessing(train_features, train_labels)
    )
    train_standardized = (
        (train_features - feature_mean) / feature_scale
    ).astype(np.float32)
    train_inputs = torch.from_numpy(np.ascontiguousarray(train_standardized))
    train_targets = torch.from_numpy(_target_positions(train_labels))
    validation_standardized = (
        (validation_features - feature_mean) / feature_scale
    ).astype(np.float32)
    validation_inputs = torch.from_numpy(np.ascontiguousarray(validation_standardized))
    validation_targets = torch.from_numpy(_target_positions(validation_labels))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Linear(FEATURE_DIMENSION, len(CLASS_LABELS), device="cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        loss_function = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights.copy())
        )
        history: list[dict[str, object]] = []
        best_key: tuple[float, float] | None = None
        best_state: tuple[np.ndarray, np.ndarray] | None = None
        best_epoch = 0
        best_metrics: dict[str, object] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_inputs), train_targets)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                validation_logits = model(validation_inputs)
                validation_loss = float(
                    nn.functional.cross_entropy(
                        validation_logits,
                        validation_targets,
                    ).item()
                )
                predicted_positions = validation_logits.argmax(dim=1).cpu().numpy()
            validation_predictions = [
                CLASS_LABELS[int(position)] for position in predicted_positions
            ]
            validation_metrics = calculate_classification_metrics(
                validation_labels,
                validation_predictions,
            )
            macro_f1 = float(validation_metrics["macro_f1"])
            selection_key = (macro_f1, -validation_loss)
            history.append(
                {
                    "epoch": epoch,
                    "training_weighted_cross_entropy": float(loss.item()),
                    "validation_cross_entropy": validation_loss,
                    "validation_metrics": validation_metrics,
                }
            )

            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_epoch = epoch
                best_metrics = validation_metrics
                best_state = (
                    model.weight.detach().cpu().numpy().copy(),
                    model.bias.detach().cpu().numpy().copy(),
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    classifier = LinearFeatureClassifier(
        classes=CLASS_LABELS,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=best_state[0],
        bias=best_state[1],
        index_sha256=index_sha256,
    )
    return classifier, best_epoch, history, class_counts, class_weights


def _split_counts(
    labels: np.ndarray,
    splits: np.ndarray,
) -> dict[str, dict[str, int]]:
    return {
        split: {
            label: int(np.count_nonzero((splits == split) & (labels == label)))
            for label in CLASS_LABELS
        }
        for split in SPLIT_NAMES
    }


def _validate_configuration(
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs < 1
    ):
        raise ValueError("max_epochs must be a positive integer.")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")


def _artifact_paths(
    index_path: Path,
    manifest_path: Path,
    model_path: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    model_path = Path(model_path).resolve()
    report_path = Path(report_path).resolve()
    protected_paths = {Path(index_path).resolve(), Path(manifest_path).resolve()}
    if (
        model_path == report_path
        or model_path in protected_paths
        or report_path in protected_paths
    ):
        raise ValueError(
            "Model and report paths must be distinct from each other and inputs."
        )
    for output_path in (model_path, report_path):
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite existing LettuceDeF artifact: '{output_path}'."
            )
    return model_path, report_path


def _write_artifacts(
    classifier: LinearFeatureClassifier,
    model_path: Path,
    report_path: Path,
    model_metadata: dict[str, object],
    report: dict[str, object],
) -> None:
    import io

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_buffer = io.BytesIO()
    np.savez_compressed(
        model_buffer,
        metadata=np.asarray(json.dumps(model_metadata, sort_keys=True)),
        classes=np.asarray(classifier.classes, dtype=np.str_),
        feature_mean=classifier.feature_mean,
        feature_scale=classifier.feature_scale,
        weights=classifier.weights,
        bias=classifier.bias,
    )
    model_bytes = model_buffer.getvalue()
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode("utf-8")
    created: list[Path] = []
    try:
        with model_path.open("xb") as target:
            created.append(model_path)
            target.write(model_bytes)
        with report_path.open("xb") as target:
            created.append(report_path)
            target.write(report_bytes)
    except OSError:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise


def train_and_evaluate(
    index_path: Path = DEFAULT_INDEX_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> dict[str, object]:
    """Fit on Protocol B train, select by validation, and evaluate test once."""
    _validate_configuration(seed, max_epochs, patience, learning_rate, weight_decay)
    model_path, report_path = _artifact_paths(
        index_path,
        manifest_path,
        model_path,
        report_path,
    )
    dataset = load_and_align_dataset(index_path, manifest_path)
    split_positions = {
        split: np.flatnonzero(dataset.splits == split)
        for split in SPLIT_NAMES
    }
    for split, positions in split_positions.items():
        if len(positions) == 0:
            raise ValueError(f"Manifest split '{split}' contains no images.")
        missing = [
            label
            for label in CLASS_LABELS
            if not np.any(dataset.labels[positions] == label)
        ]
        if missing:
            raise ValueError(
                f"Manifest split '{split}' is missing classes: "
                + ", ".join(missing)
                + "."
            )

    train_positions = split_positions["train"]
    validation_positions = split_positions["val"]
    test_positions = split_positions["test"]
    classifier, selected_epoch, history, training_class_counts, class_weights = (
        _fit_with_validation(
            dataset.features[train_positions],
            dataset.labels[train_positions].tolist(),
            dataset.features[validation_positions],
            dataset.labels[validation_positions].tolist(),
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            index_sha256=dataset.index_sha256,
        )
    )
    validation_predictions = classifier.predict(
        dataset.features[validation_positions]
    )
    validation_metrics = calculate_classification_metrics(
        dataset.labels[validation_positions].tolist(),
        validation_predictions,
    )

    # Test rows are first used after the selected validation checkpoint is fixed.
    test_predictions = classifier.predict(dataset.features[test_positions])
    test_metrics = calculate_classification_metrics(
        dataset.labels[test_positions].tolist(),
        test_predictions,
    )
    class_mapping = {
        str(position): label for position, label in enumerate(CLASS_LABELS)
    }
    training_configuration = {
        "framework": "PyTorch CPU",
        "architecture": "torch.nn.Linear(576, 4)",
        "seed": seed,
        "max_epochs": max_epochs,
        "selected_epoch": selected_epoch,
        "early_stopping_patience": patience,
        "selection_metric": "validation macro F1; ties use lower validation cross-entropy",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss": "class-weighted cross entropy",
        "class_weight_formula": "N_train / (4 * class_count_train)",
        "feature_scaling": "per-feature mean and standard deviation fitted on train split only",
        "validation_use": "checkpoint and epoch selection only",
        "test_use": "one evaluation after model selection; not used for fitting or selection",
    }
    split_counts = _split_counts(dataset.labels, dataset.splits)
    model_metadata: dict[str, object] = {
        "format": "numpy-npz-linear-pytorch-state",
        "dataset_name": EXPECTED_DATASET_NAME,
        "protocol": PROTOCOL_B_NAME,
        "source_manifest_path": dataset.source_manifest_path,
        "source_manifest_sha256": dataset.manifest_sha256,
        "feature_index_sha256": dataset.index_sha256,
        "model_id": dataset.model_id,
        "weight_id": dataset.weight_id,
        "input_feature_dimension": FEATURE_DIMENSION,
        "feature_dimension": dataset.feature_dimension,
        "feature_dtype": dataset.feature_dtype,
        "preprocessing": dataset.preprocessing,
        "output_class_count": len(CLASS_LABELS),
        "class_mapping": class_mapping,
        "architecture": "torch.nn.Linear(576, 4)",
        "feature_scaling": training_configuration["feature_scaling"],
        "selected_epoch": selected_epoch,
        "training_class_counts": training_class_counts,
        "training_class_weights": {
            label: float(class_weights[position])
            for position, label in enumerate(CLASS_LABELS)
        },
    }
    report: dict[str, object] = {
        "experiment": "LettuceDeF frozen MobileNetV3-Small feature classification",
        "target_label_source": (
            "Manifest-provided treatment classes; these are classification metrics "
            "against source labels, not diagnostic accuracy."
        ),
        "dataset_name": EXPECTED_DATASET_NAME,
        "protocol": PROTOCOL_B_NAME,
        "index_path": str(Path(index_path).resolve()),
        "index_sha256": dataset.index_sha256,
        "manifest_path": str(Path(manifest_path).resolve()),
        "source_manifest_path": dataset.source_manifest_path,
        "manifest_sha256": dataset.manifest_sha256,
        "encoder": {
            "model_id": dataset.model_id,
            "weight_id": dataset.weight_id,
            "feature_dimension": dataset.feature_dimension,
            "feature_dtype": dataset.feature_dtype,
            "preprocessing": dataset.preprocessing,
        },
        "input_feature_dimension": FEATURE_DIMENSION,
        "sample_count": len(dataset.labels),
        "class_mapping": class_mapping,
        "split": {
            "protocol": PROTOCOL_B_NAME,
            "assignments": {
                name: int(len(split_positions[name])) for name in SPLIT_NAMES
            },
            "class_counts": split_counts,
            "feature_paths_aligned_by": "source_relative_paths (class/file_name), not array row order",
            "all_manifest_paths_match_exactly_one_feature": True,
            "test_used_for_fit_preprocessing": False,
            "test_used_for_training": False,
            "test_used_for_model_selection": False,
        },
        "training": {
            **training_configuration,
            "training_class_counts": training_class_counts,
            "training_class_weights": model_metadata["training_class_weights"],
        },
        "selected_validation_metrics": validation_metrics,
        "validation_history": history,
        "test_metrics": test_metrics,
        "metric_definitions": {
            "accuracy": "correct predictions divided by all predictions",
            "balanced_accuracy": "mean recall across the four classes",
            "macro_precision": "unweighted mean of per-class precision",
            "macro_recall": "unweighted mean of per-class recall",
            "macro_f1": "unweighted mean of per-class F1",
            "per_class_support": "number of true samples for that class",
            "confusion_matrix": "rows are true labels and columns are predicted labels, in labels_in_order",
            "zero_division": "0.0 when a precision, recall, or F1 denominator is zero",
        },
        "saved_model": str(model_path),
        "limitations": [
            "The target labels are source-provided nutrient-treatment classes, not independently confirmed diagnoses.",
            "Protocol B does not establish plant-independent or independent-greenhouse generalization; stable plant identifiers and separate crop-cycle data are unavailable.",
            "Test performance is an estimate on this dataset and is not diagnostic accuracy.",
        ],
    }
    _write_artifacts(
        classifier,
        model_path,
        report_path,
        model_metadata,
        report,
    )
    return report


def load_lettucedef_classifier(path: Path) -> LinearFeatureClassifier:
    """Load the separate array-only LettuceDeF classifier artifact."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            classes = tuple(str(label) for label in archive["classes"].tolist())
            feature_mean = np.asarray(archive["feature_mean"], dtype=np.float32).copy()
            feature_scale = np.asarray(archive["feature_scale"], dtype=np.float32).copy()
            weights = np.asarray(archive["weights"], dtype=np.float32).copy()
            bias = np.asarray(archive["bias"], dtype=np.float32).copy()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read LettuceDeF classifier artifact: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Classifier metadata must be a JSON object.")
    if metadata.get("dataset_name") != EXPECTED_DATASET_NAME:
        raise ValueError("Classifier artifact does not identify LettuceDeF.")
    if metadata.get("input_feature_dimension") != FEATURE_DIMENSION:
        raise ValueError("Classifier artifact has an incompatible feature dimension.")
    if classes != CLASS_LABELS or metadata.get("class_mapping") != {
        str(position): label for position, label in enumerate(CLASS_LABELS)
    }:
        raise ValueError("Classifier artifact has an incompatible class mapping.")
    index_sha256 = metadata.get("feature_index_sha256")
    if (
        not isinstance(index_sha256, str)
        or len(index_sha256) != 64
        or any(character not in "0123456789abcdef" for character in index_sha256)
    ):
        raise ValueError("Classifier artifact has an invalid feature-index SHA-256.")
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        report = train_and_evaluate(
            args.index,
            args.manifest,
            args.model,
            args.report,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Saved LettuceDeF classifier: {Path(args.model).resolve()}")
    print(f"Saved LettuceDeF evaluation: {Path(args.report).resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the separate LettuceDeF linear classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Fit on train, select on validation, and evaluate test once.",
    )
    train_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    train_parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser


if __name__ == "__main__":
    main()
