"""Train a separate classifier using the LettuceDeF Protocol A split."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from app.cbir import CLASS_LABELS, PROJECT_ROOT
from app.classification import calculate_classification_metrics
from app.lettucedef_classification import (
    DEFAULT_INDEX_PATH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    LinearFeatureClassifier,
    _artifact_paths,
    _fit_with_validation,
    _sha256_file,
    _split_counts,
    _validate_configuration,
    _write_artifacts,
    load_and_align_dataset,
)
from app.lettucedef_index import map_manifest_class

DEFAULT_PROTOCOL_A_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "raw" / "LettuceDeF" / "manifest_protocol_A.csv"
)
DEFAULT_PROTOCOL_B_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "raw" / "LettuceDeF" / "manifest_protocol_B.csv"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lettucedef-protocol-a-linear-classifier.npz"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lettucedef-protocol-a-evaluation.json"
)
PROTOCOL_A_NAME = "Protocol A"
PROTOCOL_A_MANIFEST_NAME = "manifest_protocol_A.csv"
PROTOCOL_A_MANIFEST_PATH = "data/raw/LettuceDeF/manifest_protocol_A.csv"
PROTOCOL_A_MANIFEST_SHA256 = (
    "26ee4f6cac887fc23c08499f1b1eb2647c1f585079aed95d553380fa7f6785d0"
)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ProtocolAManifestEntry:
    source_class: str
    label: str
    split: str
    sha256: str


@dataclass(frozen=True)
class ProtocolAAlignedDataset:
    features: np.ndarray
    labels: np.ndarray
    paths: tuple[str, ...]
    source_classes: np.ndarray
    splits: np.ndarray
    manifest_sha256: str
    manifest_path: str
    index_sha256: str
    index_source_manifest_path: str
    index_source_manifest_sha256: str
    model_id: str
    weight_id: str
    feature_dimension: int
    feature_dtype: str
    preprocessing: str


def _read_manifest(
    manifest_path: Path,
    *,
    expected_filename: str,
) -> dict[str, ProtocolAManifestEntry]:
    if manifest_path.name != expected_filename:
        raise ValueError(
            f"{expected_filename} is required; received '{manifest_path.name}'."
        )

    entries: dict[str, ProtocolAManifestEntry] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"file_name", "class", "split", "sha256"}
        if reader.fieldnames is None:
            raise ValueError(f"Manifest '{manifest_path}' is empty or has no header.")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Manifest '{manifest_path}' is missing required columns: "
                + ", ".join(sorted(missing_columns))
                + "."
            )
        for row_number, row in enumerate(reader, start=2):
            source_class = (row.get("class") or "").strip()
            label = map_manifest_class(source_class)
            filename = (row.get("file_name") or "").strip()
            if (
                not filename
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
            ):
                raise ValueError(
                    f"Invalid file_name in manifest row {row_number}: {filename!r}."
                )
            path_key = PurePosixPath(source_class, filename).as_posix()
            if path_key in entries:
                raise ValueError(f"Manifest contains duplicate path '{path_key}'.")
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
            entries[path_key] = ProtocolAManifestEntry(
                source_class=source_class,
                label=label,
                split=split,
                sha256=checksum,
            )
    if not entries:
        raise ValueError(f"Manifest '{manifest_path}' contains no image rows.")
    return entries


def load_and_align_protocol_a_dataset(
    index_path: Path = DEFAULT_INDEX_PATH,
    manifest_path: Path = DEFAULT_PROTOCOL_A_MANIFEST_PATH,
    *,
    index_source_manifest_path: Path = DEFAULT_PROTOCOL_B_MANIFEST_PATH,
) -> ProtocolAAlignedDataset:
    """Validate the Protocol B-built index, then apply Protocol A assignments."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    index_source_manifest_path = Path(index_source_manifest_path).resolve(strict=True)
    if manifest_path.name != PROTOCOL_A_MANIFEST_NAME:
        raise ValueError(
            "Protocol A is required: supply the manifest named "
            f"'{PROTOCOL_A_MANIFEST_NAME}', not '{manifest_path.name}'."
        )
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != PROTOCOL_A_MANIFEST_SHA256:
        raise ValueError(
            "Protocol A manifest SHA-256 mismatch: expected the verified "
            f"Protocol A manifest ({PROTOCOL_A_MANIFEST_SHA256}), got "
            f"{manifest_sha256}. Restore the original manifest before training."
        )
    protocol_a_entries = _read_manifest(
        manifest_path,
        expected_filename=PROTOCOL_A_MANIFEST_NAME,
    )

    # The existing feature index records Protocol B as its source. Validate that
    # provenance and the encoder configuration before applying Protocol A splits.
    indexed = load_and_align_dataset(index_path, index_source_manifest_path)
    protocol_b_entries = _read_manifest(
        index_source_manifest_path,
        expected_filename="manifest_protocol_B.csv",
    )
    if set(protocol_a_entries) != set(protocol_b_entries):
        raise ValueError(
            "Protocol A and the feature-index source manifest contain different "
            "image paths."
        )

    index_positions: dict[str, int] = {}
    for position, project_path in enumerate(indexed.paths):
        path_parts = PurePosixPath(project_path).parts
        if len(path_parts) < 2:
            raise ValueError(
                f"Invalid project-relative feature path: '{project_path}'."
            )
        path_key = PurePosixPath(*path_parts[-2:]).as_posix()
        if path_key in index_positions:
            raise ValueError(f"Feature index contains duplicate path '{path_key}'.")
        index_positions[path_key] = position

    ordered_positions: list[int] = []
    ordered_labels: list[str] = []
    ordered_source_classes: list[str] = []
    ordered_splits: list[str] = []
    ordered_paths: list[str] = []
    for path_key, protocol_a_entry in protocol_a_entries.items():
        protocol_b_entry = protocol_b_entries[path_key]
        position = index_positions.get(path_key)
        if position is None:
            raise ValueError(
                f"Protocol A image '{path_key}' has no matching feature vector."
            )
        if (
            protocol_a_entry.source_class != protocol_b_entry.source_class
            or protocol_a_entry.label != protocol_b_entry.label
        ):
            raise ValueError(f"Class mismatch between manifests for '{path_key}'.")
        if protocol_a_entry.sha256 != protocol_b_entry.sha256:
            raise ValueError(f"Image SHA-256 mismatch between manifests for '{path_key}'.")
        ordered_positions.append(position)
        ordered_labels.append(protocol_a_entry.label)
        ordered_source_classes.append(protocol_a_entry.source_class)
        ordered_splits.append(protocol_a_entry.split)
        ordered_paths.append(indexed.paths[position])

    if len(index_positions) != len(protocol_a_entries):
        raise ValueError(
            "Protocol A manifest and feature index do not contain the same "
            "number of unique image paths."
        )

    return ProtocolAAlignedDataset(
        features=indexed.features[np.asarray(ordered_positions, dtype=np.int64)].copy(),
        labels=np.asarray(ordered_labels, dtype=np.str_),
        paths=tuple(ordered_paths),
        source_classes=np.asarray(ordered_source_classes, dtype=np.str_),
        splits=np.asarray(ordered_splits, dtype=np.str_),
        manifest_sha256=manifest_sha256,
        manifest_path=PROTOCOL_A_MANIFEST_PATH,
        index_sha256=indexed.index_sha256,
        index_source_manifest_path=indexed.source_manifest_path,
        index_source_manifest_sha256=indexed.manifest_sha256,
        model_id=indexed.model_id,
        weight_id=indexed.weight_id,
        feature_dimension=indexed.feature_dimension,
        feature_dtype=indexed.feature_dtype,
        preprocessing=indexed.preprocessing,
    )


def _validate_output_paths(
    index_path: Path,
    manifest_path: Path,
    index_source_manifest_path: Path,
    model_path: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    model_path = Path(model_path).resolve()
    report_path = Path(report_path).resolve()
    index_source_manifest_path = Path(index_source_manifest_path).resolve()
    if (
        model_path == index_source_manifest_path
        or report_path == index_source_manifest_path
    ):
        raise ValueError(
            "Protocol A output paths must not overwrite the Protocol B source manifest."
        )
    return _artifact_paths(index_path, manifest_path, model_path, report_path)


def _require_all_classes(dataset: ProtocolAAlignedDataset) -> dict[str, np.ndarray]:
    split_positions = {
        split: np.flatnonzero(dataset.splits == split)
        for split in SPLIT_NAMES
    }
    for split, positions in split_positions.items():
        if len(positions) == 0:
            raise ValueError(f"Protocol A split '{split}' contains no images.")
        missing = [
            label
            for label in CLASS_LABELS
            if not np.any(dataset.labels[positions] == label)
        ]
        if missing:
            raise ValueError(
                f"Protocol A split '{split}' is missing classes: "
                + ", ".join(missing)
                + "."
            )
    return split_positions


def train_and_evaluate(
    index_path: Path = DEFAULT_INDEX_PATH,
    manifest_path: Path = DEFAULT_PROTOCOL_A_MANIFEST_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    index_source_manifest_path: Path = DEFAULT_PROTOCOL_B_MANIFEST_PATH,
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> dict[str, object]:
    """Train on Protocol A train, select on validation, evaluate test once."""
    _validate_configuration(seed, max_epochs, patience, learning_rate, weight_decay)
    model_path, report_path = _validate_output_paths(
        index_path,
        manifest_path,
        index_source_manifest_path,
        model_path,
        report_path,
    )
    dataset = load_and_align_protocol_a_dataset(
        index_path,
        manifest_path,
        index_source_manifest_path=index_source_manifest_path,
    )
    split_positions = _require_all_classes(dataset)
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
    validation_metrics = calculate_classification_metrics(
        dataset.labels[validation_positions].tolist(),
        classifier.predict(dataset.features[validation_positions]),
    )

    test_metrics = calculate_classification_metrics(
        dataset.labels[test_positions].tolist(),
        classifier.predict(dataset.features[test_positions]),
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
        "feature_scaling": "per-feature mean and standard deviation fitted on Protocol A train only",
        "validation_use": "checkpoint and epoch selection only",
        "test_use": "one evaluation after model selection; not used for fitting or selection",
    }
    encoder_metadata = {
        "model_id": dataset.model_id,
        "weight_id": dataset.weight_id,
        "feature_dimension": dataset.feature_dimension,
        "feature_dtype": dataset.feature_dtype,
        "preprocessing": dataset.preprocessing,
    }
    model_metadata: dict[str, object] = {
        "format": "numpy-npz-linear-pytorch-state",
        "dataset_name": "LettuceDeF",
        "protocol": PROTOCOL_A_NAME,
        "source_manifest_path": dataset.manifest_path,
        "source_manifest_sha256": dataset.manifest_sha256,
        "index_source_manifest_path": dataset.index_source_manifest_path,
        "index_source_manifest_sha256": dataset.index_source_manifest_sha256,
        "feature_index_sha256": dataset.index_sha256,
        **encoder_metadata,
        "input_feature_dimension": dataset.feature_dimension,
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
        "dataset_name": "LettuceDeF",
        "protocol": PROTOCOL_A_NAME,
        "index_path": str(Path(index_path).resolve()),
        "index_sha256": dataset.index_sha256,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": dataset.manifest_sha256,
        "index_source_manifest_path": dataset.index_source_manifest_path,
        "index_source_manifest_sha256": dataset.index_source_manifest_sha256,
        "encoder": encoder_metadata,
        "input_feature_dimension": dataset.feature_dimension,
        "sample_count": len(dataset.labels),
        "class_mapping": class_mapping,
        "split": {
            "protocol": PROTOCOL_A_NAME,
            "assignments": {
                name: int(len(split_positions[name])) for name in SPLIT_NAMES
            },
            "class_counts": _split_counts(dataset.labels, dataset.splits),
            "feature_paths_aligned_by": (
                "source_relative_paths (class/file_name), not array row order"
            ),
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
            "confusion_matrix": (
                "rows are true labels and columns are predicted labels, "
                "in labels_in_order"
            ),
            "zero_division": "0.0 when a precision, recall, or F1 denominator is zero",
        },
        "saved_model": str(model_path),
        "limitations": [
            "The target labels are source-provided nutrient-treatment classes, not independently confirmed diagnoses.",
            "Protocol A does not establish plant-independent or independent-greenhouse generalization; stable plant identifiers and separate crop-cycle data are unavailable.",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the separate LettuceDeF Protocol A classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Fit on Protocol A train, select on validation, and evaluate test once.",
    )
    train_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    train_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_A_MANIFEST_PATH,
    )
    train_parser.add_argument(
        "--index-source-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_B_MANIFEST_PATH,
        help="Protocol B manifest used when the shared feature index was built.",
    )
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    train_parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        report = train_and_evaluate(
            args.index,
            args.manifest,
            args.model,
            args.report,
            index_source_manifest_path=args.index_source_manifest,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Saved Protocol A classifier: {Path(args.model).resolve()}")
    print(f"Saved Protocol A evaluation: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
