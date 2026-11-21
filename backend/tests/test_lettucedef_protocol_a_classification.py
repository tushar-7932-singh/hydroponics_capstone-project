import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.cbir import (
    CLASS_LABELS,
    FEATURE_DIMENSION,
    MODEL_ID,
    PREPROCESSING,
    WEIGHT_ID,
)
from app.lettucedef_classification import load_lettucedef_classifier
from app.lettucedef_index import CLASS_MAPPING
from app import lettucedef_protocol_a_classification as protocol_a

SOURCE_CLASSES = tuple(CLASS_MAPPING)
PROTOCOL_A_SPLITS = ("train", "val", "test", "train")
PROTOCOL_B_SPLITS = ("train", "train", "val", "test")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_dataset(
    tmp_path: Path,
    *,
    test_feature_offset: float = 0.0,
    index_metadata_updates: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, np.ndarray, list[dict[str, str]]]:
    manifest_a = tmp_path / "manifest_protocol_A.csv"
    manifest_b = tmp_path / "manifest_protocol_B.csv"
    index_path = tmp_path / "lettucedef-index.npz"
    manifest_a.parent.mkdir(parents=True, exist_ok=True)
    rows_a: list[dict[str, str]] = []
    rows_b: list[dict[str, str]] = []
    feature_rows: list[np.ndarray] = []

    for class_position, source_class in enumerate(SOURCE_CLASSES):
        for sample_position in range(4):
            filename = f"sample-{class_position}-{sample_position}.jpg"
            relative_path = f"{source_class}/{filename}"
            checksum = _sha256(relative_path.encode("utf-8"))
            base_row = {
                "file_name": filename,
                "class": source_class,
                "cultivar": "synthetic",
                "capture_date": "2025-04-10",
                "capture_time": "09:03:17",
                "bytes": "123",
                "sha256": checksum,
            }
            rows_a.append(
                {**base_row, "split": PROTOCOL_A_SPLITS[sample_position]}
            )
            rows_b.append(
                {**base_row, "split": PROTOCOL_B_SPLITS[sample_position]}
            )
            features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
            features[:4] = class_position * 3 + sample_position
            if PROTOCOL_A_SPLITS[sample_position] == "test":
                features += np.float32(test_feature_offset)
            feature_rows.append(features)

    for path, rows in ((manifest_a, rows_a), (manifest_b, rows_b)):
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest_b_hash = _sha256(manifest_b.read_bytes())
    metadata = {
        "dataset_name": "LettuceDeF",
        "dataset_version": "1.0.0",
        "source_manifest_path": "data/raw/LettuceDeF/manifest_protocol_B.csv",
        "source_manifest_sha256": manifest_b_hash,
        "model_id": MODEL_ID,
        "weight_id": WEIGHT_ID,
        "preprocessing": PREPROCESSING,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
    }
    if index_metadata_updates:
        metadata.update(index_metadata_updates)

    reverse_order = list(reversed(range(len(rows_b))))
    with index_path.open("wb") as target:
        np.savez_compressed(
            target,
            features=np.asarray(feature_rows, dtype=np.float32)[reverse_order],
            labels=np.asarray(
                [CLASS_MAPPING[row["class"]] for row in rows_b],
                dtype=np.str_,
            )[reverse_order],
            paths=np.asarray(
                [
                    "data/raw/LettuceDeF/extracted/images/"
                    f"{row['class']}/{row['file_name']}"
                    for row in rows_b
                ],
                dtype=np.str_,
            )[reverse_order],
            source_relative_paths=np.asarray(
                [
                    f"{row['class']}/{row['file_name']}" for row in rows_b
                ],
                dtype=np.str_,
            )[reverse_order],
            manifest_classes=np.asarray(
                [row["class"] for row in rows_b],
                dtype=np.str_,
            )[reverse_order],
            split_assignments=np.asarray(
                [row["split"] for row in rows_b],
                dtype=np.str_,
            )[reverse_order],
            image_sha256=np.asarray(
                [row["sha256"] for row in rows_b],
                dtype=np.str_,
            )[reverse_order],
            model_id=np.asarray(MODEL_ID),
            weight_id=np.asarray(WEIGHT_ID),
            preprocessing=np.asarray(PREPROCESSING),
            feature_dimension=np.asarray(FEATURE_DIMENSION, dtype=np.int64),
            metadata=np.asarray(json.dumps(metadata)),
        )
    return (
        index_path,
        manifest_a,
        manifest_b,
        np.asarray(feature_rows, dtype=np.float32),
        rows_a,
    )


def _use_synthetic_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
) -> str:
    digest = _sha256(manifest_path.read_bytes())
    monkeypatch.setattr(protocol_a, "PROTOCOL_A_MANIFEST_SHA256", digest)
    return digest


def _rewrite_index_metadata(index_path: Path, updates: dict[str, object]) -> None:
    with np.load(index_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata.update(updates)
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    with index_path.open("wb") as target:
        np.savez_compressed(target, **arrays)


def test_protocol_a_manifest_aligns_by_path_and_supplies_its_own_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, manifest_a, manifest_b, expected_features, rows_a = (
        _synthetic_dataset(tmp_path)
    )
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)

    aligned = protocol_a.load_and_align_protocol_a_dataset(
        index_path,
        manifest_a,
        index_source_manifest_path=manifest_b,
    )

    assert aligned.paths[0].endswith(
        f"{rows_a[0]['class']}/{rows_a[0]['file_name']}"
    )
    np.testing.assert_array_equal(aligned.features, expected_features)
    assert aligned.splits.tolist() == [
        row["split"] for row in rows_a
    ]
    assert aligned.splits.tolist() != PROTOCOL_B_SPLITS * len(SOURCE_CLASSES)


def test_protocol_b_manifest_cannot_be_used_as_protocol_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, manifest_a, manifest_b, _, _ = _synthetic_dataset(tmp_path)
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)

    with pytest.raises(ValueError, match="Protocol A is required"):
        protocol_a.load_and_align_protocol_a_dataset(
            index_path,
            manifest_b,
            index_source_manifest_path=manifest_b,
        )


def test_protocol_a_manifest_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, manifest_a, manifest_b, _, _ = _synthetic_dataset(tmp_path)
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)
    manifest_a.write_bytes(manifest_a.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Protocol A manifest SHA-256 mismatch"):
        protocol_a.load_and_align_protocol_a_dataset(
            index_path,
            manifest_a,
            index_source_manifest_path=manifest_b,
        )


def test_protocol_b_index_manifest_hash_is_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, manifest_a, manifest_b, _, _ = _synthetic_dataset(tmp_path)
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)
    manifest_b.write_bytes(manifest_b.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Protocol B manifest SHA-256 mismatch"):
        protocol_a.load_and_align_protocol_a_dataset(
            index_path,
            manifest_a,
            index_source_manifest_path=manifest_b,
        )


@pytest.mark.parametrize(
    ("field", "incorrect_value"),
    [
        ("model_id", "torchvision.models.mobilenet_v3_large"),
        ("weight_id", "MobileNet_V3_Large_Weights.DEFAULT"),
        ("preprocessing", "unverified preprocessing"),
        ("feature_dimension", 999),
        ("feature_dtype", "float64"),
    ],
)
def test_protocol_a_rejects_incorrect_index_encoder_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    incorrect_value: object,
) -> None:
    index_path, manifest_a, manifest_b, _, _ = _synthetic_dataset(
        tmp_path,
        index_metadata_updates={field: incorrect_value},
    )
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)

    with pytest.raises(ValueError, match=field):
        protocol_a.load_and_align_protocol_a_dataset(
            index_path,
            manifest_a,
            index_source_manifest_path=manifest_b,
        )


def test_protocol_a_test_features_do_not_change_checkpoint_or_scaler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _synthetic_dataset(tmp_path / "first")
    second = _synthetic_dataset(
        tmp_path / "second",
        test_feature_offset=10000.0,
    )
    index_a, manifest_a, manifest_b, _, _ = first
    index_b, manifest_b_a, manifest_b_b, _, _ = second
    _use_synthetic_manifest_hash(monkeypatch, manifest_a)
    _use_synthetic_manifest_hash(monkeypatch, manifest_b_a)
    first_model = tmp_path / "first-model.npz"
    first_report = tmp_path / "first-report.json"
    second_model = tmp_path / "second-model.npz"
    second_report = tmp_path / "second-report.json"

    first_result = protocol_a.train_and_evaluate(
        index_a,
        manifest_a,
        first_model,
        first_report,
        index_source_manifest_path=manifest_b,
        max_epochs=4,
        patience=4,
    )
    second_result = protocol_a.train_and_evaluate(
        index_b,
        manifest_b_a,
        second_model,
        second_report,
        index_source_manifest_path=manifest_b_b,
        max_epochs=4,
        patience=4,
    )
    first_classifier = load_lettucedef_classifier(first_model)
    second_classifier = load_lettucedef_classifier(second_model)

    np.testing.assert_array_equal(
        first_classifier.feature_mean,
        second_classifier.feature_mean,
    )
    np.testing.assert_array_equal(
        first_classifier.feature_scale,
        second_classifier.feature_scale,
    )
    np.testing.assert_array_equal(first_classifier.weights, second_classifier.weights)
    np.testing.assert_array_equal(first_classifier.bias, second_classifier.bias)
    assert (
        first_result["training"]["selected_epoch"]
        == second_result["training"]["selected_epoch"]
    )
    assert (
        first_result["selected_validation_metrics"]
        == second_result["selected_validation_metrics"]
    )


def test_protocol_a_training_saves_separate_artifacts_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, manifest_a, manifest_b, _, _ = _synthetic_dataset(tmp_path)
    manifest_hash = _use_synthetic_manifest_hash(monkeypatch, manifest_a)
    model_path = tmp_path / "lettucedef-protocol-a-linear-classifier.npz"
    report_path = tmp_path / "lettucedef-protocol-a-evaluation.json"

    report = protocol_a.train_and_evaluate(
        index_path,
        manifest_a,
        model_path,
        report_path,
        index_source_manifest_path=manifest_b,
        max_epochs=3,
        patience=3,
    )

    assert model_path.is_file()
    assert report_path.is_file()
    assert report["protocol"] == "Protocol A"
    assert report["manifest_sha256"] == manifest_hash
    assert report["split"]["protocol"] == "Protocol A"
    assert report["split"]["test_used_for_training"] is False
    assert report["split"]["test_used_for_model_selection"] is False
    assert set(report["test_metrics"]) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class",
        "confusion_matrix",
    }
    with np.load(model_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["protocol"] == "Protocol A"
    assert metadata["source_manifest_sha256"] == manifest_hash
    assert metadata["feature_index_sha256"] == report["index_sha256"]
    assert metadata["model_id"] == MODEL_ID
    assert metadata["weight_id"] == WEIGHT_ID
    assert metadata["feature_dimension"] == FEATURE_DIMENSION
    assert metadata["preprocessing"] == PREPROCESSING
    assert len(load_lettucedef_classifier(model_path).predict(
        np.zeros((1, FEATURE_DIMENSION), dtype=np.float32)
    )) == 1

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        protocol_a.train_and_evaluate(
            index_path,
            manifest_a,
            model_path,
            report_path,
            index_source_manifest_path=manifest_b,
            max_epochs=1,
        )


def test_protocol_a_model_and_report_defaults_are_separate() -> None:
    assert protocol_a.DEFAULT_MODEL_PATH.name == (
        "lettucedef-protocol-a-linear-classifier.npz"
    )
    assert protocol_a.DEFAULT_REPORT_PATH.name == (
        "lettucedef-protocol-a-evaluation.json"
    )
