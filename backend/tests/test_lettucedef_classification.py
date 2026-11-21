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
from app.lettucedef_classification import (
    DEFAULT_MODEL_PATH,
    PROTOCOL_B_MANIFEST_PATH,
    _fit_training_preprocessing,
    load_and_align_dataset,
    load_lettucedef_classifier,
    train_and_evaluate,
)
from app.lettucedef_index import CLASS_MAPPING

SOURCE_CLASSES = tuple(CLASS_MAPPING)


def _synthetic_dataset(
    tmp_path: Path,
    *,
    test_feature_offset: float = 0.0,
    protocol: str = "B",
    metadata_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, list[dict[str, str]], np.ndarray]:
    manifest_path = tmp_path / f"manifest_protocol_{protocol}.csv"
    index_path = tmp_path / "lettucedef-index.npz"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    feature_rows: list[np.ndarray] = []
    for class_position, source_class in enumerate(SOURCE_CLASSES):
        for sample_position, split in enumerate(("train", "train", "val", "test")):
            filename = f"sample-{class_position}-{sample_position}.jpg"
            relative_path = f"{source_class}/{filename}"
            checksum = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "file_name": filename,
                    "class": source_class,
                    "split": split,
                    "cultivar": "test-cultivar",
                    "capture_date": "2025-04-10",
                    "capture_time": "09:03:17",
                    "bytes": "123",
                    "sha256": checksum,
                }
            )
            features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
            features[:4] = class_position * 3 + sample_position
            if split == "test":
                features += np.float32(test_feature_offset)
            feature_rows.append(features)

    fieldnames = list(rows[0])
    with manifest_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reverse_order = list(reversed(range(len(rows))))
    metadata = {
        "dataset_name": "LettuceDeF",
        "source_manifest_path": (
            PROTOCOL_B_MANIFEST_PATH
            if protocol == "B"
            else "data/raw/LettuceDeF/manifest_protocol_A.csv"
        ),
        "source_manifest_sha256": manifest_sha256,
        "model_id": MODEL_ID,
        "weight_id": WEIGHT_ID,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
        "preprocessing": PREPROCESSING,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    with index_path.open("wb") as target:
        np.savez_compressed(
            target,
            features=np.asarray(feature_rows, dtype=np.float32)[reverse_order],
            labels=np.asarray(
                [CLASS_MAPPING[row["class"]] for row in rows],
                dtype=np.str_,
            )[reverse_order],
            paths=np.asarray(
                [
                    f"data/raw/LettuceDeF/extracted/images/"
                    f"{row['class']}/{row['file_name']}"
                    for row in rows
                ],
                dtype=np.str_,
            )[reverse_order],
            source_relative_paths=np.asarray(
                [
                    f"{row['class']}/{row['file_name']}"
                    for row in rows
                ],
                dtype=np.str_,
            )[reverse_order],
            manifest_classes=np.asarray(
                [row["class"] for row in rows],
                dtype=np.str_,
            )[reverse_order],
            split_assignments=np.asarray(
                [row["split"] for row in rows],
                dtype=np.str_,
            )[reverse_order],
            image_sha256=np.asarray(
                [row["sha256"] for row in rows],
                dtype=np.str_,
            )[reverse_order],
            model_id=np.asarray(MODEL_ID),
            weight_id=np.asarray(WEIGHT_ID),
            preprocessing=np.asarray(PREPROCESSING),
            feature_dimension=np.asarray(FEATURE_DIMENSION, dtype=np.int64),
            metadata=np.asarray(json.dumps(metadata)),
        )
    return index_path, manifest_path, rows, np.asarray(feature_rows, dtype=np.float32)


def _rewrite_index(
    index_path: Path,
    *,
    metadata_updates: dict[str, object] | None = None,
    array_updates: dict[str, np.ndarray] | None = None,
) -> None:
    with np.load(index_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    if metadata_updates:
        metadata.update(metadata_updates)
    if array_updates:
        arrays.update(array_updates)
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    with index_path.open("wb") as target:
        np.savez_compressed(target, **arrays)


@pytest.mark.parametrize(
    ("source_class", "expected"),
    [
        ("FN", "FN"),
        ("N-deficiency", "-N"),
        ("P-deficiency", "-P"),
        ("K-deficiency", "-K"),
    ],
)
def test_class_mapping(source_class: str, expected: str) -> None:
    assert CLASS_MAPPING[source_class] == expected


def test_manifest_alignment_uses_image_paths_not_index_order(tmp_path: Path) -> None:
    index_path, manifest_path, rows, expected_features = _synthetic_dataset(tmp_path)

    aligned = load_and_align_dataset(index_path, manifest_path)

    assert aligned.paths[0].endswith(f"{rows[0]['class']}/{rows[0]['file_name']}")
    np.testing.assert_array_equal(aligned.features, expected_features)
    assert aligned.labels.tolist() == [
        CLASS_MAPPING[row["class"]] for row in rows
    ]
    assert aligned.splits.tolist() == [row["split"] for row in rows]


def test_protocol_b_index_and_manifest_are_accepted(tmp_path: Path) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(tmp_path)

    aligned = load_and_align_dataset(index_path, manifest_path)

    assert aligned.source_manifest_path == PROTOCOL_B_MANIFEST_PATH
    assert aligned.model_id == MODEL_ID
    assert aligned.weight_id == WEIGHT_ID
    assert aligned.feature_dimension == FEATURE_DIMENSION
    assert aligned.feature_dtype == "float32"
    assert aligned.preprocessing == PREPROCESSING


def test_protocol_a_index_and_manifest_are_rejected(tmp_path: Path) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(
        tmp_path,
        protocol="A",
    )

    with pytest.raises(ValueError, match="Protocol B is required"):
        load_and_align_dataset(index_path, manifest_path)


def test_manifest_hash_must_match_index_metadata(tmp_path: Path) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Protocol B manifest SHA-256 mismatch"):
        load_and_align_dataset(index_path, manifest_path)


@pytest.mark.parametrize(
    ("field", "incorrect_value"),
    [
        ("model_id", "torchvision.models.mobilenet_v3_large"),
        ("weight_id", "MobileNet_V3_Large_Weights.DEFAULT"),
        ("feature_dimension", 999),
        ("feature_dtype", "float64"),
        ("preprocessing", "unverified preprocessing"),
    ],
)
def test_incorrect_encoder_configuration_is_rejected(
    tmp_path: Path,
    field: str,
    incorrect_value: object,
) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(
        tmp_path,
        metadata_overrides={field: incorrect_value},
    )

    with pytest.raises(ValueError, match=field):
        load_and_align_dataset(index_path, manifest_path)


@pytest.mark.parametrize(
    ("array_name", "incorrect_value"),
    [
        ("model_id", "wrong-model"),
        ("weight_id", "wrong-weights"),
        ("feature_dimension", np.asarray(575, dtype=np.int64)),
        ("preprocessing", "wrong-preprocessing"),
    ],
)
def test_top_level_encoder_configuration_must_match_metadata(
    tmp_path: Path,
    array_name: str,
    incorrect_value: object,
) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(tmp_path)
    replacement = np.asarray(incorrect_value)
    _rewrite_index(
        index_path,
        array_updates={array_name: replacement},
    )

    with pytest.raises(ValueError, match=array_name):
        load_and_align_dataset(index_path, manifest_path)


def test_training_preprocessing_and_class_weights_use_training_rows_only() -> None:
    features = np.zeros((7, FEATURE_DIMENSION), dtype=np.float32)
    features[:, 0] = [1.0, 3.0, 9.0, 13.0, 17.0, 1000.0, -1000.0]
    train_labels = ["FN", "FN", "-N", "-P", "-K"]

    mean, scale, class_weights, class_counts = _fit_training_preprocessing(
        features[:5],
        train_labels,
    )

    assert mean[0] == pytest.approx(np.mean([1.0, 3.0, 9.0, 13.0, 17.0]))
    assert scale[0] == pytest.approx(np.std([1.0, 3.0, 9.0, 13.0, 17.0]))
    assert class_counts == {
        "FN": 2,
        "-N": 1,
        "-P": 1,
        "-K": 1,
    }
    np.testing.assert_allclose(class_weights, [0.625, 1.25, 1.25, 1.25])


def test_manifest_class_labels_must_match_index(tmp_path: Path) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(tmp_path)
    with np.load(index_path, allow_pickle=False) as archive:
        labels = archive["labels"].copy()
    labels[0] = "FN" if labels[0] != "FN" else "-N"
    _rewrite_index(index_path, array_updates={"labels": labels})

    with pytest.raises(ValueError, match="Label mismatch"):
        load_and_align_dataset(index_path, manifest_path)


def test_test_split_features_do_not_change_checkpoint_or_preprocessing(
    tmp_path: Path,
) -> None:
    first_index, manifest_path, _, _ = _synthetic_dataset(tmp_path / "first")
    second_index, second_manifest, _, _ = _synthetic_dataset(
        tmp_path / "second",
        test_feature_offset=10000.0,
    )
    first_model = tmp_path / "first-model.npz"
    first_report = tmp_path / "first-report.json"
    second_model = tmp_path / "second-model.npz"
    second_report = tmp_path / "second-report.json"

    first_result = train_and_evaluate(
        first_index,
        manifest_path,
        first_model,
        first_report,
        max_epochs=4,
        patience=4,
    )
    second_result = train_and_evaluate(
        second_index,
        second_manifest,
        second_model,
        second_report,
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


def test_training_saves_separate_model_and_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    index_path, manifest_path, _, _ = _synthetic_dataset(tmp_path)
    model_path = tmp_path / "lettucedef-classifier.npz"
    report_path = tmp_path / "lettucedef-evaluation.json"

    report = train_and_evaluate(
        index_path,
        manifest_path,
        model_path,
        report_path,
        max_epochs=3,
        patience=3,
    )

    assert model_path.is_file()
    assert report_path.is_file()
    assert report["split"]["test_used_for_training"] is False
    assert report["split"]["test_used_for_model_selection"] is False
    assert report["protocol"] == "Protocol B"
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
        assert archive["weights"].shape == (4, FEATURE_DIMENSION)
        assert archive["feature_mean"].shape == (FEATURE_DIMENSION,)
        assert archive["classes"].tolist() == list(CLASS_LABELS)
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["dataset_name"] == "LettuceDeF"
        assert metadata["protocol"] == "Protocol B"
        assert metadata["source_manifest_path"] == PROTOCOL_B_MANIFEST_PATH
        assert metadata["model_id"] == MODEL_ID
        assert metadata["weight_id"] == WEIGHT_ID
        assert metadata["feature_dimension"] == FEATURE_DIMENSION
        assert metadata["feature_dtype"] == "float32"
        assert metadata["preprocessing"] == PREPROCESSING
    assert report["encoder"] == {
        "model_id": MODEL_ID,
        "weight_id": WEIGHT_ID,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
        "preprocessing": PREPROCESSING,
    }
    loaded = load_lettucedef_classifier(model_path)
    assert len(loaded.predict(np.zeros((1, FEATURE_DIMENSION), dtype=np.float32))) == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_and_evaluate(
            index_path,
            manifest_path,
            model_path,
            report_path,
            max_epochs=1,
        )


def test_default_model_is_separate_from_fnnpk_artifact() -> None:
    assert DEFAULT_MODEL_PATH.name == "lettucedef-linear-classifier.npz"
