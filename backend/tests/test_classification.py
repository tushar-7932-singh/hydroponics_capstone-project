from pathlib import Path

import numpy as np
import pytest

from app.cbir import CLASS_LABELS, FEATURE_DIMENSION, MODEL_ID, PREPROCESSING, WEIGHT_ID
from app.classification import (
    LinearFeatureClassifier,
    calculate_classification_metrics,
    evaluate_and_train,
    feature_group_ids,
    load_classifier,
    stratified_group_folds,
)


def test_class_mapping_and_prediction() -> None:
    classifier = LinearFeatureClassifier(
        classes=CLASS_LABELS,
        feature_mean=np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIMENSION, dtype=np.float32),
        weights=np.zeros((len(CLASS_LABELS), FEATURE_DIMENSION), dtype=np.float32),
        bias=np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float32),
    )

    assert classifier.classes == ("FN", "-N", "-P", "-K")
    assert classifier.predict(np.zeros((2, FEATURE_DIMENSION), dtype=np.float32)) == [
        "-P",
        "-P",
    ]


def test_feature_group_ids_combine_identical_rows() -> None:
    features = np.zeros((4, FEATURE_DIMENSION), dtype=np.float32)
    features[1, 0] = 1
    features[2] = features[0]
    features[3, 0] = 2

    groups = feature_group_ids(features)

    assert groups[0] == groups[2]
    assert groups[0] != groups[1]
    assert groups[1] != groups[3]


def test_stratified_group_folds_prevent_group_leakage_and_cover_each_sample() -> None:
    labels = ["FN"] * 8 + ["-N"] * 8 + ["-P"] * 8 + ["-K"] * 8
    groups = np.arange(len(labels))
    groups[1] = groups[0]
    groups[9] = groups[8]

    folds = stratified_group_folds(labels, groups, n_splits=4, seed=7)
    test_positions = []
    for train_positions, test_positions_for_fold in folds:
        assert not set(groups[train_positions]).intersection(groups[test_positions_for_fold])
        test_positions.extend(test_positions_for_fold.tolist())

    assert sorted(test_positions) == list(range(len(labels)))
    repeated_folds = stratified_group_folds(labels, groups, n_splits=4, seed=7)
    assert len(repeated_folds) == len(folds)
    for (train, test), (repeat_train, repeat_test) in zip(folds, repeated_folds):
        np.testing.assert_array_equal(train, repeat_train)
        np.testing.assert_array_equal(test, repeat_test)


def test_metrics_include_per_class_macro_and_confusion_matrix() -> None:
    metrics = calculate_classification_metrics(
        ["FN", "FN", "-N", "-P", "-K"],
        ["FN", "-N", "-N", "-K", "-K"],
    )

    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.625)
    assert metrics["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.5) / 4)
    assert metrics["per_class"]["FN"]["recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["rows_true_columns_predicted"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]


@pytest.mark.parametrize(
    "features",
    [
        np.zeros((1, 3), dtype=np.float32),
        np.full((1, FEATURE_DIMENSION), np.nan, dtype=np.float32),
        np.zeros((0, FEATURE_DIMENSION), dtype=np.float32),
    ],
)
def test_prediction_rejects_invalid_features(features: np.ndarray) -> None:
    classifier = LinearFeatureClassifier(
        classes=CLASS_LABELS,
        feature_mean=np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIMENSION, dtype=np.float32),
        weights=np.zeros((len(CLASS_LABELS), FEATURE_DIMENSION), dtype=np.float32),
        bias=np.zeros(len(CLASS_LABELS), dtype=np.float32),
    )

    with pytest.raises(ValueError):
        classifier.predict(features)


def test_evaluate_and_train_saves_model_and_report_without_duplicate_leakage(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(12)
    labels = []
    feature_rows = []
    paths = []
    class_centers = rng.normal(size=(len(CLASS_LABELS), FEATURE_DIMENSION)).astype(np.float32)
    duplicate_pairs = []
    for class_position, label in enumerate(CLASS_LABELS):
        for sample_position in range(8):
            row = class_centers[class_position] + rng.normal(
                scale=0.1,
                size=FEATURE_DIMENSION,
            ).astype(np.float32)
            if sample_position == 1:
                row = feature_rows[-1].copy()
                duplicate_pairs.append((len(feature_rows) - 1, len(feature_rows)))
            feature_rows.append(row)
            labels.append(label)
            paths.append(f"data/raw/FNNPK/{label}/sample-{sample_position}.png")

    index_path = tmp_path / "index.npz"
    with index_path.open("wb") as target:
        np.savez_compressed(
            target,
            index_version=np.asarray(1),
            features=np.asarray(feature_rows, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.str_),
            paths=np.asarray(paths, dtype=np.str_),
            model_id=np.asarray(MODEL_ID),
            weight_id=np.asarray(WEIGHT_ID),
            preprocessing=np.asarray(PREPROCESSING),
            feature_dimension=np.asarray(FEATURE_DIMENSION),
        )

    model_path = tmp_path / "classifier.npz"
    report_path = tmp_path / "evaluation.json"
    report = evaluate_and_train(
        index_path,
        model_path,
        report_path,
        n_splits=4,
        seed=12,
        epochs=5,
    )

    assert model_path.is_file()
    assert report_path.is_file()
    assert report["sample_count"] == 32
    assert report["split"]["duplicate_groups_split_across_train_and_test"] is False
    assert report["split"]["all_samples_tested_once"] is True
    assert report["metrics"]["confusion_matrix"]["labels_in_order"] == list(CLASS_LABELS)
    with np.load(model_path, allow_pickle=False) as archive:
        assert archive["weights"].shape == (4, FEATURE_DIMENSION)
        assert archive["classes"].tolist() == list(CLASS_LABELS)
    loaded_classifier = load_classifier(model_path)
    assert len(loaded_classifier.predict(np.asarray(feature_rows[:2]))) == 2


def test_metrics_reject_invalid_labels_and_empty_data() -> None:
    with pytest.raises(ValueError, match="Unknown class labels"):
        calculate_classification_metrics(["FN"], ["UNKNOWN"])
    with pytest.raises(ValueError, match="At least one prediction"):
        calculate_classification_metrics([], [])
