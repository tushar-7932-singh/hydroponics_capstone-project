from pathlib import Path

import numpy as np
import pytest

from app.cbir import CLASS_LABELS, FEATURE_DIMENSION, MODEL_ID, PREPROCESSING, WEIGHT_ID
from app.evaluation import (
    calculate_metrics,
    evaluate_index,
    rank_candidate_positions,
)


def _write_index(
    path: Path,
    labels: list[str],
    features: np.ndarray | None = None,
) -> Path:
    if features is None:
        features = np.ones((len(labels), FEATURE_DIMENSION), dtype=np.float32)
    paths = [f"data/raw/FNNPK/{label}/image-{position}.png" for position, label in enumerate(labels)]
    with path.open("wb") as target:
        np.savez_compressed(
            target,
            index_version=np.asarray(1),
            features=features.astype(np.float32),
            labels=np.asarray(labels, dtype=np.str_),
            paths=np.asarray(paths, dtype=np.str_),
            model_id=np.asarray(MODEL_ID),
            weight_id=np.asarray(WEIGHT_ID),
            preprocessing=np.asarray(PREPROCESSING),
            feature_dimension=np.asarray(FEATURE_DIMENSION),
        )
    return path


def test_calculate_metrics_for_known_ranking() -> None:
    metrics = calculate_metrics([1, 0, 1, 1, 0], total_relevant=4)

    assert metrics.precision_at_1 == pytest.approx(1.0)
    assert metrics.precision_at_5 == pytest.approx(0.6)
    assert metrics.recall_at_5 == pytest.approx(0.75)
    assert metrics.average_precision == pytest.approx(29 / 48)


def test_calculate_metrics_uses_available_candidates_for_short_rankings() -> None:
    metrics = calculate_metrics([1, 0], total_relevant=1)

    assert metrics.precision_at_1 == pytest.approx(1.0)
    assert metrics.precision_at_5 == pytest.approx(0.5)
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(1.0)


def test_calculate_metrics_marks_no_relevant_references_undefined() -> None:
    metrics = calculate_metrics([0, 0], total_relevant=0)

    assert metrics.precision_at_1 == 0.0
    assert metrics.precision_at_5 == 0.0
    assert metrics.recall_at_5 is None
    assert metrics.average_precision is None


def test_calculate_metrics_handles_empty_candidate_list() -> None:
    metrics = calculate_metrics([], total_relevant=0)

    assert metrics.precision_at_1 is None
    assert metrics.precision_at_5 is None
    assert metrics.recall_at_5 is None
    assert metrics.average_precision is None


@pytest.mark.parametrize(
    ("relevance", "total_relevant"),
    [
        ([2, 0], 1),
        ([1, 1], 1),
        ([True, False], -1),
        ([0, 1], 1.5),
    ],
)
def test_calculate_metrics_rejects_invalid_inputs(
    relevance: list[bool | int],
    total_relevant: int,
) -> None:
    with pytest.raises(ValueError):
        calculate_metrics(relevance, total_relevant)  # type: ignore[arg-type]


def test_evaluate_index_excludes_exact_query_position_and_breaks_ties_stably(
    tmp_path: Path,
) -> None:
    labels = ["FN", "FN", "-N", "-N", "-P", "-K"]
    features = np.ones((len(labels), FEATURE_DIMENSION), dtype=np.float32)
    index_path = _write_index(tmp_path / "index.npz", labels, features)

    report = evaluate_index(index_path)
    queries = report["per_query"]
    first_query = queries[0]

    assert len(queries) == len(labels)
    ranked_positions = rank_candidate_positions(features, query_position=0)
    assert ranked_positions == [1, 2, 3, 4, 5]
    assert 0 not in ranked_positions
    assert first_query["query_path"] == "data/raw/FNNPK/FN/image-0.png"


def test_evaluate_index_reports_per_class_and_macro_averages(
    tmp_path: Path,
) -> None:
    labels = ["FN", "FN", "-N", "-N", "-P", "-K"]
    report = evaluate_index(_write_index(tmp_path / "index.npz", labels))

    by_label = {entry["label"]: entry for entry in report["per_class"]}
    assert {label: by_label[label]["query_count"] for label in CLASS_LABELS} == {
        "FN": 2,
        "-N": 2,
        "-P": 1,
        "-K": 1,
    }
    assert by_label["FN"]["metrics"]["precision_at_1"] == pytest.approx(1.0)
    assert by_label["-N"]["metrics"]["precision_at_1"] == pytest.approx(0.0)
    assert by_label["-P"]["metrics"]["average_precision"] is None
    assert by_label["-K"]["defined_query_counts"]["recall_at_5"] == 0

    macro = report["aggregates"]["macro_across_classes"]
    assert macro["metrics"]["precision_at_1"] == pytest.approx(0.25)
    assert macro["metrics"]["precision_at_5"] == pytest.approx(0.1)
    assert macro["metrics"]["recall_at_5"] == pytest.approx(1.0)
    assert macro["metrics"]["mean_average_precision"] == pytest.approx(2 / 3)
    assert macro["contributing_class_counts"]["average_precision"] == 2
