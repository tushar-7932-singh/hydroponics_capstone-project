import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.cbir import (
    CLASS_LABELS,
    DEFAULT_INDEX_PATH,
    PROJECT_ROOT,
    FeatureIndex,
    cosine_similarities,
    load_index,
)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "cbir-evaluation-report.md"
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "cbir-evaluation.json"
METRIC_KEYS = (
    "precision_at_1",
    "precision_at_5",
    "recall_at_5",
    "average_precision",
)


@dataclass(frozen=True)
class QueryMetrics:
    precision_at_1: float | None
    precision_at_5: float | None
    recall_at_5: float | None
    average_precision: float | None


@dataclass(frozen=True)
class QueryEvaluation:
    query_position: int
    query_path: str
    label: str
    relevant_reference_count: int
    metrics: QueryMetrics


def calculate_metrics(
    ranked_relevance: Sequence[bool | int],
    total_relevant: int,
) -> QueryMetrics:
    """Calculate ranking metrics from a full, ordered list of candidates."""
    if isinstance(total_relevant, bool) or not isinstance(total_relevant, int):
        raise ValueError("total_relevant must be a non-negative integer.")
    if total_relevant < 0:
        raise ValueError("total_relevant must be a non-negative integer.")

    relevance: list[int] = []
    for value in ranked_relevance:
        if isinstance(value, (bool, np.bool_)):
            relevance.append(int(value))
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            relevance.append(int(value))
        else:
            raise ValueError("ranked_relevance values must be booleans or 0/1.")

    retrieved_relevant = sum(relevance)
    if retrieved_relevant > total_relevant:
        raise ValueError(
            "ranked_relevance contains more relevant items than total_relevant."
        )

    candidate_count = len(relevance)
    precision_at_1 = (
        float(relevance[0]) if candidate_count else None
    )
    precision_denominator = min(5, candidate_count)
    precision_at_5 = (
        sum(relevance[:5]) / precision_denominator
        if precision_denominator
        else None
    )

    if total_relevant == 0:
        recall_at_5 = None
        average_precision = None
    else:
        recall_at_5 = sum(relevance[:5]) / total_relevant
        relevant_ranks = [
            rank
            for rank, is_relevant in enumerate(relevance, start=1)
            if is_relevant
        ]
        average_precision = (
            sum(
                sum(relevance[:rank]) / rank
                for rank in relevant_ranks
            )
            / total_relevant
        )

    return QueryMetrics(
        precision_at_1=precision_at_1,
        precision_at_5=precision_at_5,
        recall_at_5=recall_at_5,
        average_precision=average_precision,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_candidate_positions(
    features: np.ndarray,
    query_position: int,
) -> list[int]:
    """Rank candidate rows while excluding the query row by its exact position."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    if isinstance(query_position, bool) or not isinstance(query_position, int):
        raise ValueError("query_position must be a valid row index.")
    if query_position < 0 or query_position >= len(features):
        raise ValueError("query_position must be a valid row index.")

    candidate_positions = [
        position for position in range(len(features))
        if position != query_position
    ]
    similarities = cosine_similarities(
        features[candidate_positions],
        features[query_position],
    )
    ranked_candidate_positions = sorted(
        range(len(candidate_positions)),
        key=lambda candidate_position: (
            -float(similarities[candidate_position]),
            candidate_positions[candidate_position],
        ),
    )
    return [
        candidate_positions[candidate_position]
        for candidate_position in ranked_candidate_positions
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    defined_values = [value for value in values if value is not None]
    if not defined_values:
        return None
    return float(sum(defined_values) / len(defined_values))


def _metric_values(metrics: QueryMetrics) -> dict[str, float | None]:
    return {
        "precision_at_1": metrics.precision_at_1,
        "precision_at_5": metrics.precision_at_5,
        "recall_at_5": metrics.recall_at_5,
        "average_precision": metrics.average_precision,
    }


def evaluate_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, object]:
    """Evaluate every indexed embedding as a query without recomputing features."""
    index_path = Path(index_path).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Feature index does not exist: '{index_path}'.")

    index = load_index(index_path)
    image_count = len(index.features)
    if image_count == 0:
        raise ValueError("Cannot evaluate an empty feature index.")

    labels = [str(label) for label in index.labels]
    paths = [str(path) for path in index.paths]
    query_results: list[QueryEvaluation] = []

    for query_position, query_label in enumerate(labels):
        ranked_positions = rank_candidate_positions(index.features, query_position)
        ranked_relevance = [
            labels[position] == query_label for position in ranked_positions
        ]
        relevant_reference_count = sum(
            label == query_label for position, label in enumerate(labels)
            if position != query_position
        )
        query_results.append(
            QueryEvaluation(
                query_position=query_position,
                query_path=paths[query_position],
                label=query_label,
                relevant_reference_count=relevant_reference_count,
                metrics=calculate_metrics(
                    ranked_relevance,
                    relevant_reference_count,
                ),
            )
        )

    class_labels = list(CLASS_LABELS)
    class_labels.extend(sorted(set(labels).difference(CLASS_LABELS)))
    per_class: list[dict[str, object]] = []
    for label in class_labels:
        class_queries = [query for query in query_results if query.label == label]
        if not class_queries:
            continue
        class_metric_values = {
            key: _mean(
                [_metric_values(query.metrics)[key] for query in class_queries]
            )
            for key in METRIC_KEYS
        }
        defined_query_counts = {
            key: sum(
                _metric_values(query.metrics)[key] is not None
                for query in class_queries
            )
            for key in METRIC_KEYS
        }
        per_class.append(
            {
                "label": label,
                "query_count": len(class_queries),
                "defined_query_counts": defined_query_counts,
                "metrics": class_metric_values,
            }
        )

    query_weighted_metrics = {
        key: _mean([_metric_values(query.metrics)[key] for query in query_results])
        for key in METRIC_KEYS
    }
    macro_metrics = {
        key: _mean(
            [entry["metrics"][key] for entry in per_class]  # type: ignore[index]
        )
        for key in METRIC_KEYS
    }
    macro_class_counts = {
        key: sum(
            entry["metrics"][key] is not None  # type: ignore[index]
            for entry in per_class
        )
        for key in METRIC_KEYS
    }

    class_counts = {
        label: labels.count(label)
        for label in class_labels
        if label in labels
    }
    report_data: dict[str, object] = {
        "schema_version": 1,
        "evaluation": {
            "method": "leave-one-image-out retrieval using saved embeddings",
            "query_count": image_count,
            "candidate_count_per_query": image_count - 1,
            "self_exclusion": "exclude the query's exact index position",
            "relevance_definition": (
                "A reference is relevant when its supplied folder label matches "
                "the query's supplied folder label."
            ),
            "tie_breaking": (
                "Sort by descending cosine similarity, then ascending original "
                "index position."
            ),
            "short_ranking_convention": (
                "Precision@5 divides by min(5, available candidates); the full "
                "candidate ranking is used for Average Precision."
            ),
            "no_relevant_reference_convention": (
                "Recall@5 and Average Precision are null when no relevant "
                "reference exists. Such queries are excluded from those means; "
                "defined-query and contributing-class counts are reported."
            ),
        },
        "input_index": {
            "path": index_path.relative_to(PROJECT_ROOT).as_posix()
            if index_path.is_relative_to(PROJECT_ROOT)
            else str(index_path),
            "sha256": _sha256_file(index_path),
            "indexed_image_count": image_count,
            "feature_dimension": index.feature_dimension,
            "model_id": index.model_id,
            "weight_id": index.weight_id,
            "preprocessing": index.preprocessing,
            "class_counts": class_counts,
        },
        "metric_definitions": {
            "precision_at_1": "Relevant items in rank 1 divided by 1.",
            "precision_at_5": (
                "Relevant items in the first five ranks divided by "
                "min(5, available candidates)."
            ),
            "recall_at_5": (
                "Relevant items in the first five ranks divided by all relevant "
                "reference images after self-exclusion."
            ),
            "average_precision": (
                "Mean of precision at every relevant rank, divided by the total "
                "number of relevant reference images."
            ),
            "mean_average_precision": (
                "Mean of per-query Average Precision values for queries with at "
                "least one relevant reference."
            ),
        },
        "aggregates": {
            "query_weighted": {
                "query_count": image_count,
                "defined_query_counts": {
                    key: sum(
                        _metric_values(query.metrics)[key] is not None
                        for query in query_results
                    )
                    for key in METRIC_KEYS
                },
                "metrics": {
                    **query_weighted_metrics,
                    "mean_average_precision": query_weighted_metrics[
                        "average_precision"
                    ],
                },
            },
            "macro_across_classes": {
                "class_count": len(per_class),
                "contributing_class_counts": macro_class_counts,
                "metrics": {
                    **macro_metrics,
                    "mean_average_precision": macro_metrics["average_precision"],
                },
            },
        },
        "per_class": per_class,
        "per_query": [
            {
                "query_position": query.query_position,
                "query_path": query.query_path,
                "label": query.label,
                "relevant_reference_count": query.relevant_reference_count,
                "metrics": {
                    **_metric_values(query.metrics),
                    "average_precision": query.metrics.average_precision,
                },
            }
            for query in query_results
        ],
        "limitations": [
            (
                "Relevance uses the dataset's supplied folder labels; these "
                "labels are not independently verified nutrient diagnoses."
            ),
            (
                "Leave-one-image-out evaluates queries against the same 208 "
                "images used to build the feature index, so it is not an "
                "independent held-out evaluation."
            ),
            (
                "Near-duplicates, shared plants, or shared capture sessions may "
                "inflate retrieval scores; the available index has no group IDs "
                "to support group-aware splitting."
            ),
        ],
    }
    return report_data


def _format_metric(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def render_markdown_report(report_data: dict[str, object]) -> str:
    index_data = report_data["input_index"]
    evaluation_data = report_data["evaluation"]
    aggregates = report_data["aggregates"]
    assert isinstance(index_data, dict)
    assert isinstance(evaluation_data, dict)
    assert isinstance(aggregates, dict)
    query_weighted = aggregates["query_weighted"]
    macro = aggregates["macro_across_classes"]
    assert isinstance(query_weighted, dict)
    assert isinstance(macro, dict)

    lines = [
        "# CBIR Baseline Evaluation",
        "",
        "## Summary",
        "",
        (
            "This report evaluates the existing MobileNetV3-Small feature index "
            "using leave-one-image-out retrieval. Relevance is based on matching "
            "the dataset's supplied folder labels. It does **not** validate "
            "nutrient diagnoses."
        ),
        "",
        f"- Indexed images / queries: {index_data['indexed_image_count']}",
        f"- Feature dimension: {index_data['feature_dimension']}",
        f"- Model: `{index_data['model_id']}`",
        f"- Weights: `{index_data['weight_id']}`",
        f"- Preprocessing: {index_data['preprocessing']}",
        f"- Input index: `{index_data['path']}`",
        f"- Input index SHA-256: `{index_data['sha256']}`",
        "",
        "## Dataset labels",
        "",
        "| Supplied folder label | Image / query count |",
        "|---|---:|",
    ]
    for label, count in index_data["class_counts"].items():
        lines.append(f"| `{label}` | {count} |")

    metric_names = (
        ("precision_at_1", "Precision@1"),
        ("precision_at_5", "Precision@5"),
        ("recall_at_5", "Recall@5"),
        ("mean_average_precision", "mAP"),
    )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Class / aggregation | Queries / classes | Precision@1 | Precision@5 | Recall@5 | mAP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in report_data["per_class"]:
        metrics = entry["metrics"]
        lines.append(
            f"| `{entry['label']}` | {entry['query_count']} | "
            f"{_format_metric(metrics['precision_at_1'])} | "
            f"{_format_metric(metrics['precision_at_5'])} | "
            f"{_format_metric(metrics['recall_at_5'])} | "
            f"{_format_metric(metrics['average_precision'])} |"
        )
    for aggregation_name, aggregation in (
        ("Query-weighted mean", query_weighted),
        ("Macro mean across classes", macro),
    ):
        metrics = aggregation["metrics"]
        count_label = (
            f"{aggregation['class_count']} classes"
            if "class_count" in aggregation
            else f"{aggregation['query_count']} queries"
        )
        lines.append(
            f"| **{aggregation_name}** | {count_label} | "
            + " | ".join(
                _format_metric(metrics[key])
                for key, _ in metric_names
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Evaluation methodology",
            "",
            "- Every indexed embedding is used once as a query; the saved features are reused without model inference.",
            "- The query is excluded from its candidate set by its exact index position.",
            "- A reference is relevant when its supplied folder label matches the query label.",
            "- Ranking is by descending cosine similarity; tied scores use ascending original index position.",
            "- Precision@1 is relevant results at rank 1 divided by 1.",
            "- Precision@5 is relevant results in the first five divided by `min(5, available candidates)`.",
            "- Recall@5 is relevant results in the first five divided by the number of relevant references after query exclusion.",
            "- AP averages precision at each relevant rank and divides by the total number of relevant references; mAP is mean AP across eligible queries.",
            "- Macro results give each class average equal weight. Query-weighted means are also included for reference.",
            "- If there are no relevant references, Recall@5 and AP are `N/A` and excluded from means; defined-query and contributing-class counts are recorded in the JSON.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report_data["limitations"])
    lines.extend(
        [
            "",
            "This is an initial same-dataset retrieval baseline, not evidence of "
            "generalization to unseen plants or growing environments. A stronger "
            "evaluation requires duplicate review and plant/capture-group-aware "
            "query/reference separation where group metadata is available.",
            "",
            "Machine-readable per-query results and metric definitions are in "
            "`data/processed/cbir-evaluation.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation(
    index_path: Path = DEFAULT_INDEX_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
) -> dict[str, object]:
    report_data = evaluate_index(index_path)
    report_path = Path(report_path)
    json_path = Path(json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(report_data), encoding="utf-8")
    json_path.write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate leave-one-out CBIR retrieval from a saved feature index."
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    args = parser.parse_args()

    try:
        report_data = write_evaluation(args.index, args.report, args.json)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    macro = report_data["aggregates"]["macro_across_classes"]["metrics"]
    print(f"Evaluated {report_data['evaluation']['query_count']} queries.")
    print(
        "Macro metrics: "
        f"P@1={_format_metric(macro['precision_at_1'])}, "
        f"P@5={_format_metric(macro['precision_at_5'])}, "
        f"R@5={_format_metric(macro['recall_at_5'])}, "
        f"mAP={_format_metric(macro['mean_average_precision'])}"
    )
    print(f"Report: {Path(args.report).resolve()}")
    print(f"JSON: {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
