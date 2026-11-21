from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import cbir, main
from app.cbir import CLASS_LABELS, FEATURE_DIMENSION, CBIRRetrievalEngine, FeatureIndex
from app.classification import LinearFeatureClassifier

INDEX_HASH = "a" * 64
CLASS_COLORS = {
    "FN": (255, 0, 0),
    "-N": (0, 255, 0),
    "-P": (0, 0, 255),
    "-K": (255, 255, 0),
}
encoder_calls: list[tuple[int, int, int]] = []


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_encoder(image: Image.Image) -> np.ndarray:
    color = image.getpixel((0, 0))
    encoder_calls.append(color)
    features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    features[tuple(CLASS_COLORS.values()).index(color)] = 1.0
    return features


def _classifier(index_hash: str = INDEX_HASH) -> LinearFeatureClassifier:
    weights = np.zeros((len(CLASS_LABELS), FEATURE_DIMENSION), dtype=np.float32)
    weights[:, : len(CLASS_LABELS)] = np.eye(len(CLASS_LABELS), dtype=np.float32)
    return LinearFeatureClassifier(
        classes=CLASS_LABELS,
        feature_mean=np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        feature_scale=np.ones(FEATURE_DIMENSION, dtype=np.float32),
        weights=weights,
        bias=np.zeros(len(CLASS_LABELS), dtype=np.float32),
        index_sha256=index_hash,
    )


def _engine() -> CBIRRetrievalEngine:
    features = np.zeros((len(CLASS_LABELS), FEATURE_DIMENSION), dtype=np.float32)
    features[:, : len(CLASS_LABELS)] = np.eye(len(CLASS_LABELS), dtype=np.float32)
    index = FeatureIndex(
        features=features,
        labels=np.asarray(CLASS_LABELS, dtype=np.str_),
        paths=np.asarray(
            [f"data/raw/FNNPK/{label}/reference.png" for label in CLASS_LABELS],
            dtype=np.str_,
        ),
        model_id="test-model",
        weight_id="test-weights",
        preprocessing="test-preprocessing",
        feature_dimension=FEATURE_DIMENSION,
    )
    return CBIRRetrievalEngine(index, _fake_encoder, index_sha256=INDEX_HASH)


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    encoder_calls.clear()
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cbir, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_load_retrieval_engine", _engine)
    monkeypatch.setattr(main, "_load_classifier", _classifier)
    with TestClient(main.app) as test_client:
        yield test_client


def _upload(client: TestClient, color: tuple[int, int, int]) -> str:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("leaf.png", _image_bytes(color), "image/png")},
    )
    assert response.status_code == 201
    return str(response.json()["upload_id"])


@pytest.mark.parametrize(
    ("label", "class_name"),
    [
        ("FN", "Fully Nutritional"),
        ("-N", "Nitrogen Deficient"),
        ("-P", "Phosphorus Deficient"),
        ("-K", "Potassium Deficient"),
    ],
)
def test_classification_extracts_features_with_shared_encoder_and_returns_prediction(
    client: TestClient,
    label: str,
    class_name: str,
) -> None:
    upload_id = _upload(client, CLASS_COLORS[label])

    response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "upload_id": upload_id,
        "status": "predicted",
        "predicted_class": label,
        "class_name": class_name,
        "note": (
            "This is a model prediction based on dataset-provided labels, "
            "not a confirmed nutrient-deficiency diagnosis."
        ),
    }
    assert encoder_calls == [CLASS_COLORS[label]]
    assert "confidence" not in result
    assert "probability" not in result


@pytest.mark.parametrize(
    ("upload_id", "expected_status"),
    [("bad-id", 422), ("a" * 32, 404)],
)
def test_classification_rejects_invalid_or_missing_upload_ids(
    client: TestClient,
    upload_id: str,
    expected_status: int,
) -> None:
    response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == expected_status


def test_classification_returns_404_when_uploaded_image_is_missing(
    client: TestClient,
) -> None:
    upload_id = _upload(client, CLASS_COLORS["FN"])
    (main.UPLOAD_DIR / f"{upload_id}.png").unlink()

    response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == 404


def test_classification_returns_400_when_stored_image_cannot_be_decoded(
    client: TestClient,
) -> None:
    upload_id = _upload(client, CLASS_COLORS["FN"])
    (main.UPLOAD_DIR / f"{upload_id}.png").write_bytes(b"not an image")

    response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == 400


def test_classifier_index_hash_mismatch_disables_classification_but_not_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cbir, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_load_retrieval_engine", _engine)
    monkeypatch.setattr(main, "_load_classifier", lambda: _classifier("b" * 64))
    with TestClient(main.app) as client:
        upload_id = _upload(client, CLASS_COLORS["FN"])

        classification_response = client.post(
            f"/api/v1/images/{upload_id}/classify"
        )
        retrieval_response = client.post(
            f"/api/v1/images/{upload_id}/retrieve"
        )

    assert classification_response.status_code == 503
    assert "classification is unavailable" in classification_response.json()["detail"]
    assert retrieval_response.status_code == 200


def test_classification_returns_503_when_classifier_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cbir, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_load_retrieval_engine", _engine)

    def unavailable_classifier() -> LinearFeatureClassifier:
        raise FileNotFoundError("test classifier missing")

    monkeypatch.setattr(main, "_load_classifier", unavailable_classifier)
    with TestClient(main.app) as client:
        upload_id = _upload(client, CLASS_COLORS["FN"])
        response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == 503


def test_classification_handles_encoder_inference_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_id = _upload(client, CLASS_COLORS["FN"])
    engine = main.app.state.cbir_retrieval_engine

    def fail_extraction(image_path: Path) -> np.ndarray:
        raise RuntimeError("test inference failure")

    monkeypatch.setattr(engine, "extract_features", fail_extraction)

    response = client.post(f"/api/v1/images/{upload_id}/classify")

    assert response.status_code == 503
    assert "classification failed" in response.json()["detail"]


def test_classification_failure_does_not_break_retrieval(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_id = _upload(client, CLASS_COLORS["FN"])
    classifier = main.app.state.lettuce_classifier

    def fail_prediction(
        classifier_instance: LinearFeatureClassifier,
        features: np.ndarray,
    ) -> list[str]:
        raise RuntimeError("test classifier failure")

    monkeypatch.setattr(type(classifier), "predict", fail_prediction)

    classification_response = client.post(
        f"/api/v1/images/{upload_id}/classify"
    )
    retrieval_response = client.post(
        f"/api/v1/images/{upload_id}/retrieve"
    )

    assert classification_response.status_code == 503
    assert retrieval_response.status_code == 200


def test_retrieval_inference_failure_does_not_break_classification(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_id = _upload(client, CLASS_COLORS["-P"])
    engine = main.app.state.cbir_retrieval_engine

    def fail_retrieval(
        query_image: Path,
        *,
        top_k: int = 5,
        project_root: Path = cbir.PROJECT_ROOT,
    ) -> list[object]:
        raise RuntimeError("test retrieval failure")

    monkeypatch.setattr(engine, "query", fail_retrieval)

    retrieval_response = client.post(
        f"/api/v1/images/{upload_id}/retrieve"
    )
    classification_response = client.post(
        f"/api/v1/images/{upload_id}/classify"
    )

    assert retrieval_response.status_code == 503
    assert classification_response.status_code == 200
    assert classification_response.json()["predicted_class"] == "-P"
