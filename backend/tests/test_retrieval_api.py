import hashlib
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import cbir, main
from app.cbir import (
    CLASS_LABELS,
    FEATURE_DIMENSION,
    CBIRRetrievalEngine,
    FeatureIndex,
)


def _image_bytes() -> bytes:
    from io import BytesIO

    image_buffer = BytesIO()
    Image.new("RGB", (4, 3), color=(255, 0, 0)).save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def _fake_encoder(image: Image.Image) -> np.ndarray:
    red, green, blue = image.getpixel((0, 0))
    features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    features[:3] = (red, green, blue)
    return features


def _fake_engine() -> CBIRRetrievalEngine:
    labels = [CLASS_LABELS[position % len(CLASS_LABELS)] for position in range(7)]
    features = np.zeros((len(labels), FEATURE_DIMENSION), dtype=np.float32)
    features[:, 0] = 1.0
    index = FeatureIndex(
        features=features,
        labels=np.asarray(labels, dtype=np.str_),
        paths=np.asarray(
            [
                f"data/raw/FNNPK/{label}/reference-{position}.png"
                for position, label in enumerate(labels)
            ],
            dtype=np.str_,
        ),
        model_id="test-model",
        weight_id="test-weights",
        preprocessing="test-preprocessing",
        feature_dimension=FEATURE_DIMENSION,
    )
    return CBIRRetrievalEngine(index, _fake_encoder)


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cbir, "PROJECT_ROOT", tmp_path)
    dataset_root = tmp_path / "data" / "raw" / "FNNPK"
    monkeypatch.setattr(main, "REFERENCE_DATASET_DIR", dataset_root)
    for position in range(7):
        label = CLASS_LABELS[position % len(CLASS_LABELS)]
        reference_path = dataset_root / label / f"reference-{position}.png"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), color=(position * 20, 100, 50)).save(reference_path)
    monkeypatch.setattr(main, "_load_retrieval_engine", _fake_engine)
    with TestClient(main.app) as test_client:
        yield test_client


def _upload(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("leaf.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 201
    return response.json()


def test_health_and_upload_response_remain_backward_compatible(
    client: TestClient,
) -> None:
    health_response = client.get("/health")
    upload_response = client.post(
        "/api/v1/images/upload",
        files={"file": ("leaf.png", _image_bytes(), "image/png")},
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert upload_response.status_code == 201
    assert set(upload_response.json()) == {
        "upload_id",
        "original_filename",
        "detected_format",
        "width",
        "height",
        "status",
    }
    assert upload_response.json()["status"] == "uploaded"


def test_retrieval_uses_stored_upload_and_returns_safe_top_five_matches(
    client: TestClient,
) -> None:
    uploaded = _upload(client)
    upload_id = uploaded["upload_id"]
    response = client.post(f"/api/v1/images/{upload_id}/retrieve")

    assert response.status_code == 200
    result = response.json()
    assert result["upload_id"] == upload_id
    assert result["status"] == "retrieved"
    assert result["label_semantics"] == "dataset_provided_label"
    assert result["similarity_semantics"] == "cosine_similarity_not_probability"
    assert "not confirmed diagnoses" in result["note"]
    assert len(result["matches"]) == 5
    for match in result["matches"]:
        assert set(match) == {
            "reference_id",
            "dataset_label",
            "display_filename",
            "similarity_score",
        }
        assert match["dataset_label"] in CLASS_LABELS
        assert match["display_filename"].startswith("reference-")
        assert -1.0 <= match["similarity_score"] <= 1.0
        assert "data/" not in str(match)
        assert "C:\\" not in str(match)


@pytest.mark.parametrize(
    ("upload_id", "expected_status"),
    [
        ("not-an-id", 422),
        ("a" * 32, 404),
    ],
)
def test_retrieval_rejects_invalid_or_unknown_upload_ids(
    client: TestClient,
    upload_id: str,
    expected_status: int,
) -> None:
    response = client.post(f"/api/v1/images/{upload_id}/retrieve")

    assert response.status_code == expected_status


def test_retrieval_does_not_follow_upload_symlinks(
    client: TestClient,
    tmp_path: Path,
) -> None:
    upload_id = "a" * 32
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(_image_bytes())
    upload_dir = main.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    link = upload_dir / f"{upload_id}.png"
    try:
        link.symlink_to(outside_image)
    except OSError:
        pytest.skip("This Windows environment does not permit creating symlinks.")

    response = client.post(f"/api/v1/images/{upload_id}/retrieve")

    assert response.status_code == 404


def test_retrieval_returns_503_when_optional_engine_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")

    def unavailable_engine():
        raise ImportError("optional ML package is unavailable")

    monkeypatch.setattr(main, "_load_retrieval_engine", unavailable_engine)
    with TestClient(main.app) as client:
        uploaded = _upload(client)
        response = client.post(
            f"/api/v1/images/{uploaded['upload_id']}/retrieve"
        )

    assert response.status_code == 503
    assert "CBIR retrieval is unavailable" in response.json()["detail"]


def _reference_id(path: str) -> str:
    return "ref_" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def test_serves_indexed_reference_image_with_detected_media_type(
    client: TestClient,
) -> None:
    index_path = "data/raw/FNNPK/FN/reference-0.png"
    response = client.get(
        f"/api/v1/references/{_reference_id(index_path)}/image"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (4, 3)


def test_reference_image_endpoint_rejects_unknown_and_malformed_ids(
    client: TestClient,
) -> None:
    for reference_id in ("ref_" + "0" * 24, "not-a-reference-id"):
        response = client.get(f"/api/v1/references/{reference_id}/image")

        assert response.status_code == 404
        assert "reference-0.png" not in response.text
        assert "C:\\" not in response.text


def test_reference_image_endpoint_returns_not_found_for_missing_indexed_file(
    client: TestClient,
) -> None:
    reference_path = Path(main.REFERENCE_DATASET_DIR) / "FN" / "reference-0.png"
    reference_path.unlink()
    reference_id = _reference_id("data/raw/FNNPK/FN/reference-0.png")

    response = client.get(f"/api/v1/references/{reference_id}/image")

    assert response.status_code == 404
    assert "C:\\" not in response.text


def test_reference_image_endpoint_rejects_path_traversal_index_entries(
    client: TestClient,
) -> None:
    engine = main.app.state.cbir_retrieval_engine
    engine._index.paths[0] = np.str_("data/raw/FNNPK/../../outside.png")
    reference_id = _reference_id("data/raw/FNNPK/../../outside.png")

    response = client.get(f"/api/v1/references/{reference_id}/image")

    assert response.status_code == 404
    assert "outside.png" not in response.text


def test_reference_image_endpoint_rejects_symlink_escape_when_supported(
    client: TestClient,
    tmp_path: Path,
) -> None:
    relative_path = "data/raw/FNNPK/FN/reference-0.png"
    reference_path = Path(main.REFERENCE_DATASET_DIR) / "FN" / "reference-0.png"
    outside_path = tmp_path / "outside.png"
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(outside_path)
    reference_path.unlink()
    try:
        reference_path.symlink_to(outside_path)
    except OSError:
        pytest.skip("This Windows environment does not permit creating symlinks.")

    response = client.get(
        f"/api/v1/references/{_reference_id(relative_path)}/image"
    )

    assert response.status_code == 404
