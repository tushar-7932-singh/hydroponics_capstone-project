from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(main.app)


def make_image(image_format: str) -> bytes:
    image_buffer = BytesIO()
    Image.new("RGB", (4, 3), color=(45, 120, 75)).save(image_buffer, format=image_format)
    return image_buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("WEBP", "image/webp"),
    ],
)
def test_uploads_supported_image_formats(
    client: TestClient,
    image_format: str,
    mime_type: str,
) -> None:
    suffix = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image_format]
    response = client.post(
        "/api/v1/images/upload",
        files={"file": (f"leaf.{suffix}", make_image(image_format), mime_type)},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "uploaded"
    assert result["original_filename"] == f"leaf.{suffix}"
    assert result["detected_format"] == image_format
    assert result["width"] == 4
    assert result["height"] == 3
    assert len(result["upload_id"]) == 32

    stored_files = list((main.UPLOAD_DIR).iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].name == f"{result['upload_id']}.{suffix}"
    assert stored_files[0].read_bytes() == make_image(image_format)


def test_detects_actual_format_instead_of_trusting_filename_or_mime_type(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("../../leaf.png", make_image("JPEG"), "image/png")},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["original_filename"] == "leaf.png"
    assert result["detected_format"] == "JPEG"
    assert all("/" not in path.name and "\\" not in path.name for path in main.UPLOAD_DIR.iterdir())


def test_rejects_non_image_with_forged_image_mime_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("leaf.png", b"this is not an image", "image/png")},
    )

    assert response.status_code == 415
    assert "JPEG, PNG, or WebP" in response.json()["detail"]
    assert not main.UPLOAD_DIR.exists()


def test_rejects_unsupported_image_format(client: TestClient) -> None:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("leaf.bmp", make_image("BMP"), "image/bmp")},
    )

    assert response.status_code == 415
    assert "Unsupported image format" in response.json()["detail"]
    assert not main.UPLOAD_DIR.exists()


def test_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."
    assert not main.UPLOAD_DIR.exists()


def test_rejects_oversized_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/images/upload",
        files={
            "file": (
                "large.png",
                b"x" * (main.MAX_UPLOAD_SIZE + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]
    assert not main.UPLOAD_DIR.exists()


def test_rejects_corrupt_image(client: TestClient) -> None:
    corrupt_png = make_image("PNG")[:-12]
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("corrupt.png", corrupt_png, "image/png")},
    )

    assert response.status_code == 400
    assert "corrupt or cannot be decoded" in response.json()["detail"]
    assert not main.UPLOAD_DIR.exists()


def test_health_endpoint_remains_available(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_only_the_local_vite_origin(client: TestClient) -> None:
    allowed_responses = [
        client.options(
            "/api/v1/images/upload",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173")
    ]
    disallowed = client.options(
        "/api/v1/images/upload",
        headers={
            "Origin": "https://example.invalid",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    for origin, response in zip(
        ("http://localhost:5173", "http://127.0.0.1:5173"),
        allowed_responses,
        strict=True,
    ):
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "access-control-allow-credentials" not in response.headers
    assert "access-control-allow-origin" not in disallowed.headers
