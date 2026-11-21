from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import cbir
from app.cbir import (
    CLASS_LABELS,
    FEATURE_DIMENSION,
    build_index,
    cosine_similarities,
    load_index,
    query_index,
)


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    project_root = tmp_path
    dataset_dir = project_root / "data" / "raw" / "FNNPK"
    image_paths = []
    colors = {
        "FN": (255, 0, 0),
        "-N": (0, 255, 0),
        "-P": (0, 0, 255),
        "-K": (255, 255, 0),
    }
    for label in CLASS_LABELS:
        class_dir = dataset_dir / label
        class_dir.mkdir(parents=True)
        image_path = class_dir / f"{label.replace('-', 'minus')}.png"
        Image.new("RGB", (4, 3), colors[label]).save(image_path)
        image_paths.append(image_path)
    return project_root, dataset_dir, image_paths


def fake_encoder(image: Image.Image) -> np.ndarray:
    red, green, blue = image.getpixel((0, 0))
    features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    features[:3] = (red, green, blue)
    return features


def test_build_index_saves_576_features_and_metadata(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, _ = dataset
    output_path = tmp_path / "processed" / "baseline.npz"

    count = build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )

    index = load_index(output_path)
    assert count == 4
    assert index.features.shape == (4, 576)
    assert index.features.dtype == np.float32
    assert index.feature_dimension == 576
    assert index.model_id == "torchvision.models.mobilenet_v3_small"
    assert index.weight_id == "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    assert "center crop 224" in index.preprocessing


def test_build_index_preserves_labels_and_project_relative_paths(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, _ = dataset
    output_path = tmp_path / "baseline.npz"

    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )
    index = load_index(output_path)

    assert index.labels.tolist() == list(CLASS_LABELS)
    assert index.paths.tolist() == [
        f"data/raw/FNNPK/{label}/{label.replace('-', 'minus')}.png"
        for label in CLASS_LABELS
    ]
    with np.load(output_path, allow_pickle=False) as archive:
        assert archive["labels"].dtype.kind == "U"
        assert archive["paths"].dtype.kind == "U"


def test_cosine_similarity_orders_matches_correctly() -> None:
    references = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    query = np.array([0.8, 0.6], dtype=np.float32)

    similarities = cosine_similarities(references, query)

    assert similarities.tolist() == pytest.approx([0.8, 0.6, -0.8])
    assert np.argsort(-similarities, kind="stable").tolist() == [0, 1, 2]


def test_cosine_similarity_handles_zero_vectors() -> None:
    references = np.array([[0, 0], [1, 0]], dtype=np.float32)
    query = np.array([0, 0], dtype=np.float32)

    similarities = cosine_similarities(references, query)

    assert similarities.tolist() == [0.0, 0.0]
    assert np.isfinite(similarities).all()


def test_query_excludes_the_query_image_from_results(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, image_paths = dataset
    output_path = tmp_path / "baseline.npz"
    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )

    results = query_index(
        image_paths[0],
        output_path,
        top_k=10,
        encoder=fake_encoder,
        project_root=project_root,
    )

    assert len(results) == 3
    assert all(result.path != "data/raw/FNNPK/FN/FN.png" for result in results)
    assert results[0].label == "-K"
    assert results[0].similarity == pytest.approx(2**-0.5)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_query_rejects_invalid_top_k(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
    top_k: int,
) -> None:
    project_root, dataset_dir, image_paths = dataset
    output_path = tmp_path / "baseline.npz"
    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        query_index(
            image_paths[0],
            output_path,
            top_k=top_k,
            encoder=fake_encoder,
            project_root=project_root,
        )


def test_query_rejects_nonexistent_image(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, _ = dataset
    output_path = tmp_path / "baseline.npz"
    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )

    with pytest.raises(FileNotFoundError, match="Query image does not exist"):
        query_index(
            tmp_path / "missing.png",
            output_path,
            encoder=fake_encoder,
            project_root=project_root,
        )


def test_query_rejects_unreadable_image(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, _ = dataset
    output_path = tmp_path / "baseline.npz"
    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )
    unreadable_image = tmp_path / "unreadable.png"
    unreadable_image.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Cannot read image"):
        query_index(
            unreadable_image,
            output_path,
            encoder=fake_encoder,
            project_root=project_root,
        )


def test_query_reports_decompression_bomb_as_unreadable(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, dataset_dir, _ = dataset
    output_path = tmp_path / "baseline.npz"
    build_index(
        dataset_dir,
        output_path,
        encoder=fake_encoder,
        project_root=project_root,
    )
    query_image = tmp_path / "large.png"
    query_image.write_bytes(b"image data")

    def raise_decompression_bomb(*args: object, **kwargs: object) -> None:
        raise Image.DecompressionBombError("image dimensions exceed limit")

    monkeypatch.setattr(cbir.Image, "open", raise_decompression_bomb)

    with pytest.raises(ValueError, match="Cannot read image"):
        query_index(
            query_image,
            output_path,
            encoder=fake_encoder,
            project_root=project_root,
        )


def test_build_does_not_skip_unreadable_dataset_images(
    tmp_path: Path,
    dataset: tuple[Path, Path, list[Path]],
) -> None:
    project_root, dataset_dir, _ = dataset
    unreadable_image = dataset_dir / "FN" / "broken.png"
    unreadable_image.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Cannot read image"):
        build_index(
            dataset_dir,
            tmp_path / "baseline.npz",
            encoder=fake_encoder,
            project_root=project_root,
        )
