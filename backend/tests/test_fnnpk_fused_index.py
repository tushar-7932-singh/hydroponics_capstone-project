import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.cbir import CLASS_LABELS
from app.feature_fusion import FUSED_FEATURE_DIMENSION
from app.fnnpk_fused_index import (
    CLASS_MAPPING,
    DEFAULT_OUTPUT_PATH,
    EXPECTED_IMAGE_COUNT,
    build_fused_index,
)


EXPECTED_CLASS_COUNTS = {"FN": 12, "-N": 58, "-P": 66, "-K": 72}


class FakeFusionEncoder:
    def extract_features(self, image: Image.Image) -> np.ndarray:
        value = float(image.getpixel((0, 0))[0] + 1)
        return np.full(FUSED_FEATURE_DIMENSION, value, dtype=np.float32)


@pytest.fixture
def fnnpk_dataset(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    project_root = tmp_path
    dataset_dir = project_root / "data" / "raw" / "FNNPK"
    filenames: dict[str, list[str]] = {}
    for label in CLASS_LABELS:
        class_dir = dataset_dir / label
        class_dir.mkdir(parents=True)
        filenames[label] = []
        count = EXPECTED_CLASS_COUNTS[label]
        for position in range(count, 0, -1):
            filename = f"sample_{position:03}.png"
            filenames[label].append(filename)
            Image.new("RGB", (5, 4), (position, 30, 60)).save(class_dir / filename)
    return dataset_dir, filenames


def test_build_fused_index_saves_all_features_and_metadata(
    tmp_path: Path,
    fnnpk_dataset: tuple[Path, dict[str, list[str]]],
) -> None:
    dataset_dir, _ = fnnpk_dataset
    output_path = tmp_path / "processed" / "fused.npz"

    count = build_fused_index(
        dataset_dir,
        output_path,
        encoder=FakeFusionEncoder(),
        project_root=tmp_path,
    )

    assert count == EXPECTED_IMAGE_COUNT
    assert output_path.is_file()
    with np.load(output_path, allow_pickle=False) as archive:
        features = archive["features"]
        labels = archive["labels"].tolist()
        paths = archive["paths"].tolist()
        assert features.shape == (208, 880)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
        assert archive["feature_dimension"].item() == 880
        assert json.loads(archive["class_mapping"].item()) == CLASS_MAPPING
        assert json.loads(archive["class_counts"].item()) == EXPECTED_CLASS_COUNTS
        metadata = json.loads(archive["metadata"].item())
        assert metadata["image_count"] == 208
        assert metadata["feature_dimension"] == 880
        assert metadata["encoder_details"]["fusion_encoder"] == "DeepFeatureFusionEncoder"
        assert metadata["encoder_details"]["deep_feature_dimension"] == 576
        assert metadata["encoder_details"]["lbp_radius"] == 1

    assert labels == [
        label
        for label in CLASS_LABELS
        for _ in range(EXPECTED_CLASS_COUNTS[label])
    ]
    assert paths == [
        f"data/raw/FNNPK/{label}/sample_{position:03}.png"
        for label in CLASS_LABELS
        for position in range(1, EXPECTED_CLASS_COUNTS[label] + 1)
    ]
    # Files were created in descending order; this also checks deterministic sorting.
    for label_position, label in enumerate(CLASS_LABELS):
        first_row = sum(EXPECTED_CLASS_COUNTS[x] for x in CLASS_LABELS[:label_position])
        assert features[first_row, 0] == 2


def test_default_output_path_is_the_separate_fused_index() -> None:
    assert DEFAULT_OUTPUT_PATH.name == "fnnpk-fused-mobilenet-hsv-lbp.npz"
    assert DEFAULT_OUTPUT_PATH.parent.name == "processed"


def test_builder_refuses_to_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "already-exists.npz"
    original = b"preserve existing artifact"
    output_path.write_bytes(original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_fused_index(
            tmp_path / "missing-dataset",
            output_path,
            encoder=FakeFusionEncoder(),
            project_root=tmp_path,
        )

    assert output_path.read_bytes() == original


def test_builder_rejects_dataset_with_wrong_image_count(
    tmp_path: Path,
    fnnpk_dataset: tuple[Path, dict[str, list[str]]],
) -> None:
    dataset_dir, _ = fnnpk_dataset
    (dataset_dir / "FN" / "sample_001.png").unlink()

    with pytest.raises(ValueError, match="Expected exactly 208"):
        build_fused_index(
            dataset_dir,
            tmp_path / "fused.npz",
            encoder=FakeFusionEncoder(),
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("bad_vector", "message"),
    [
        (np.zeros(FUSED_FEATURE_DIMENSION - 1, dtype=np.float32), "expected \\(880,\\)"),
        (np.full(FUSED_FEATURE_DIMENSION, np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_builder_rejects_invalid_fused_features_without_writing_index(
    tmp_path: Path,
    fnnpk_dataset: tuple[Path, dict[str, list[str]]],
    bad_vector: np.ndarray,
    message: str,
) -> None:
    dataset_dir, _ = fnnpk_dataset

    class InvalidEncoder:
        def extract_features(self, image: Image.Image) -> np.ndarray:
            return bad_vector

    output_path = tmp_path / "fused.npz"
    with pytest.raises(ValueError, match=message):
        build_fused_index(
            dataset_dir,
            output_path,
            encoder=InvalidEncoder(),
            project_root=tmp_path,
        )
    assert not output_path.exists()

