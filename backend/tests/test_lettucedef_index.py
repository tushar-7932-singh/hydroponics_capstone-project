import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.cbir import CLASS_LABELS, FEATURE_DIMENSION, load_index
from app.lettucedef_index import (
    CLASS_MAPPING,
    build_lettucedef_index,
    map_manifest_class,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 9), color).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "class",
        "split",
        "cultivar",
        "capture_date",
        "capture_time",
        "bytes",
        "sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _valid_dataset(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, str]]]:
    project_root = tmp_path
    image_root = project_root / "data" / "raw" / "LettuceDeF" / "images"
    manifest_path = project_root / "data" / "raw" / "LettuceDeF" / "manifest.csv"
    colors = {
        "FN": (20, 80, 40),
        "N-deficiency": (40, 100, 20),
        "P-deficiency": (60, 120, 10),
        "K-deficiency": (80, 140, 5),
    }
    rows = []
    for source_class, color in colors.items():
        filename = f"{source_class.replace('-', '_')}.jpg"
        image_path = image_root / source_class / filename
        sha256 = _write_image(image_path, color)
        rows.append(
            {
                "file_name": filename,
                "class": source_class,
                "split": "train",
                "cultivar": "blanes",
                "capture_date": "2025-04-10",
                "capture_time": "09:03:17",
                "bytes": str(image_path.stat().st_size),
                "sha256": sha256,
            }
        )
    _write_manifest(manifest_path, rows)
    return project_root, image_root, manifest_path, rows


def _fake_encoder(image: Image.Image) -> np.ndarray:
    red, green, blue = image.getpixel((0, 0))
    features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    features[:3] = (red, green, blue)
    return features


@pytest.mark.parametrize(
    ("source_class", "expected"),
    [
        ("FN", "FN"),
        ("N-deficiency", "-N"),
        ("P-deficiency", "-P"),
        ("K-deficiency", "-K"),
    ],
)
def test_map_manifest_class(source_class: str, expected: str) -> None:
    assert map_manifest_class(source_class) == expected


def test_map_manifest_class_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown LettuceDeF class"):
        map_manifest_class("Other")


def test_build_index_validates_manifest_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    project_root, image_root, manifest_path, rows = _valid_dataset(tmp_path)
    output_path = project_root / "data" / "processed" / "lettucedef.npz"

    image_count = build_lettucedef_index(
        image_root,
        manifest_path,
        output_path,
        encoder=_fake_encoder,
        project_root=project_root,
    )

    assert image_count == 4
    index = load_index(output_path)
    assert index.features.shape == (4, FEATURE_DIMENSION)
    assert index.features.dtype == np.float32
    assert index.labels.tolist() == [CLASS_MAPPING[row["class"]] for row in rows]
    assert index.paths.tolist() == [
        f"data/raw/LettuceDeF/images/{row['class']}/{row['file_name']}"
        for row in rows
    ]

    with np.load(output_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["dataset_name"] == "LettuceDeF"
        assert metadata["dataset_version"] == "1.0.0"
        assert metadata["feature_dimension"] == FEATURE_DIMENSION
        assert metadata["feature_dtype"] == "float32"
        assert metadata["image_count"] == 4
        assert metadata["source_manifest_sha256"] == hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        assert metadata["class_distribution"] == {
            "FN": 1,
            "-K": 1,
            "-N": 1,
            "-P": 1,
        }
        assert archive["source_relative_paths"].tolist() == [
            f"{row['class']}/{row['file_name']}" for row in rows
        ]
        assert archive["manifest_classes"].tolist() == [
            row["class"] for row in rows
        ]
        assert archive["filenames"].tolist() == [
            row["file_name"] for row in rows
        ]
        assert archive["split_assignments"].tolist() == [
            row["split"] for row in rows
        ]
        assert archive["image_sha256"].tolist() == [
            row["sha256"] for row in rows
        ]


def test_manifest_duplicate_image_path_is_rejected(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, rows = _valid_dataset(tmp_path)
    _write_manifest(manifest_path, rows + [rows[0]])

    with pytest.raises(ValueError, match="duplicate image path"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            project_root / "index.npz",
            encoder=_fake_encoder,
            project_root=project_root,
        )


def test_manifest_missing_image_is_rejected(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, _ = _valid_dataset(tmp_path)
    (image_root / "N-deficiency" / "N_deficiency.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="Manifest image does not exist"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            project_root / "index.npz",
            encoder=_fake_encoder,
            project_root=project_root,
        )


def test_manifest_class_must_match_image_folder(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, rows = _valid_dataset(tmp_path)
    wrong_row = rows[1].copy()
    wrong_row["file_name"] = rows[0]["file_name"]
    _write_manifest(manifest_path, [rows[0], wrong_row, rows[2], rows[3]])

    with pytest.raises(ValueError, match="does not match the image folder"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            project_root / "index.npz",
            encoder=_fake_encoder,
            project_root=project_root,
        )


def test_manifest_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, rows = _valid_dataset(tmp_path)
    rows[0]["sha256"] = "0" * 64
    _write_manifest(manifest_path, rows)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            project_root / "index.npz",
            encoder=_fake_encoder,
            project_root=project_root,
        )


def test_encoder_feature_shape_must_be_576(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, _ = _valid_dataset(tmp_path)
    output_path = project_root / "index.npz"

    with pytest.raises(ValueError, match="expected 576"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            output_path,
            encoder=lambda image: np.zeros(575, dtype=np.float32),
            project_root=project_root,
        )
    assert not output_path.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, _ = _valid_dataset(tmp_path)
    output_path = project_root / "index.npz"
    original_content = b"keep existing index"
    output_path.write_bytes(original_content)
    encoder_called = False

    def encoder(image: Image.Image) -> np.ndarray:
        nonlocal encoder_called
        encoder_called = True
        return _fake_encoder(image)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_lettucedef_index(
            image_root,
            manifest_path,
            output_path,
            encoder=encoder,
            project_root=project_root,
        )

    assert not encoder_called
    assert output_path.read_bytes() == original_content


def test_saved_labels_follow_existing_class_order(tmp_path: Path) -> None:
    project_root, image_root, manifest_path, _ = _valid_dataset(tmp_path)
    output_path = project_root / "index.npz"

    build_lettucedef_index(
        image_root,
        manifest_path,
        output_path,
        encoder=_fake_encoder,
        project_root=project_root,
    )
    index = load_index(output_path)

    assert set(index.labels.tolist()) == set(CLASS_LABELS)
