"""Build a separate frozen-feature index for the LettuceDeF dataset."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from app.cbir import (
    FEATURE_DIMENSION,
    MODEL_ID,
    PREPROCESSING,
    PROJECT_ROOT,
    WEIGHT_ID,
    ImageEncoder,
    MobileNetV3SmallEncoder,
    _extract_image_features,
)

DATASET_NAME = "LettuceDeF"
DATASET_VERSION = "1.0.0"
CLASS_MAPPING = {
    "FN": "FN",
    "N-deficiency": "-N",
    "P-deficiency": "-P",
    "K-deficiency": "-K",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "LettuceDeF" / "extracted" / "images"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lettucedef-mobilenet-v3-small.npz"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)


@dataclass(frozen=True)
class ManifestImage:
    source_class: str
    label: str
    filename: str
    relative_image_path: str
    project_relative_path: str
    path: Path
    sha256: str
    split: str
    cultivar: str
    capture_date: str
    capture_time: str
    byte_count: int | None


def map_manifest_class(source_class: str) -> str:
    """Map a LettuceDeF folder label to the existing classifier class code."""
    try:
        return CLASS_MAPPING[source_class]
    except KeyError as exc:
        raise ValueError(f"Unknown LettuceDeF class: {source_class!r}.") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Image '{path}' is outside project root '{project_root}'."
        ) from exc


def _load_manifest_images(
    image_root: Path,
    manifest_path: Path,
    project_root: Path,
) -> list[ManifestImage]:
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: '{image_root}'.")

    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest is not a file: '{manifest_path}'.")

    image_inventory: dict[str, list[Path]] = {}
    filename_inventory: dict[str, list[Path]] = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in JPEG_EXTENSIONS:
            continue
        try:
            relative = path.resolve(strict=True).relative_to(image_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Image '{path}' resolves outside image root '{image_root}'."
            ) from exc
        image_inventory.setdefault(relative.casefold(), []).append(path)
        filename_inventory.setdefault(path.name.casefold(), []).append(path)

    required_columns = {"class", "file_name", "sha256"}
    manifest_images: list[ManifestImage] = []
    seen_paths: set[str] = set()
    present_classes: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Manifest is empty or has no header.")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
                + "."
            )

        for row_number, row in enumerate(reader, start=2):
            source_class = (row.get("class") or "").strip()
            label = map_manifest_class(source_class)
            filename = (row.get("file_name") or "").strip()
            if (
                not filename
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
            ):
                raise ValueError(
                    f"Invalid file_name in manifest row {row_number}: {filename!r}."
                )
            if Path(filename).suffix.lower() not in JPEG_EXTENSIONS:
                raise ValueError(
                    f"Manifest row {row_number} does not name a JPEG image."
                )

            relative_image_path = PurePosixPath(
                source_class, filename
            ).as_posix()
            inventory_key = relative_image_path.casefold()
            if inventory_key in seen_paths:
                raise ValueError(
                    f"Manifest contains duplicate image path '{relative_image_path}'."
                )
            seen_paths.add(inventory_key)

            matches = image_inventory.get(inventory_key, [])
            if not matches:
                same_name = filename_inventory.get(filename.casefold(), [])
                if same_name:
                    found_paths = [
                        path.resolve(strict=True).relative_to(image_root).as_posix()
                        for path in same_name
                    ]
                    raise ValueError(
                        f"Manifest class '{source_class}' does not match the "
                        f"image folder for '{filename}'; found: "
                        f"{', '.join(found_paths)}."
                    )
                raise FileNotFoundError(
                    f"Manifest image does not exist: '{relative_image_path}'."
                )
            if len(matches) != 1:
                raise ValueError(
                    f"Manifest path '{relative_image_path}' resolves to "
                    f"{len(matches)} images; expected exactly one."
                )

            image_path = matches[0].resolve(strict=True)
            try:
                actual_relative = image_path.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest image '{relative_image_path}' resolves outside "
                    "the image root."
                ) from exc
            if (
                len(actual_relative.parts) != 2
                or actual_relative.parts[0] != source_class
                or actual_relative.name != filename
            ):
                raise ValueError(
                    f"Manifest class '{source_class}' does not match the image "
                    f"folder for '{filename}'."
                )

            expected_sha256 = (row.get("sha256") or "").strip().lower()
            if SHA256_PATTERN.fullmatch(expected_sha256) is None:
                raise ValueError(
                    f"Invalid SHA-256 value in manifest row {row_number}."
                )
            actual_sha256 = _sha256_file(image_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for '{relative_image_path}': expected "
                    f"{expected_sha256}, got {actual_sha256}."
                )

            byte_value = (row.get("bytes") or "").strip()
            if byte_value:
                try:
                    byte_count = int(byte_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid byte count in manifest row {row_number}."
                    ) from exc
                if byte_count != image_path.stat().st_size:
                    raise ValueError(
                        f"Byte-size mismatch for '{relative_image_path}'."
                    )
            else:
                byte_count = None

            present_classes.add(source_class)
            manifest_images.append(
                ManifestImage(
                    source_class=source_class,
                    label=label,
                    filename=filename,
                    relative_image_path=relative_image_path,
                    project_relative_path=_project_relative_path(
                        image_path, project_root
                    ),
                    path=image_path,
                    sha256=actual_sha256,
                    split=(row.get("split") or "").strip(),
                    cultivar=(row.get("cultivar") or "").strip(),
                    capture_date=(row.get("capture_date") or "").strip(),
                    capture_time=(row.get("capture_time") or "").strip(),
                    byte_count=byte_count,
                )
            )

    if not manifest_images:
        raise ValueError("Manifest contains no image rows.")
    missing_classes = set(CLASS_MAPPING).difference(present_classes)
    if missing_classes:
        raise ValueError(
            "Manifest is missing required classes: "
            + ", ".join(sorted(missing_classes))
            + "."
        )
    return manifest_images


def build_lettucedef_index(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    manifest_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    encoder: ImageEncoder | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    """Build a separate index from one validated LettuceDeF manifest."""
    output_path = Path(output_path).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing LettuceDeF index: '{output_path}'."
        )
    if manifest_path is None:
        raise ValueError("A source manifest path must be supplied.")

    project_root = Path(project_root).resolve()
    image_root = Path(image_root)
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest_images = _load_manifest_images(
        image_root,
        manifest_path,
        project_root,
    )

    feature_encoder = (
        encoder
        if encoder is not None
        else MobileNetV3SmallEncoder(require_cached_weights=True)
    )
    features = np.empty(
        (len(manifest_images), FEATURE_DIMENSION),
        dtype=np.float32,
    )
    for position, entry in enumerate(manifest_images):
        features[position] = _extract_image_features(
            entry.path,
            feature_encoder,
        )

    manifest_sha256 = _sha256_file(manifest_path)
    manifest_path_record = _project_relative_path(manifest_path, project_root)
    image_root_record = _project_relative_path(image_root, project_root)
    mapped_distribution = Counter(entry.label for entry in manifest_images)
    source_distribution = Counter(
        entry.source_class for entry in manifest_images
    )
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "model_id": MODEL_ID,
        "weight_id": WEIGHT_ID,
        "preprocessing": PREPROCESSING,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
        "source_manifest_path": manifest_path_record,
        "source_manifest_sha256": manifest_sha256,
        "image_root": image_root_record,
        "image_count": len(manifest_images),
        "class_distribution": dict(sorted(mapped_distribution.items())),
        "original_manifest_class_distribution": dict(
            sorted(source_distribution.items())
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as target:
            np.savez_compressed(
                target,
                index_version=np.asarray(1, dtype=np.int64),
                features=features,
                labels=np.asarray(
                    [entry.label for entry in manifest_images],
                    dtype=np.str_,
                ),
                paths=np.asarray(
                    [entry.project_relative_path for entry in manifest_images],
                    dtype=np.str_,
                ),
                model_id=np.asarray(MODEL_ID),
                weight_id=np.asarray(WEIGHT_ID),
                preprocessing=np.asarray(PREPROCESSING),
                feature_dimension=np.asarray(FEATURE_DIMENSION, dtype=np.int64),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                source_relative_paths=np.asarray(
                    [entry.relative_image_path for entry in manifest_images],
                    dtype=np.str_,
                ),
                filenames=np.asarray(
                    [entry.filename for entry in manifest_images],
                    dtype=np.str_,
                ),
                manifest_classes=np.asarray(
                    [entry.source_class for entry in manifest_images],
                    dtype=np.str_,
                ),
                split_assignments=np.asarray(
                    [entry.split for entry in manifest_images],
                    dtype=np.str_,
                ),
                cultivars=np.asarray(
                    [entry.cultivar for entry in manifest_images],
                    dtype=np.str_,
                ),
                capture_dates=np.asarray(
                    [entry.capture_date for entry in manifest_images],
                    dtype=np.str_,
                ),
                capture_times=np.asarray(
                    [entry.capture_time for entry in manifest_images],
                    dtype=np.str_,
                ),
                image_sha256=np.asarray(
                    [entry.sha256 for entry in manifest_images],
                    dtype=np.str_,
                ),
                image_bytes=np.asarray(
                    [
                        entry.byte_count if entry.byte_count is not None else -1
                        for entry in manifest_images
                    ],
                    dtype=np.int64,
                ),
            )
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing LettuceDeF index: '{output_path}'."
        ) from exc
    return len(manifest_images)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a separate CPU feature index for LettuceDeF."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Root containing the LettuceDeF class folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Protocol A or B CSV manifest used as the image inventory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="New output NPZ path; existing files are never overwritten.",
    )
    return parser


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    try:
        image_count = build_lettucedef_index(
            args.images,
            args.manifest,
            args.output,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Indexed {image_count} LettuceDeF images to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

