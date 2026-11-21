"""Build a separate MobileNet + HSV + LBP FNNPK feature index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from app.cbir import (
    CLASS_LABELS,
    DEFAULT_DATASET_DIR,
    IMAGE_EXTENSIONS,
    MODEL_ID,
    PREPROCESSING,
    PROJECT_ROOT,
    WEIGHT_ID,
    _read_rgb_image,
)
from app.feature_fusion import (
    COLOUR_FEATURE_DIMENSION,
    DEEP_FEATURE_DIMENSION,
    DESCRIPTOR_IMAGE_SIZE,
    FUSED_FEATURE_DIMENSION,
    HSV_BINS_PER_CHANNEL,
    LBP_POINTS,
    LBP_RADIUS,
    TEXTURE_FEATURE_DIMENSION,
    DeepFeatureFusionEncoder,
)

EXPECTED_IMAGE_COUNT = 208
FUSED_INDEX_VERSION = 1
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fnnpk-fused-mobilenet-hsv-lbp.npz"
)
CLASS_MAPPING = {label: label for label in CLASS_LABELS}


class FusionEncoder(Protocol):
    def extract_features(self, image: Image.Image) -> np.ndarray: ...


def _collect_image_entries(
    dataset_dir: Path,
    project_root: Path,
) -> list[tuple[str, Path, str]]:
    entries: list[tuple[str, Path, str]] = []
    for label in CLASS_LABELS:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required FNNPK class folder does not exist: '{class_dir}'.")

        images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        for image_path in images:
            try:
                relative_path = image_path.resolve().relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Image '{image_path}' is outside project root '{project_root}'."
                ) from exc
            entries.append((label, image_path, relative_path))

    if len(entries) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_IMAGE_COUNT} FNNPK images; found {len(entries)}."
        )
    return entries


def build_fused_index(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    encoder: FusionEncoder | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    """Extract and save all 208 fused FNNPK feature vectors, one image at a time."""
    dataset_dir = Path(dataset_dir).resolve()
    project_root = Path(project_root).resolve()
    output_path = Path(output_path).resolve()

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing fused index: '{output_path}'.")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"FNNPK dataset directory does not exist: '{dataset_dir}'.")

    image_entries = _collect_image_entries(dataset_dir, project_root)
    feature_encoder = encoder if encoder is not None else DeepFeatureFusionEncoder()

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    image_paths: list[str] = []
    for label, image_path, relative_path in image_entries:
        image = _read_rgb_image(image_path)
        try:
            feature_vector = np.asarray(
                feature_encoder.extract_features(image), dtype=np.float32
            )
        finally:
            image.close()

        if feature_vector.shape != (FUSED_FEATURE_DIMENSION,):
            raise ValueError(
                f"Fusion encoder returned shape {feature_vector.shape} for "
                f"'{image_path}'; expected ({FUSED_FEATURE_DIMENSION},)."
            )
        if not np.isfinite(feature_vector).all():
            raise ValueError(f"Fusion encoder returned non-finite values for '{image_path}'.")
        feature_rows.append(feature_vector.copy())
        labels.append(label)
        image_paths.append(relative_path)

    features = np.stack(feature_rows).astype(np.float32, copy=False)
    if features.shape != (EXPECTED_IMAGE_COUNT, FUSED_FEATURE_DIMENSION):
        raise ValueError(
            "Fused index has unexpected shape "
            f"{features.shape}; expected ({EXPECTED_IMAGE_COUNT}, "
            f"{FUSED_FEATURE_DIMENSION})."
        )
    if not np.isfinite(features).all():
        raise ValueError("Fused index contains non-finite feature values.")

    class_counts = {
        label: int(Counter(labels)[label])
        for label in CLASS_LABELS
    }
    encoder_details = {
        "fusion_encoder": "DeepFeatureFusionEncoder",
        "deep_model_id": MODEL_ID,
        "deep_weight_id": WEIGHT_ID,
        "deep_feature_dimension": DEEP_FEATURE_DIMENSION,
        "colour_feature_dimension": COLOUR_FEATURE_DIMENSION,
        "texture_feature_dimension": TEXTURE_FEATURE_DIMENSION,
        "fused_feature_dimension": FUSED_FEATURE_DIMENSION,
        "hsv_bins_per_channel": HSV_BINS_PER_CHANNEL,
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
    }
    preprocessing_metadata = {
        "deep": PREPROCESSING,
        "handcrafted_image_size": list(DESCRIPTOR_IMAGE_SIZE),
        "handcrafted_crop": "center crop fitted to 224x224; bilinear resampling",
        "colour": "RGB converted to HSV; independent 16-bin histograms for H, S, and V; L1 then block L2 normalization",
        "texture": "grayscale 8-neighbor LBP histogram at radius 1; L1 then block L2 normalization",
        "fusion": "concatenate deep, HSV, and LBP blocks; final L2 normalization",
    }
    metadata = {
        "dataset_name": "FNNPK",
        "image_count": EXPECTED_IMAGE_COUNT,
        "feature_dimension": FUSED_FEATURE_DIMENSION,
        "class_mapping": CLASS_MAPPING,
        "class_counts": class_counts,
        "encoder_details": encoder_details,
        "preprocessing_metadata": preprocessing_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against an output appearing during extraction.
    with output_path.open("xb") as index_file:
        np.savez_compressed(
            index_file,
            index_version=np.asarray(FUSED_INDEX_VERSION, dtype=np.int64),
            features=features,
            labels=np.asarray(labels, dtype=np.str_),
            paths=np.asarray(image_paths, dtype=np.str_),
            feature_dimension=np.asarray(FUSED_FEATURE_DIMENSION, dtype=np.int64),
            class_mapping=np.asarray(json.dumps(CLASS_MAPPING, sort_keys=True)),
            class_counts=np.asarray(json.dumps(class_counts, sort_keys=True)),
            encoder_details=np.asarray(json.dumps(encoder_details, sort_keys=True)),
            preprocessing_metadata=np.asarray(
                json.dumps(preprocessing_metadata, sort_keys=True)
            ),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return len(image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the separate 880-dimensional FNNPK fused-feature index."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        count = build_fused_index(args.dataset, args.output)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(f"Indexed {count} FNNPK images to {Path(args.output).resolve()}")
    print(f"Feature shape: ({count}, {FUSED_FEATURE_DIMENSION})")
    with np.load(args.output, allow_pickle=False) as archive:
        print(f"Class counts: {archive['class_counts'].item()}")


if __name__ == "__main__":
    main()
