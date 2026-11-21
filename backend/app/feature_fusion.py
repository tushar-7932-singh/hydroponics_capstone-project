"""Standalone deep + handcrafted feature fusion for FNNPK images.

The deep block delegates to the project's existing MobileNetV3-Small encoder;
this module does not define or train another CNN.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from app.cbir import FEATURE_DIMENSION, ImageEncoder, MobileNetV3SmallEncoder

DEEP_FEATURE_DIMENSION = FEATURE_DIMENSION
HSV_BINS_PER_CHANNEL = 16
COLOUR_FEATURE_DIMENSION = 3 * HSV_BINS_PER_CHANNEL
LBP_POINTS = 8
LBP_RADIUS = 1
TEXTURE_FEATURE_DIMENSION = 2 ** LBP_POINTS
FUSED_FEATURE_DIMENSION = (
    DEEP_FEATURE_DIMENSION + COLOUR_FEATURE_DIMENSION + TEXTURE_FEATURE_DIMENSION
)
DESCRIPTOR_IMAGE_SIZE = (224, 224)


def _rgb_copy(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image instance.")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive.")
    try:
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("image must be readable and non-empty.") from exc


def _normalize_l2(features: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{name} features must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} features must contain only finite values.")
    norm = float(np.linalg.norm(values.astype(np.float64)))
    if norm == 0.0:
        return values.copy()
    return (values / np.float32(norm)).astype(np.float32, copy=False)


def extract_hsv_histogram(image: Image.Image) -> np.ndarray:
    """Return a 48-value L1-normalized HSV histogram (16 bins per channel)."""
    rgb = _rgb_copy(image)
    try:
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)
    finally:
        rgb.close()

    parts = []
    for channel in range(3):
        quantized = (
            hsv[:, :, channel].astype(np.uint16) * HSV_BINS_PER_CHANNEL
        ) // 256
        histogram = np.bincount(
            quantized.reshape(-1), minlength=HSV_BINS_PER_CHANNEL
        ).astype(np.float32)
        parts.append(histogram)
    result = np.concatenate(parts)
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("HSV histogram is empty.")
    return result / np.float32(total)


def extract_lbp_histogram(image: Image.Image) -> np.ndarray:
    """Return a 256-value LBP histogram using 8 neighbors at radius 1."""
    rgb = _rgb_copy(image)
    try:
        small = ImageOps.fit(
            rgb, DESCRIPTOR_IMAGE_SIZE, method=Image.Resampling.BILINEAR
        )
    finally:
        rgb.close()
    try:
        gray = np.asarray(small.convert("L"), dtype=np.uint8)
    finally:
        small.close()

    center = gray[LBP_RADIUS:-LBP_RADIUS, LBP_RADIUS:-LBP_RADIUS]
    codes = np.zeros(center.shape, dtype=np.uint8)
    bit = 0
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1),
    ):
        neighbor = gray[
            LBP_RADIUS + dy : gray.shape[0] - LBP_RADIUS + dy,
            LBP_RADIUS + dx : gray.shape[1] - LBP_RADIUS + dx,
        ]
        codes |= (neighbor >= center).astype(np.uint8) << bit
        bit += 1

    histogram = np.bincount(
        codes.reshape(-1), minlength=TEXTURE_FEATURE_DIMENSION
    ).astype(np.float32)
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("LBP histogram is empty.")
    return histogram / np.float32(total)


class DeepFeatureFusionEncoder:
    """Fuse normalized MobileNet, HSV histogram, and LBP feature blocks.

    Each block is L2-normalized to give the three modalities equal norm before
    concatenation. The returned 880-value vector is L2-normalized for cosine
    similarity. The default deep encoder runs the existing MobileNet on CPU.
    """

    def __init__(self, deep_encoder: ImageEncoder | None = None) -> None:
        self._deep_encoder = (
            deep_encoder if deep_encoder is not None else MobileNetV3SmallEncoder()
        )

    def extract_blocks(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return individually L2-normalized (deep, HSV, LBP) feature blocks."""
        rgb = _rgb_copy(image)
        try:
            deep = np.asarray(self._deep_encoder(rgb), dtype=np.float32).reshape(-1)
            if deep.shape != (DEEP_FEATURE_DIMENSION,):
                raise ValueError(
                    "deep encoder must return exactly "
                    f"{DEEP_FEATURE_DIMENSION} values; got {deep.size}."
                )
            deep = _normalize_l2(deep, "deep")

            small = ImageOps.fit(
                rgb, DESCRIPTOR_IMAGE_SIZE, method=Image.Resampling.BILINEAR
            )
            try:
                colour = extract_hsv_histogram(small)
                texture = extract_lbp_histogram(small)
            finally:
                small.close()
        finally:
            rgb.close()

        # Histograms are first L1-normalized as distributions, then L2-normalized
        # here so each feature family contributes an equal-norm block.
        colour = _normalize_l2(colour, "colour")
        texture = _normalize_l2(texture, "texture")
        return deep, colour, texture

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Return one finite, normalized fused feature vector for an image."""
        fused = np.concatenate(self.extract_blocks(image)).astype(
            np.float32, copy=False
        )
        if fused.shape != (FUSED_FEATURE_DIMENSION,):
            raise ValueError(
                f"fused vector must have {FUSED_FEATURE_DIMENSION} values."
            )
        return _normalize_l2(fused, "fused")

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract_features(image)
