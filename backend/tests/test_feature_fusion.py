import numpy as np
import pytest
from PIL import Image

from app.cbir import FEATURE_DIMENSION
from app.feature_fusion import (
    COLOUR_FEATURE_DIMENSION,
    DEEP_FEATURE_DIMENSION,
    FUSED_FEATURE_DIMENSION,
    TEXTURE_FEATURE_DIMENSION,
    DeepFeatureFusionEncoder,
    extract_hsv_histogram,
    extract_lbp_histogram,
)


def fake_encoder(image: Image.Image) -> np.ndarray:
    return np.arange(1, FEATURE_DIMENSION + 1, dtype=np.float32)


@pytest.fixture
def sample_image() -> Image.Image:
    image = Image.new("RGB", (96, 72))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 3) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
    return image


def test_handcrafted_feature_block_dimensions_and_histogram_normalization(
    sample_image: Image.Image,
) -> None:
    colour = extract_hsv_histogram(sample_image)
    texture = extract_lbp_histogram(sample_image)

    assert colour.shape == (COLOUR_FEATURE_DIMENSION,) == (48,)
    assert texture.shape == (TEXTURE_FEATURE_DIMENSION,) == (256,)
    assert colour.dtype == np.float32
    assert texture.dtype == np.float32
    assert colour.sum() == pytest.approx(1.0)
    assert texture.sum() == pytest.approx(1.0)
    assert np.isfinite(colour).all()
    assert np.isfinite(texture).all()


def test_fused_vector_has_expected_dimension_finite_values_and_unit_norm(
    sample_image: Image.Image,
) -> None:
    encoder = DeepFeatureFusionEncoder(fake_encoder)

    fused = encoder.extract_features(sample_image)

    assert DEEP_FEATURE_DIMENSION == FEATURE_DIMENSION == 576
    assert FUSED_FEATURE_DIMENSION == 880
    assert fused.shape == (880,)
    assert fused.dtype == np.float32
    assert np.isfinite(fused).all()
    assert np.linalg.norm(fused) == pytest.approx(1.0)


def test_feature_blocks_are_individually_normalized_before_fusion(
    sample_image: Image.Image,
) -> None:
    encoder = DeepFeatureFusionEncoder(fake_encoder)

    deep, colour, texture = encoder.extract_blocks(sample_image)
    fused = encoder(sample_image)

    assert deep.shape == (DEEP_FEATURE_DIMENSION,)
    assert colour.shape == (COLOUR_FEATURE_DIMENSION,)
    assert texture.shape == (TEXTURE_FEATURE_DIMENSION,)
    assert np.linalg.norm(deep) == pytest.approx(1.0)
    assert np.linalg.norm(colour) == pytest.approx(1.0)
    assert np.linalg.norm(texture) == pytest.approx(1.0)
    ends = [DEEP_FEATURE_DIMENSION, DEEP_FEATURE_DIMENSION + COLOUR_FEATURE_DIMENSION]
    for block in np.split(fused, ends):
        assert np.linalg.norm(block) == pytest.approx(1 / np.sqrt(3), rel=1e-6)


@pytest.mark.parametrize("value", [None, "not an image", object()])
def test_rejects_non_image_inputs(value: object) -> None:
    encoder = DeepFeatureFusionEncoder(fake_encoder)
    with pytest.raises(ValueError, match="PIL.Image"):
        encoder.extract_features(value)  # type: ignore[arg-type]


def test_rejects_closed_image() -> None:
    image = Image.new("RGB", (10, 10))
    image.close()
    encoder = DeepFeatureFusionEncoder(fake_encoder)
    with pytest.raises(ValueError, match="readable"):
        encoder.extract_features(image)


@pytest.mark.parametrize(
    ("bad_features", "message"),
    [
        (np.zeros(FEATURE_DIMENSION - 1, dtype=np.float32), "exactly 576"),
        (np.full(FEATURE_DIMENSION, np.nan, dtype=np.float32), "finite"),
    ],
)
def test_rejects_invalid_deep_encoder_output(
    sample_image: Image.Image,
    bad_features: np.ndarray,
    message: str,
) -> None:
    encoder = DeepFeatureFusionEncoder(lambda image: bad_features)
    with pytest.raises(ValueError, match=message):
        encoder.extract_features(sample_image)
