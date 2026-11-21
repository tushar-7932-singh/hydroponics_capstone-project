import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from threading import Lock
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "FNNPK"
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "data" / "processed" / "fnnpk-mobilenet-v3-small.npz"
)
CLASS_LABELS = ("FN", "-N", "-P", "-K")
IMAGE_EXTENSIONS = {".png"}
FEATURE_DIMENSION = 576
MODEL_ID = "torchvision.models.mobilenet_v3_small"
WEIGHT_ID = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
PREPROCESSING = (
    "TorchVision IMAGENET1K_V1 transforms; RGB; resize 256; center crop 224; "
    "scale to [0, 1] and ImageNet mean/std normalization"
)
INDEX_VERSION = 1

ImageEncoder = Callable[[Image.Image], np.ndarray]


@dataclass(frozen=True)
class FeatureIndex:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    model_id: str
    weight_id: str
    preprocessing: str
    feature_dimension: int


@dataclass(frozen=True)
class RetrievalResult:
    label: str
    path: str
    similarity: float


class ReferenceImageNotFoundError(FileNotFoundError):
    """Raised when an opaque reference ID cannot resolve to a safe image."""


class MobileNetV3SmallEncoder:
    """Load pretrained MobileNetV3-Small once and run inference on CPU."""

    def __init__(self, *, require_cached_weights: bool = False) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self._torch = torch
        self._weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        if require_cached_weights:
            checkpoint_path = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / Path(self._weights.url).name
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Pretrained MobileNetV3-Small weights are not cached locally."
                )
            self._model = mobilenet_v3_small(weights=None)
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        else:
            self._model = mobilenet_v3_small(weights=self._weights)
        self._model.classifier = torch.nn.Identity()
        self._model.eval()
        self._model.to("cpu")

    def __call__(self, image: Image.Image) -> np.ndarray:
        input_tensor = self._weights.transforms()(image).unsqueeze(0)
        with self._torch.inference_mode():
            features = self._model(input_tensor)
        return features.squeeze(0).detach().cpu().numpy()


def _read_rgb_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as source:
            source.load()
            return source.convert("RGB")
    except (
        OSError,
        SyntaxError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Cannot read image '{image_path}': {exc}") from exc


def _extract_image_features(
    image_path: Path,
    encoder: ImageEncoder,
) -> np.ndarray:
    image = _read_rgb_image(image_path)
    try:
        features = np.asarray(encoder(image), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"Feature extraction failed for '{image_path}': {exc}") from exc
    finally:
        image.close()

    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Encoder returned {features.size} features for '{image_path}'; "
            f"expected {FEATURE_DIMENSION}."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Encoder returned non-finite features for '{image_path}'.")
    return features


def build_index(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_path: Path = DEFAULT_INDEX_PATH,
    *,
    encoder: ImageEncoder | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    """Extract each class image in sequence and save a reusable NPZ index."""
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path)
    project_root = Path(project_root).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: '{dataset_dir}'.")

    image_entries: list[tuple[str, Path, str]] = []
    for label in CLASS_LABELS:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class folder does not exist: '{class_dir}'.")

        class_images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if not class_images:
            raise ValueError(f"No PNG images found in class folder '{class_dir}'.")

        for image_path in class_images:
            try:
                relative_path = image_path.resolve().relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Image '{image_path}' is outside project root '{project_root}'."
                ) from exc
            image_entries.append((label, image_path, relative_path))

    feature_encoder = encoder or MobileNetV3SmallEncoder()
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    relative_paths: list[str] = []
    for label, image_path, relative_path in image_entries:
        feature_rows.append(_extract_image_features(image_path, feature_encoder))
        labels.append(label)
        relative_paths.append(relative_path)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    with output_path.open("wb") as index_file:
        np.savez_compressed(
            index_file,
            index_version=np.asarray(INDEX_VERSION),
            features=features,
            labels=np.asarray(labels, dtype=np.str_),
            paths=np.asarray(relative_paths, dtype=np.str_),
            model_id=np.asarray(MODEL_ID),
            weight_id=np.asarray(WEIGHT_ID),
            preprocessing=np.asarray(PREPROCESSING),
            feature_dimension=np.asarray(FEATURE_DIMENSION),
        )
    return len(relative_paths)


def load_index(index_path: Path) -> FeatureIndex:
    """Load an index without permitting pickled object arrays."""
    try:
        with np.load(Path(index_path), allow_pickle=False) as archive:
            required_fields = {
                "features",
                "labels",
                "paths",
                "model_id",
                "weight_id",
                "preprocessing",
                "feature_dimension",
            }
            missing_fields = required_fields.difference(archive.files)
            if missing_fields:
                raise ValueError(
                    f"Index is missing required fields: {', '.join(sorted(missing_fields))}."
                )
            result = FeatureIndex(
                features=np.asarray(archive["features"], dtype=np.float32),
                labels=np.asarray(archive["labels"]).copy(),
                paths=np.asarray(archive["paths"]).copy(),
                model_id=str(archive["model_id"].item()),
                weight_id=str(archive["weight_id"].item()),
                preprocessing=str(archive["preprocessing"].item()),
                feature_dimension=int(archive["feature_dimension"].item()),
            )
    except (OSError, KeyError) as exc:
        raise ValueError(f"Cannot read feature index '{index_path}': {exc}") from exc

    if result.feature_dimension != FEATURE_DIMENSION:
        raise ValueError(
            f"Index feature dimension is {result.feature_dimension}; "
            f"expected {FEATURE_DIMENSION}."
        )
    if result.features.ndim != 2 or result.features.shape[1] != FEATURE_DIMENSION:
        raise ValueError("Index features must have shape (image_count, 576).")
    if result.labels.ndim != 1 or result.paths.ndim != 1:
        raise ValueError("Index labels and paths must be one-dimensional.")
    if not (len(result.features) == len(result.labels) == len(result.paths)):
        raise ValueError("Index feature, label, and path counts do not match.")
    if not np.isfinite(result.features).all():
        raise ValueError("Index contains non-finite feature values.")
    return result


def cosine_similarities(
    reference_features: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity, assigning zero when either vector is zero."""
    references = np.asarray(reference_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if references.ndim != 2:
        raise ValueError("Reference features must be a two-dimensional array.")
    if query.ndim != 1:
        raise ValueError("Query features must be a one-dimensional array.")
    if references.shape[1] != query.shape[0]:
        raise ValueError("Reference and query feature dimensions do not match.")
    if not np.isfinite(references).all() or not np.isfinite(query).all():
        raise ValueError("Cosine similarity requires finite feature values.")

    reference_norms = np.linalg.norm(references, axis=1)
    query_norm = np.linalg.norm(query)
    similarities = np.zeros(len(references), dtype=np.float64)
    valid_references = reference_norms > 0
    if query_norm > 0:
        similarities[valid_references] = (
            references[valid_references] @ query
        ) / (reference_norms[valid_references] * query_norm)
    return similarities


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    return int(top_k)


def query_index(
    query_image: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    *,
    top_k: int = 5,
    encoder: ImageEncoder | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[RetrievalResult]:
    """Return top-k cosine matches, excluding the query if it is indexed."""
    top_k = _validate_top_k(top_k)
    query_path = Path(query_image).resolve()
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image does not exist: '{query_path}'.")

    index = load_index(index_path)
    feature_encoder = encoder or MobileNetV3SmallEncoder()
    return query_loaded_index(
        query_path,
        index,
        encoder=feature_encoder,
        top_k=top_k,
        project_root=project_root,
    )


def query_loaded_index(
    query_image: Path,
    index: FeatureIndex,
    *,
    encoder: ImageEncoder,
    top_k: int = 5,
    project_root: Path = PROJECT_ROOT,
) -> list[RetrievalResult]:
    """Query an already loaded feature index without reloading model resources."""
    top_k = _validate_top_k(top_k)
    query_path = Path(query_image).resolve()
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image does not exist: '{query_path}'.")

    query_features = _extract_image_features(query_path, encoder)
    similarities = cosine_similarities(index.features, query_features)

    query_relative_path: str | None = None
    try:
        query_relative_path = os.path.normcase(
            query_path.relative_to(Path(project_root).resolve()).as_posix()
        )
    except ValueError:
        pass

    candidates = [
        position
        for position, indexed_path in enumerate(index.paths)
        if query_relative_path is None
        or os.path.normcase(str(indexed_path)) != query_relative_path
    ]
    ordered_positions = sorted(candidates, key=lambda position: -similarities[position])
    return [
        RetrievalResult(
            label=str(index.labels[position]),
            path=str(index.paths[position]),
            similarity=float(similarities[position]),
        )
        for position in ordered_positions[:top_k]
    ]


class CBIRRetrievalEngine:
    """Keep the saved index and encoder resident and serialize CPU inference."""

    def __init__(
        self,
        index: FeatureIndex,
        encoder: ImageEncoder,
        *,
        index_sha256: str | None = None,
    ) -> None:
        self._index = index
        self._encoder = encoder
        self.index_sha256 = index_sha256
        self._inference_lock = Lock()

    @classmethod
    def load(cls, index_path: Path = DEFAULT_INDEX_PATH) -> "CBIRRetrievalEngine":
        index = load_index(index_path)
        encoder = MobileNetV3SmallEncoder(require_cached_weights=True)
        digest = hashlib.sha256()
        with Path(index_path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(index, encoder, index_sha256=digest.hexdigest())

    def extract_features(self, image_path: Path) -> np.ndarray:
        """Extract one validated 576-dimensional vector using the shared encoder."""
        with self._inference_lock:
            return _extract_image_features(Path(image_path), self._encoder)

    def query(
        self,
        query_image: Path,
        *,
        top_k: int = 5,
        project_root: Path = PROJECT_ROOT,
    ) -> list[RetrievalResult]:
        with self._inference_lock:
            return query_loaded_index(
                query_image,
                self._index,
                encoder=self._encoder,
                top_k=top_k,
                project_root=project_root,
            )

    def resolve_reference_image(
        self,
        reference_id: str,
        *,
        dataset_root: Path = DEFAULT_DATASET_DIR,
    ) -> Path:
        """Resolve an opaque ID only to an indexed image beneath dataset_root."""
        if re.fullmatch(r"ref_[a-f0-9]{24}", reference_id, flags=re.ASCII) is None:
            raise ReferenceImageNotFoundError

        indexed_path = next(
            (
                str(path)
                for path in self._index.paths
                if "ref_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
                == reference_id
            ),
            None,
        )
        if indexed_path is None:
            raise ReferenceImageNotFoundError

        relative_path = PurePosixPath(indexed_path)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or "\\" in indexed_path
        ):
            raise ReferenceImageNotFoundError

        resolved_root = Path(dataset_root).resolve()
        try:
            resolved_image = PROJECT_ROOT.joinpath(*relative_path.parts).resolve(strict=True)
            resolved_image.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError):
            raise ReferenceImageNotFoundError from None

        if not resolved_image.is_file():
            raise ReferenceImageNotFoundError
        return resolved_image


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or query a CPU MobileNetV3-Small image retrieval index."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="Build the dataset feature index.")
    build_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)

    query_parser = commands.add_parser("query", help="Retrieve similar indexed images.")
    query_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    query_parser.add_argument("--image", type=Path, required=True)
    query_parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    try:
        if args.command == "build":
            image_count = build_index(args.dataset, args.output)
            print(f"Indexed {image_count} images to {args.output.resolve()}")
        else:
            results = query_index(args.image, args.index, top_k=args.top_k)
            print(
                json.dumps(
                    [
                        {
                            "label": result.label,
                            "path": result.path,
                            "similarity": result.similarity,
                        }
                        for result in results
                    ],
                    indent=2,
                )
            )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
