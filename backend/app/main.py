import hashlib
import logging
import re
import uuid
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
REFERENCE_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "FNNPK"
)
CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "fnnpk-linear-classifier.npz"
)
READ_CHUNK_SIZE = 64 * 1024
REFERENCE_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
logger = logging.getLogger(__name__)


def _load_retrieval_engine():
    from app.cbir import CBIRRetrievalEngine

    return CBIRRetrievalEngine.load()


def _load_classifier():
    from app.classification import load_classifier

    return load_classifier(CLASSIFIER_PATH)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    try:
        retrieval_engine = _load_retrieval_engine()
        application.state.cbir_retrieval_engine = retrieval_engine
        application.state.cbir_retrieval_error = None
    except Exception as exc:
        retrieval_engine = None
        application.state.cbir_retrieval_engine = None
        application.state.cbir_retrieval_error = exc
        logger.warning("CBIR retrieval is unavailable: %s", exc)

    application.state.lettuce_classifier = None
    application.state.classification_error = None
    if retrieval_engine is None:
        application.state.classification_error = RuntimeError(
            "The shared CBIR feature encoder and index are unavailable."
        )
        logger.warning(
            "Lettuce classification is unavailable because the shared CBIR "
            "encoder and feature index could not be loaded."
        )
    else:
        try:
            classifier = _load_classifier()
            if classifier.index_sha256 != retrieval_engine.index_sha256:
                raise ValueError(
                    "Classifier feature-index SHA-256 does not match the loaded CBIR index."
                )
            application.state.lettuce_classifier = classifier
        except Exception as exc:
            application.state.classification_error = exc
            logger.warning("Lettuce classification is unavailable: %s", exc)
    yield


app = FastAPI(
    title="Hydroponics CBIR API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ImageUploadResponse(BaseModel):
    upload_id: str
    original_filename: str
    detected_format: Literal["JPEG", "PNG", "WEBP"]
    width: int
    height: int
    status: Literal["uploaded"]


class ImageRetrievalMatch(BaseModel):
    reference_id: str
    dataset_label: str
    display_filename: str
    similarity_score: float


class ImageRetrievalResponse(BaseModel):
    upload_id: str
    status: Literal["retrieved"]
    label_semantics: Literal["dataset_provided_label"]
    similarity_semantics: Literal["cosine_similarity_not_probability"]
    note: str
    matches: list[ImageRetrievalMatch]


class ImageClassificationResponse(BaseModel):
    upload_id: str
    status: Literal["predicted"]
    predicted_class: str
    class_name: str
    note: str


CLASS_NAMES = {
    "FN": "Fully Nutritional",
    "-N": "Nitrogen Deficient",
    "-P": "Phosphorus Deficient",
    "-K": "Potassium Deficient",
}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


async def _read_upload(upload: UploadFile) -> bytes:
    content = bytearray()
    while True:
        remaining = MAX_UPLOAD_SIZE + 1 - len(content)
        chunk = await upload.read(min(READ_CHUNK_SIZE, remaining))
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return bytes(content)


def _inspect_image(content: bytes) -> tuple[str, int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                detected_format = image.format
                if detected_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Unsupported image format. Upload a JPEG, PNG, or WebP image.",
                    )
                image.verify()

            with Image.open(BytesIO(content)) as image:
                image.load()
                width, height = image.size
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Upload a valid JPEG, PNG, or WebP image.",
        ) from exc
    except (
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is corrupt or cannot be decoded.",
        ) from exc

    return detected_format, width, height


def _display_filename(filename: str | None) -> str:
    basename = re.split(r"[\\/]", filename or "")[-1]
    printable_name = "".join(character for character in basename if character.isprintable())
    return printable_name[:255].strip() or "uploaded-image"


def _store_image(upload_id: str, suffix: str, content: bytes) -> None:
    destination = UPLOAD_DIR / f"{upload_id}{suffix}"
    file_created = False

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stored_file:
            file_created = True
            stored_file.write(content)
    except OSError as exc:
        if file_created:
            try:
                destination.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise HTTPException(
                    status_code=500,
                    detail="Could not store the image or clean up its incomplete file.",
                ) from cleanup_error
        raise HTTPException(
            status_code=500,
            detail="The image could not be stored on the server.",
        ) from exc


def _locate_uploaded_image(upload_id: str) -> Path:
    if re.fullmatch(r"[a-f0-9]{32}", upload_id, flags=re.ASCII) is None:
        raise HTTPException(status_code=422, detail="Invalid upload ID.")

    upload_root = UPLOAD_DIR.resolve()
    for suffix in ALLOWED_IMAGE_FORMATS.values():
        candidate = UPLOAD_DIR / f"{upload_id}{suffix}"
        if candidate.is_symlink():
            continue
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved_candidate.parent == upload_root and resolved_candidate.is_file():
            return resolved_candidate

    raise HTTPException(status_code=404, detail="Uploaded image was not found.")


@app.post("/api/v1/images/upload", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
    file: Annotated[UploadFile, File(description="A JPEG, PNG, or WebP image (maximum 10 MB).")],
) -> ImageUploadResponse:
    content = await _read_upload(file)
    detected_format, width, height = _inspect_image(content)
    upload_id = uuid.uuid4().hex
    _store_image(upload_id, ALLOWED_IMAGE_FORMATS[detected_format], content)

    return ImageUploadResponse(
        upload_id=upload_id,
        original_filename=_display_filename(file.filename),
        detected_format=detected_format,
        width=width,
        height=height,
        status="uploaded",
    )


@app.post(
    "/api/v1/images/{upload_id}/classify",
    response_model=ImageClassificationResponse,
)
def classify_uploaded_image(upload_id: str) -> ImageClassificationResponse:
    query_image = _locate_uploaded_image(upload_id)
    classifier = getattr(app.state, "lettuce_classifier", None)
    retrieval_engine = getattr(app.state, "cbir_retrieval_engine", None)
    if classifier is None or retrieval_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Lettuce classification is unavailable. The classifier, "
                "matching feature index, or shared image encoder may be missing."
            ),
        )

    try:
        features = retrieval_engine.extract_features(query_image)
        predicted_class = classifier.predict(features.reshape(1, -1))[0]
        class_name = CLASS_NAMES[predicted_class]
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Uploaded image was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The stored image could not be processed for classification.",
        ) from exc
    except Exception as exc:
        logger.exception("Lettuce classification failed for upload %s", upload_id)
        raise HTTPException(
            status_code=503,
            detail="Lettuce classification failed. Please try again later.",
        ) from exc

    return ImageClassificationResponse(
        upload_id=upload_id,
        status="predicted",
        predicted_class=predicted_class,
        class_name=class_name,
        note=(
            "This is a model prediction based on dataset-provided labels, "
            "not a confirmed nutrient-deficiency diagnosis."
        ),
    )


@app.post(
    "/api/v1/images/{upload_id}/retrieve",
    response_model=ImageRetrievalResponse,
)
def retrieve_similar_images(upload_id: str) -> ImageRetrievalResponse:
    query_image = _locate_uploaded_image(upload_id)
    retrieval_engine = getattr(app.state, "cbir_retrieval_engine", None)
    if retrieval_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "CBIR retrieval is unavailable. The feature index, optional "
                "machine-learning dependencies, or cached model weights may be "
                "missing."
            ),
        )

    try:
        results = retrieval_engine.query(query_image, top_k=5)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Uploaded image was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The stored image could not be processed for retrieval.",
        ) from exc
    except Exception as exc:
        logger.exception("CBIR retrieval failed for upload %s", upload_id)
        raise HTTPException(
            status_code=503,
            detail="CBIR retrieval failed. Please try again later.",
        ) from exc

    matches = [
        ImageRetrievalMatch(
            reference_id=(
                "ref_"
                + hashlib.sha256(result.path.encode("utf-8")).hexdigest()[:24]
            ),
            dataset_label=result.label,
            display_filename=_display_filename(result.path),
            similarity_score=result.similarity,
        )
        for result in results
    ]
    return ImageRetrievalResponse(
        upload_id=upload_id,
        status="retrieved",
        label_semantics="dataset_provided_label",
        similarity_semantics="cosine_similarity_not_probability",
        note=(
            "Dataset labels are not confirmed diagnoses; cosine similarity "
            "scores are not probabilities."
        ),
        matches=matches,
    )


@app.get("/api/v1/references/{reference_id}/image")
def get_reference_image(reference_id: str) -> Response:
    from app.cbir import ReferenceImageNotFoundError

    retrieval_engine = getattr(app.state, "cbir_retrieval_engine", None)
    if retrieval_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Reference images are unavailable while CBIR retrieval is unavailable.",
        )

    try:
        image_path = retrieval_engine.resolve_reference_image(
            reference_id,
            dataset_root=REFERENCE_DATASET_DIR,
        )
    except ReferenceImageNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Reference image was not found.",
        ) from None

    try:
        content = image_path.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image.load()
                media_type = REFERENCE_MEDIA_TYPES.get(image.format or "")
        if media_type is None:
            raise ValueError("Unsupported indexed image format.")
    except (
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
    ) as exc:
        logger.warning("Could not serve indexed reference image %s", reference_id)
        raise HTTPException(
            status_code=404,
            detail="Reference image is unavailable.",
        ) from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )
