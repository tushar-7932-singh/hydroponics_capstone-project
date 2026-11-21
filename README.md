# Hydroponics CBIR

A B.Tech final-year project foundation for **Content-Based Image Retrieval
(CBIR) with Deep Feature Fusion for Nutrient Deficiency Diagnosis in
IoT-Enabled Hydroponic Systems**.

The project contains a React/Vite frontend, a FastAPI image-upload API, and an
offline CBIR baseline that can build/query a MobileNetV3-Small feature index.
The CBIR baseline does not train a model or provide nutrient diagnoses. A
separate experimental classifier can be trained against dataset-provided
labels; it is not a validated diagnostic system. Feature fusion, databases,
sensor dashboards, and ESP32 integration are not included.

The project uses the [Lettuce NPK dataset](https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset)
from Kaggle. Its source-documented labels, local image counts, license note,
and known limitations are recorded in
[docs/dataset-selection.md](docs/dataset-selection.md).

| Folder label | Source-documented meaning | Local images |
| --- | --- | ---: |
| `FN` | Fully Nutritional | 12 |
| `-N` | Nitrogen Deficient | 58 |
| `-P` | Phosphorus Deficient | 66 |
| `-K` | Potassium Deficient | 72 |
| **Total** |  | **208** |

These are labels from the dataset documentation, not diagnoses independently
validated by this project. The dataset is small and imbalanced, particularly
the `FN` class, and five exact duplicate image pairs have been found. Do not
interpret CBIR matches or future classifier predictions as scientifically
confirmed nutrient diagnoses.

## Project structure

```text
backend/
  app/
    main.py
  tests/
    test_images.py
    test_cbir.py
  requirements.txt
  requirements-dev.txt
  requirements-ml.txt  (optional CPU retrieval dependencies)
  uploads/  (created when an image is uploaded; ignored by Git)
data/
  metadata/
  reference_images/
docs/
frontend/
  src/
scripts/
```

## Requirements

- Windows 11
- Python 3.14.6
- Node.js and npm

The backend uses FastAPI and Uvicorn. Create its virtual environment from
PowerShell with the Python version you intend to use; these instructions do not
switch Python versions.

## Start the backend (Windows PowerShell)

Open PowerShell in the project root and run:

```powershell
cd backend
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

If PowerShell prevents activation because scripts are disabled, allow scripts
for this PowerShell process only, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

With the backend running, open <http://127.0.0.1:8000/health>. It should return:

```json
{"status":"ok"}
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.
Press `Ctrl+C` in the backend terminal to stop the server. To leave the virtual
environment, run `deactivate`.

## Image upload API

`POST /api/v1/images/upload` accepts one image as multipart form data in the
field named `file`. The API decodes the uploaded content with Pillow, checks the
detected format (it does not trust the filename or browser-provided MIME type),
and accepts JPEG, PNG, or WebP images up to 10 MB. Accepted files are stored
under `backend/uploads/` using a generated identifier; this directory is
excluded from Git.

The response describes the upload only. It does not contain a diagnosis or
model results:

```json
{
  "upload_id": "1b5917d530e24a4b867297fa44a9a153",
  "original_filename": "leaf.jpg",
  "detected_format": "JPEG",
  "width": 640,
  "height": 480,
  "status": "uploaded"
}
```

In the interactive documentation at <http://127.0.0.1:8000/docs>, expand
`POST /api/v1/images/upload`, select **Try it out**, choose an image file, and
select **Execute**. A successful upload returns `201 Created`. Empty and
corrupt images return `400`, unsupported image formats return `415`, and files
over 10 MB return `413`.

You can also test from PowerShell with curl (replace the example path with an
image on your computer):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/images/upload" -F "file=@C:\path\to\leaf.jpg"
```

## Classification API

After uploading an image, call
`POST /api/v1/images/{upload_id}/classify` with the returned `upload_id`; the
endpoint does not accept a filesystem path or a second image body. It reuses
the CBIR engine's already-loaded MobileNetV3-Small encoder and exact image
preprocessing. A successful response contains the predicted class code and
readable name, but no confidence or probability:

```json
{
  "upload_id": "1b5917d530e24a4b867297fa44a9a153",
  "status": "predicted",
  "predicted_class": "-N",
  "class_name": "Nitrogen Deficient",
  "note": "This is a model prediction based on dataset-provided labels, not a confirmed nutrient-deficiency diagnosis."
}
```

Classification is optional and loads only when its saved artifact's feature
index SHA-256 matches the loaded CBIR index. If classification is unavailable,
the upload and CBIR retrieval APIs remain available. This is an experimental
prediction against source-provided labels, not a scientifically confirmed
diagnosis.

The frontend uses `http://127.0.0.1:8000` as its default API base URL. To use a
different development API URL, set `VITE_API_BASE_URL` before starting Vite, for
example:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

The backend allows browser requests only from the two local Vite origins
`http://localhost:5173` and `http://127.0.0.1:5173`. Credentialed CORS requests
are not enabled.

### Run backend tests

From the project root in PowerShell:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests use temporary directories for uploaded files, so they do not populate
`backend/uploads/`.

## Optional CPU image retrieval baseline

The retrieval baseline uses pretrained MobileNetV3-Small weights and runs on
CPU. It processes one dataset image at a time and stores features plus labels
and project-relative paths in
`data/processed/fnnpk-mobilenet-v3-small.npz`. The generated index is ignored by
Git. The original images under `data/raw/FNNPK/` are read only.

The optional machine-learning dependencies are kept separate from the normal
backend requirements. From a PowerShell window in the project root, install
them into the existing backend environment with the official PyTorch CPU
package index:

```powershell
& ".\backend\.venv\Scripts\python.exe" -m pip install -r ".\backend\requirements-ml.txt" --extra-index-url "https://download.pytorch.org/whl/cpu"
```

Verify the CPU packages and imports:

```powershell
& ".\backend\.venv\Scripts\python.exe" -c "import torch, torchvision, numpy; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Run all backend tests, including CBIR unit tests that use a fake encoder and do
not download model weights:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Build the reusable feature index (the first build downloads pretrained model
weights if they are not already cached):

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.cbir build --dataset "..\data\raw\FNNPK" --output "..\data\processed\fnnpk-mobilenet-v3-small.npz"
```

Retrieve the five most similar indexed images for a query image:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.cbir query --index "..\data\processed\fnnpk-mobilenet-v3-small.npz" --image "..\data\raw\FNNPK\FN\fn1.png" --top-k 5
```

The query output contains similarity scores and dataset labels/paths. Scores
are visual-similarity values, not probabilities or diagnoses. The query image
is excluded from results when it is already present in the index.

## Build a separate LettuceDeF feature index

The LettuceDeF index builder is separate from the existing FNNPK index builder.
It reads the image inventory and SHA-256 values from a supplied LettuceDeF
manifest, verifies every image, and maps the source classes `FN`,
`N-deficiency`, `P-deficiency`, and `K-deficiency` to `FN`, `-N`, `-P`, and
`-K`. It reuses the cached CPU MobileNetV3-Small encoder and the same weights
and preprocessing used by the FNNPK index. Images are decoded and processed
one at a time; the complete image dataset is not loaded into RAM.

The default output is the separate file
`data/processed/lettucedef-mobilenet-v3-small.npz`. The builder refuses to
overwrite an existing output. Protocol B is recommended as the manifest
provided at build time because it is the primary chronological evaluation
protocol. The index also preserves manifest class names, split assignments,
filenames, paths, and checksums. The existing FNNPK index and classifier are
not changed. **Building this index is a separate, explicit step; it has not
been performed by this documentation change.**

From PowerShell at the project root, run:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.lettucedef_index --manifest "..\data\raw\LettuceDeF\manifest_protocol_B.csv"
```

The builder expects the extracted images under
`data/raw/LettuceDeF/extracted/images/`. To select Protocol A instead, change
the manifest argument to `manifest_protocol_A.csv` and choose a new output
path if the default index has already been created. This command only creates
the feature index; it does not train or evaluate a classifier.

## Evaluate CBIR retrieval

The evaluation reuses the saved feature vectors in
`data/processed/fnnpk-mobilenet-v3-small.npz`; it does not rerun MobileNet,
download weights, train, or modify the index. Every indexed image is used once
as a query, and that exact index row is excluded from its reference candidates.
References with the same supplied folder label are counted as relevant. This
measures retrieval agreement with the dataset labels, not independently
verified nutrient diagnoses.

The evaluator calculates Precision@1, Precision@5, Recall@5, and Average
Precision for each query, then reports per-class averages, a macro average
that weights each class equally, and query-weighted averages. Tied cosine
similarities are ordered by the original index position. Precision@5 divides
by `min(5, available candidates)` when fewer than five candidates exist. When a
query has no relevant reference images, Recall@5 and Average Precision are
undefined and excluded from their means; the JSON records defined-query and
contributing-class counts.

The same-index leave-one-out evaluation is an initial baseline, not a held-out
test: the feature index was built from these same images. Duplicate images,
shared plants, or capture sessions may inflate results. The dataset labels
have not been independently confirmed as scientific diagnoses, and the index
does not contain group identifiers needed for group-aware splitting. See the
generated report for the index hash, model metadata, results, and limitations.

From PowerShell at the project root, run the focused metric tests, then the
complete backend suite:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m pytest -q tests\test_evaluation.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Generate the Markdown report and machine-readable per-query JSON from the
existing index:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.evaluation --index "..\data\processed\fnnpk-mobilenet-v3-small.npz" --report "..\docs\cbir-evaluation-report.md" --json "..\data\processed\cbir-evaluation.json"
```

The output files are `docs/cbir-evaluation-report.md` and
`data/processed/cbir-evaluation.json`. Regenerate them with the same command
after changing the index or evaluation code.

## Frozen-feature classification experiment

The optional four-class classifier reuses the saved 576-dimensional
MobileNetV3-Small feature vectors. It trains only a small PyTorch linear layer
on CPU; it does not retrain MobileNet or download model weights. A deterministic
four-fold stratified group cross-validation reports out-of-fold accuracy,
balanced accuracy, macro and per-class precision/recall/F1, and a confusion
matrix. Identical feature vectors are kept in the same fold, and scaling is fit
on each training fold only. Because `FN` has only 12 examples, its fold metrics
are particularly uncertain. Outputs are predictions of source-provided
dataset labels, not confirmed diagnoses or diagnostic certainty.

Run the classification tests and then the full backend suite from the backend
directory:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m pytest -q tests\test_classification.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Train and evaluate the classifier from the existing feature index:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.classification train --index "..\data\processed\fnnpk-mobilenet-v3-small.npz" --model "..\data\processed\fnnpk-linear-classifier.npz" --report "..\data\processed\fnnpk-classification-evaluation.json" --folds 4 --seed 42
```

The model artifact is an array-only NPZ containing the PyTorch linear-layer
parameters, training-set scaling values, and class mapping. The JSON report
contains the index hash, split details, metrics, and limitations. These
generated files are under `data/processed/`, which is ignored by Git.

## Train and evaluate the separate LettuceDeF classifier

The LettuceDeF trainer is independent of the FNNPK training pipeline. It aligns
the feature index to Protocol B by source-relative image path and validates
class, split, checksum, and manifest/index consistency. Only the `train` split
fits feature standardization, class weights, and the linear classifier. The
`val` split selects the checkpoint and epoch using macro F1 (ties use lower
validation cross-entropy). The `test` split is evaluated once after that
checkpoint is fixed and is not used for preprocessing, fitting, or selection.
Training runs on CPU and does not retrain MobileNet.

Run the focused LettuceDeF tests and then the full backend suite:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m pytest -q tests\test_lettucedef_classification.py
& ".\.venv\Scripts\python.exe" -m pytest -q tests\test_lettucedef_protocol_a_classification.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Train and evaluate from PowerShell at the project root:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.lettucedef_classification train --index "..\data\processed\lettucedef-mobilenet-v3-small.npz" --manifest "..\data\raw\LettuceDeF\manifest_protocol_B.csv" --model "..\data\processed\lettucedef-linear-classifier.npz" --report "..\data\processed\lettucedef-classification-evaluation.json" --seed 42 --max-epochs 100 --patience 15
```

The model NPZ contains the linear parameters, training-fitted scaling
parameters, class mapping, and source hashes. The separate JSON report records
split assignments and counts, training configuration, validation history, and
the held-out test metrics. Both output paths refuse overwrite. This command has
not been run as part of adding the pipeline. Metrics describe agreement with
source-provided treatment labels; do not call them diagnostic accuracy.
Protocol B does not establish plant-independent or independent-greenhouse
generalization because stable plant identifiers and separate crop-cycle data
are unavailable.

## Train and evaluate the separate LettuceDeF Protocol A classifier

The Protocol A trainer writes a distinct model and report; it does not replace
the Protocol B artifacts. It validates the Protocol A manifest hash and uses
Protocol A's train/validation/test assignments. The shared feature index was
built from Protocol B, so the trainer also validates the index against its
recorded Protocol B source manifest and the expected MobileNetV3-Small encoder
configuration, then aligns features to Protocol A by image path. The test
split is evaluated once after validation-based checkpoint selection.

Run the focused tests and full backend suite as shown above. To train from
PowerShell at the project root:

```powershell
Set-Location ".\backend"
& ".\.venv\Scripts\python.exe" -m app.lettucedef_protocol_a_classification train --index "..\data\processed\lettucedef-mobilenet-v3-small.npz" --manifest "..\data\raw\LettuceDeF\manifest_protocol_A.csv" --index-source-manifest "..\data\raw\LettuceDeF\manifest_protocol_B.csv" --model "..\data\processed\lettucedef-protocol-a-linear-classifier.npz" --report "..\data\processed\lettucedef-protocol-a-evaluation.json" --seed 42 --max-epochs 100 --patience 15
```

Protocol A metrics are agreement with source-provided treatment labels, not
diagnostic accuracy. This protocol does not establish plant-independent or
independent-greenhouse generalization.

## Run and test the complete upload flow (Windows PowerShell)

Use two PowerShell windows opened in the project root. In the first window,
start the backend:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

In the second window, start the frontend with the local backend URL:

```powershell
cd frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

Open the Vite URL printed in the terminal (normally
<http://localhost:5173/>). Select a JPEG, PNG, or WebP image in **Leaf image
analysis**, then choose **Upload image**. During upload, the button is disabled
and displays a loading state. On success, the page displays the server's upload
ID, filename, detected format, dimensions, and status. This confirms upload
validation. The frontend also shows CBIR results; the classification endpoint
is available separately and is not yet integrated into the frontend.

To try an error, select an unsupported image format or an image larger than
10 MB and upload it. The page shows a specific validation message. To test the
backend-not-running message, stop the backend with `Ctrl+C` and try another
upload. Start it again after testing. Stop the frontend with `Ctrl+C` in its
terminal.

The backend test suite can also be run from a PowerShell window at the project
root:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pytest
```

## Notes

- The backend virtual environment is local to `backend/.venv/` and is ignored by
  Git.
- Reference image files are ignored by default to avoid accidentally committing
  a large dataset. The `data/reference_images/` folder placeholder remains
  tracked; edit `.gitignore` later if selected images should be versioned.
- Dataset exports and generated model files should not be committed.

## Standalone deep feature fusion module

backend/app/feature_fusion.py combines the existing CPU MobileNetV3-Small encoder from backend/app/cbir.py with two handcrafted descriptors. It does not define a second CNN or change the current classifier, CBIR index, API, or frontend.

For each image, the module extracts and normalizes these blocks:

- **Deep:** 576 MobileNetV3-Small values, L2-normalized.
- **Colour:** 48 HSV histogram values (16 bins for each of H, S, and V). The histogram is L1-normalized, then L2-normalized as a feature block.
- **Texture:** 256 LBP histogram values using 8 neighbors at radius 1. The histogram is L1-normalized, then L2-normalized as a feature block.
- **Fused:** concatenation of the three unit-norm blocks, followed by final L2 normalization; total dimension **880**.

Handcrafted descriptors use a 224 x 224 center-cropped image to keep processing lightweight. The module exposes DeepFeatureFusionEncoder.extract_features(image) for one fused NumPy vector and extract_blocks(image) for the separately normalized blocks. This is a standalone component and is not wired into the existing FNNPK classifier or retrieval index.

## Build the separate FNNPK fused-feature index

The standalone fused index stores an 880-dimensional vector for each of the
208 FNNPK images. It combines the existing 576-dimensional MobileNetV3-Small
features with HSV histograms and LBP texture features. Images are processed one
at a time. This command does not change the existing MobileNet-only index,
classifier, or evaluation report. The default output is
data/processed/fnnpk-fused-mobilenet-hsv-lbp.npz; an existing output is never
overwritten. The first run downloads the pretrained MobileNet weights if they
are not already cached.

From PowerShell at the project root:

    Set-Location ".\backend"
    & ".\.venv\Scripts\python.exe" -m app.fnnpk_fused_index

Use --dataset and --output to select different input/output paths when needed.
