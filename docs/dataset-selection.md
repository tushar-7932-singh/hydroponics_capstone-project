# Dataset record and selection guidance for Hydroponics CBIR

## Purpose

This document records the dataset currently used by the project and provides
guidance for assessing its suitability and any future candidate datasets. A
dataset is a collection of images and the information that tells us what each
image shows. A poor or poorly documented dataset can make even a well-written
model give misleading results.

## Current dataset

The project uses the **Lettuce NPK dataset** published on
[Kaggle](https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset). The
source documentation describes hydroponic lettuce categories as Fully
Nutritional, Nitrogen Deficient, Phosphorus Deficient, and Potassium Deficient.
The local dataset has the following folder labels and image counts:

| Folder label | Meaning stated in source documentation | Local images |
| --- | --- | ---: |
| `FN` | Fully Nutritional | 12 |
| `-N` | Nitrogen Deficient | 58 |
| `-P` | Phosphorus Deficient | 66 |
| `-K` | Potassium Deficient | 72 |
| **Total** |  | **208** |

Kaggle metadata lists the license as **CC0: Public Domain**. Retain the source
attribution and verify the applicable terms before redistributing images; do
not assume the metadata alone settles every use or redistribution question.

### Known limitations

- The class names and meanings above come from the dataset's source
  documentation. They have not been independently validated by this project;
  they must not be described as scientifically confirmed diagnoses.
- The source does not provide detailed treatment records, nutrient
  measurements, or independent expert validation. Its description says the
  images were collected in a controlled environment for a hydroponic lettuce
  deficiency project, but does not give enough detail to independently assess
  those conditions.
- The class distribution is small and imbalanced, especially `FN` (12 images)
  compared with the deficiency classes (58–72 images each).
- Five exact duplicate pairs were found: one pair in `-N` and four pairs in
  `-K`. Keep duplicate copies together or remove redundant copies when
  constructing evaluation splits to reduce data leakage.
- The project has an existing CPU MobileNetV3-Small CBIR feature extractor
  and saved feature index. Its similarity scores indicate visual similarity;
  they are not probabilities or diagnoses. A classifier is a separate future
  experiment and would need leakage-aware evaluation.

## Dataset manifest template

Use [`data/metadata/dataset_manifest_template.csv`](../data/metadata/dataset_manifest_template.csv)
as a starting point for recording image-level metadata. It contains one
marker-only example row; that row is **not a real dataset record** and should
be removed before entering actual records. All other fields in that row are
intentionally blank. Do not infer missing values or invent IDs, treatments,
measurements, or label verification.

### Manifest field guide

| Field | What to record |
| --- | --- |
| `image_filename` | The image's filename. |
| `relative_image_path` | Its project-relative path, such as a path beneath `data/raw/`; never store an absolute computer-specific path. |
| `class_code` | The label code assigned by the documented labeling process. Use the existing codes only when supported: `FN`, `-N`, `-P`, or `-K`. |
| `class_name` | The readable name for the assigned class, consistent with `class_code`. |
| `plant_id` | A stable, non-identifying ID for the individual plant. Use the same ID for all images of that plant. |
| `experiment_or_trial_id` | The ID of the growing experiment, treatment trial, or independent collection trial. |
| `capture_session_id` | An ID shared by images captured in the same session or sequence. |
| `capture_date` | The date of image capture, preferably ISO format `YYYY-MM-DD`; leave blank if unknown. |
| `crop_species` | The species represented in the image, if known. |
| `growing_system` | The growing setup, such as hydroponic system type, soil, or unknown. Record what is known rather than inferring it from the image. |
| `nutrient_treatment` | The documented treatment applied to the plant, not a treatment guessed from appearance. |
| `measured_pH` | Measured pH with units or measurement context documented consistently elsewhere. |
| `measured_EC` | Measured electrical conductivity, including units (for example, mS/cm). |
| `water_temperature` | Recorded water or nutrient-solution temperature, including units (for example, °C). |
| `image_source` | Where the image came from, such as the source dataset URL, a local collection, or a contributor-provided source. |
| `label_source` | Who or what supplied the label, such as dataset documentation, experiment records, or expert review. |
| `label_verified` | Whether the label was independently checked, and by what process. Record a clear value such as `yes`, `no`, or `unknown`; do not mark it verified without evidence. |
| `notes` | Relevant caveats, image quality issues, duplicate links, ambiguity, or other context. |

### Fields important for group-based evaluation

`plant_id` is especially important: images of one plant must not be split
between training and test data when evaluating performance on new plants.
`experiment_or_trial_id` helps hold out entire treatment trials or growing
batches. `capture_session_id` helps keep repeated frames or images from one
photo session together. Record all three whenever possible. Also identify
exact and near-duplicate images; a shared group ID or a note linking duplicate
files can prevent copies from leaking across evaluation splits. Capture date
and treatment metadata provide useful context but are not substitutes for
plant/trial/session IDs.

Do not assign an image of an unknown, unsupported, mixed, diseased, pest-
damaged, or physically damaged condition to one of the four existing nutrient
labels unless the labeling evidence supports that class. Leave the class fields
unassigned until reviewed, or manage such images in a separate, explicitly
documented out-of-scope collection. The current four-class classifier does not
recognize an unknown class.

## What this project needs

The project is about recognizing nutrient-deficiency symptoms in plants grown
in **hydroponic systems**. A useful candidate should ideally include:

- Clear photographs of leaves from hydroponically grown plants.
- The plant species for each image.
- A recorded label for each nutrient condition, including healthy plants.
- Enough examples in each condition to compare similar images fairly.
- Reliable information about how the plants were grown and how labels were
  assigned.
- Permission for the intended academic use.

A dataset of field-grown or potted plants may still be useful for early
experiments, but it is not automatically representative of hydroponics. Growth
environment, lighting, background, camera, plant age, and nutrient delivery
can all affect how leaves look. Record these differences rather than assuming
that a dataset is hydroponic.

## Assessment checklist

### 1. Growing environment

- [ ] Does the dataset state how the plants were grown?
- [ ] Are the images from hydroponics, soil, greenhouse pots, fields, or a mix?
- [ ] If hydroponics is claimed, is the system or growing method described?
- [ ] Are growing conditions recorded, such as nutrient solution, lighting,
  growth stage, or capture location?
- [ ] Could the differences between the dataset and our intended hydroponic
  system affect the visible symptoms?

Do not label a dataset "hydroponic" just because it contains green leaves or
was photographed in a greenhouse. If its growing method is unknown, record it
as unknown.

### 2. Species and condition labels

Make a list of the plant species and all labels present. Nutrient labels might
include named deficiencies, but the exact names must come from the dataset
documentation—not from guessing based on leaf colour.

- [ ] Which plant species are included?
- [ ] Which nutrient conditions or deficiencies are labelled?
- [ ] Are there combined deficiencies, other diseases, pests, or abiotic
  stresses?
- [ ] Are labels consistently defined across species and images?
- [ ] Are healthy plants included as a clearly defined condition?

Healthy examples matter: without them, a later system may have no reference for
what a non-deficient leaf looks like. A missing label is not the same as a
healthy label.

### 3. Image counts and class balance

Count images in every class (each species/condition combination, where
possible). A class with many images can dominate training or retrieval results;
a class with very few examples may not support reliable conclusions.

- [ ] Record the total number of images.
- [ ] Record the number of images for each species and each label.
- [ ] Identify classes with very few or no examples.
- [ ] Check whether some classes mostly come from one camera, site, or capture
  session.
- [ ] Decide whether the dataset is large and balanced enough for the planned
  evaluation. Do not hide class imbalance by reporting only a total count.

### 4. Image quality, duplicates, and labels

Review a sample from every class, including images from different sources or
capture sessions. If practical, review all images in a small dataset.

- [ ] Can the leaf and its visible symptoms be seen clearly?
- [ ] Are images badly blurred, too dark, overexposed, or mostly obstructed?
- [ ] Are there images that are not leaves or are not the stated species?
- [ ] Are exact duplicate files present?
- [ ] Are there near-duplicates, such as repeated frames, crops, or edited
  copies of the same original photograph?
- [ ] Do the image contents appear consistent with their recorded labels?
- [ ] Are uncertain or questionable examples recorded for later review rather
  than silently relabelled?

Duplicates can make evaluation look better than it really is if nearly
identical images end up on both sides of a data split. A visual similarity
review and, where available, file hashes or image-similarity checks can help
find duplicates. Any automated duplicate check still needs human review.

### 5. Scientific basis for labels

A visible symptom can have more than one possible cause. Yellowing, spots, or
curling alone do not prove a specific nutrient deficiency. Prefer labels
supported by documented growing treatments, measurements, expert assessment,
or a cited scientific protocol.

- [ ] Does the dataset explain who assigned the labels and how?
- [ ] Were nutrient conditions controlled or otherwise confirmed?
- [ ] Are measurements, treatment records, expert review, or published
  references provided?
- [ ] Are labels based only on visual appearance? If so, record this limitation
  and do not treat them as confirmed diagnoses.
- [ ] Are uncertain, mixed, or unverified labels identified?

Do not invent or "correct" scientific labels from appearance alone. Keep the
dataset's original label and document any concerns separately.

### 6. Licence, attribution, and permitted use

Check the terms before downloading, copying, modifying, or redistributing any
dataset or its images. Publicly viewable does not mean free to reuse.

- [ ] Record the exact source page or repository URL.
- [ ] Record the licence name and link to its text.
- [ ] Check whether academic research, modification, and local storage are
  allowed.
- [ ] Check attribution, citation, and redistribution requirements.
- [ ] Check for non-commercial, share-alike, or other restrictions that affect
  the project or its eventual demonstration.
- [ ] Confirm whether the licence applies to every image, not just the dataset
  website or metadata.
- [ ] If terms are missing or unclear, ask the provider or choose another
  dataset before using it.

Keep the required citation and attribution with the project records. Do not
commit or publish images unless their licence permits that use.

### 7. Is it suitable for CBIR?

Classification asks, "Which label should this image receive?" Content-based
image retrieval (CBIR) asks, "Which reference images look visually similar to
this query image?" A dataset designed only for classification may lack the
information needed to test retrieval well.

- [ ] Are there enough images within each relevant species and condition to
  form useful reference groups?
- [ ] Can we identify which images show the same plant, leaf, treatment, or
  capture session?
- [ ] Can we select query images that are separate from the reference images?
- [ ] Is there a sensible way to judge whether retrieved images are relevant,
  such as shared verified condition and species, expert relevance labels, or a
  documented experiment?
- [ ] Does the dataset include real variation in viewpoint, lighting, age, and
  background, rather than only near-identical copies?
- [ ] Are labels and metadata detailed enough to filter or compare retrieved
  examples fairly?

If the dataset has class labels but no useful image identities or retrieval
relevance information, it may help with a classifier experiment but may not be
enough to evaluate CBIR. Record that limitation and decide how retrieval
relevance could be established before claiming retrieval performance.

### 8. Preventing data leakage

Data leakage happens when information from an evaluation image accidentally
appears in the data used to build or tune the system. For example, putting one
photo in training and a near-identical copy in testing can make results
misleadingly strong.

- [ ] Find available group identifiers such as plant, leaf, treatment,
  greenhouse, camera, date, and capture session.
- [ ] Keep all images from the same plant, original photograph, or capture
  sequence in only one split whenever possible.
- [ ] Find and group duplicates and near-duplicates before making splits.
- [ ] Separate the data into training, validation, and test sets before model
  tuning. Use validation data for choices during development; reserve test data
  for final evaluation.
- [ ] For CBIR evaluation, keep each query image out of its own reference
  collection. Do not let a query retrieve itself or a duplicate.
- [ ] If the dataset is small, choose a group-aware evaluation strategy and
  clearly state its limits instead of randomly splitting individual images.
- [ ] Save the split rules and image IDs so the evaluation can be repeated.

Randomly splitting individual images is not safe if related images can land in
different sets. Split by the strongest available group (for example plant or
capture session) to test whether the system works on genuinely separate
examples.

## Existing baseline and possible classification approach

The project already uses pretrained MobileNetV3-Small to extract deep feature
vectors for CBIR. The saved 576-dimensional features can also serve as inputs
to an initial lightweight classifier experiment without training or
fine-tuning the image model. Such an experiment should use class balancing,
keep duplicate pairs together across data splits, and report per-class and
macro metrics. With only 12 `FN` images and no detailed plant or capture
session identifiers, any measured performance will have substantial
uncertainty.

Fine-tuning a deep classifier is not recommended as the first step for this
small dataset. Treat any future classifier output as a prediction of the
dataset-provided labels, not as an independently verified nutrient diagnosis.

## Dataset record checklist

Complete one record for each candidate dataset. Write "unknown" when
information is not provided; do not guess.

- [ ] Dataset name:
- [ ] Source URL:
- [ ] Date accessed:
- [ ] Dataset version or release:
- [ ] Licence name and licence URL:
- [ ] Required attribution or citation:
- [ ] Academic-use and redistribution restrictions:
- [ ] Growing environment (hydroponic, soil, field, mixed, or unknown):
- [ ] Growing or capture conditions described:
- [ ] Plant species:
- [ ] Class labels exactly as published:
- [ ] Is a healthy class included? How is it defined?
- [ ] Total image count:
- [ ] Image count for each species/label:
- [ ] Image quality concerns:
- [ ] Duplicate or near-duplicate concerns:
- [ ] Label source and scientific evidence:
- [ ] Can plant/session/image identities be grouped for safe splitting?
- [ ] Is there a usable CBIR query/reference and relevance evaluation plan?
- [ ] Known limitations or possible mismatch with our hydroponic project:
- [ ] Decision: accept for evaluation, use only for limited exploration, or
  reject (with reason):

## Selection decision

Do not choose a dataset based only on its image count or its name. Compare the
completed records, prioritize reliable labels, suitable growing conditions,
clear reuse rights, and an evaluation plan that prevents leakage. If no single
dataset meets every need, document the gap and discuss it before combining
sources or collecting new images.
