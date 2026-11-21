# CBIR Baseline Evaluation

## Summary

This report evaluates the existing MobileNetV3-Small feature index using leave-one-image-out retrieval. Relevance is based on matching the dataset's supplied folder labels. It does **not** validate nutrient diagnoses.

- Indexed images / queries: 208
- Feature dimension: 576
- Model: `torchvision.models.mobilenet_v3_small`
- Weights: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`
- Preprocessing: TorchVision IMAGENET1K_V1 transforms; RGB; resize 256; center crop 224; scale to [0, 1] and ImageNet mean/std normalization
- Input index: `data/processed/fnnpk-mobilenet-v3-small.npz`
- Input index SHA-256: `6273be2f7211c22ab1892b477afaf92d82c7a7ac1216192fd26659bb264058e4`

## Dataset labels

| Supplied folder label | Image / query count |
|---|---:|
| `FN` | 12 |
| `-N` | 58 |
| `-P` | 66 |
| `-K` | 72 |

## Results

| Class / aggregation | Queries / classes | Precision@1 | Precision@5 | Recall@5 | mAP |
|---|---:|---:|---:|---:|---:|
| `FN` | 12 | 0.9167 | 0.7833 | 0.3561 | 0.6462 |
| `-N` | 58 | 0.9483 | 0.8448 | 0.0741 | 0.5977 |
| `-P` | 66 | 0.9394 | 0.8727 | 0.0671 | 0.5060 |
| `-K` | 72 | 0.9444 | 0.8444 | 0.0595 | 0.5209 |
| **Query-weighted mean** | 208 queries | 0.9423 | 0.8500 | 0.0831 | 0.5448 |
| **Macro mean across classes** | 4 classes | 0.9372 | 0.8363 | 0.1392 | 0.5677 |

## Evaluation methodology

- Every indexed embedding is used once as a query; the saved features are reused without model inference.
- The query is excluded from its candidate set by its exact index position.
- A reference is relevant when its supplied folder label matches the query label.
- Ranking is by descending cosine similarity; tied scores use ascending original index position.
- Precision@1 is relevant results at rank 1 divided by 1.
- Precision@5 is relevant results in the first five divided by `min(5, available candidates)`.
- Recall@5 is relevant results in the first five divided by the number of relevant references after query exclusion.
- AP averages precision at each relevant rank and divides by the total number of relevant references; mAP is mean AP across eligible queries.
- Macro results give each class average equal weight. Query-weighted means are also included for reference.
- If there are no relevant references, Recall@5 and AP are `N/A` and excluded from means; defined-query and contributing-class counts are recorded in the JSON.

## Limitations

- Relevance uses the dataset's supplied folder labels; these labels are not independently verified nutrient diagnoses.
- Leave-one-image-out evaluates queries against the same 208 images used to build the feature index, so it is not an independent held-out evaluation.
- Near-duplicates, shared plants, or shared capture sessions may inflate retrieval scores; the available index has no group IDs to support group-aware splitting.

This is an initial same-dataset retrieval baseline, not evidence of generalization to unseen plants or growing environments. A stronger evaluation requires duplicate review and plant/capture-group-aware query/reference separation where group metadata is available.

Machine-readable per-query results and metric definitions are in `data/processed/cbir-evaluation.json`.
