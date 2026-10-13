# Design

This document explains *why* the code is structured the way it is, so collaborators can
extend it without breaking its guarantees.

## Research question

1. How well do ImageNet-pretrained CNNs classify meningioma / glioma / pituitary tumors
   on MRI when evaluated **without patient leakage**?
2. Do their Grad-CAM explanations actually point at the tumor? We measure this against the
   expert-drawn tumor masks instead of only showing hand-picked heatmaps.

## Pipeline

```
 data/raw/*.mat ──prepare_data──▶ data/processed/ (PNG + metadata.csv)
                                         │
                       configs/*.yaml ───┤
                                         ▼
                       train.py: for each test fold k (1..5)
                         split by official folds: test=k, val=k+1, train=rest
                         build timm model (ImageNet weights) ─▶ Trainer.fit
                         pick best epoch on val ─▶ evaluate once on test
                                         │
                                         ▼
                       outputs/runs/<run>/fold_k/best.pt, metrics, predictions
                                         │
                       explain.py ───────┴─▶ Grad-CAM + localization vs. masks
```

## Key decisions

| Decision | Why |
|---|---|
| **Convert `.mat` to PNG once** | MATLAB/HDF5 parsing happens in one place; training code only sees ordinary images. Converting is quick and repeatable. |
| **Use the official `cvind.mat` folds** | They are patient-disjoint (verified: 0 shared patients) and make results comparable with the literature. `data.fold_source: stratified_group` regenerates folds if ever needed. |
| **Test = fold k, val = fold k+1** | Validation is also patient-disjoint, so choosing the best epoch never sees test patients. |
| **Per-image percentile intensity normalisation** | MRI intensities have no absolute scale; clipping at the 0.5/99.5 percentiles is robust to bright outlier pixels. |
| **Store at 256 px, train at 224 px** | 224 is the native resolution of most ImageNet models; 256 leaves room for crop augmentation and keeps data small (~90 MB). |
| **Grayscale replicated to 3 channels + ImageNet normalisation** | Lets us reuse pretrained weights without modifying the first layer. |
| **`timm` for all models** | One function builds any of hundreds of architectures by name. Grad-CAM uses timm's `forward_features`/`forward_head` split, so it works for every CNN without per-model layer names. |
| **Freeze backbone for epoch 1, then fine-tune everything at the same LR** | The new head starts random; freezing first stops its large early gradients from damaging pretrained features. A 0.1x backbone LR was tried first and underfit (fold 1: 84% vs 93% test accuracy), because MRI differs a lot from ImageNet photos. |
| **Class-weighted cross-entropy; select by macro-F1** | Glioma is 46% of slices. Without weighting, the model would lean toward glioma. |
| **Typed dataclass configs, unknown keys rejected** | A typo like `train.lrr` fails loudly instead of silently running the defaults. The resolved config is saved with every run. |
| **Logic in `src/btmri`, `scripts/` are thin CLIs** | Everything is importable from notebooks (Colab) and testable. |
| **Dataset returns dicts** | New fields (masks, metadata, ...) can be added without breaking existing code. |

## Known dataset caveat: possible site confound

The README says the images come from two hospitals. Patient IDs come in two styles:
`MR0…` (1,221 slices, **all glioma**) and numeric (205 glioma + all meningioma and pituitary
slices). So glioma is largely confounded with acquisition source. A model could partly learn
"which scanner" instead of "which tumor". Ways to probe this, all good paper material:

* compare Grad-CAM localization for glioma vs. other classes (`gradcam/summary.json`),
* evaluate glioma accuracy separately for MR-prefixed vs. numeric-ID patients,
* check whether background-only crops (tumor masked out) still predict glioma.

Mention it as a limitation in any write-up.

## How to extend

| Want to... | Do this |
|---|---|
| Try another architecture | New YAML with `model.name: <any timm CNN>`, e.g. `convnext_tiny`. No code changes. |
| Tune a hyperparameter | `--set train.lr=1e-4`, or a new YAML inheriting `_base_: base.yaml`. |
| Add a config option | Add a field to the right dataclass in `config.py` (with a default), use it, and document it in `base.yaml`. |
| Add an augmentation | `data/transforms.py::build_train_transform`, plus a field in `AugmentConfig`. |
| Add an explainability method (Grad-CAM++, Score-CAM, ...) | New class in `explain/` with the same call signature as `GradCAM` (`images -> (cams, logits)`); the localization metrics and runner work unchanged. |
| Support Vision Transformers in Grad-CAM | Reshape the `(B, tokens, C)` features to a `(B, C, h, w)` grid in `GradCAM` (drop the class token). |
| Add another dataset | Write a converter producing the same `metadata.csv` columns; everything downstream works unchanged. Keep splits patient-disjoint. |
| Add a metric | `training/metrics.py::compute_metrics`; add it to `SCALAR_METRICS` to average it across folds. |

## Guarantees that must not break

* No patient appears in more than one of train / val / test (`assert_patient_disjoint`
  runs on every split, and is tested).
* The test fold is used exactly once per fold, after model selection.
* Every run directory contains the full resolved config.
