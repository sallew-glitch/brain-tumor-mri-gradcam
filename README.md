# Brain Tumor MRI Classification with Transfer Learning and Grad-CAM

Classifies brain MRI slices into **meningioma**, **glioma** or **pituitary tumor** by
fine-tuning an ImageNet-pretrained CNN. Then uses **Grad-CAM** to check, against
radiologist-drawn tumor outlines, whether the model actually looks at the tumor.

**What this project does differently from many others on this dataset:**

* **No patient leakage.** It uses the dataset's official patient-disjoint 5-fold split, so no
  patient's scans appear in both training and testing.
* **Grad-CAM is measured, not just shown.** Heatmaps are scored against the expert tumor masks
  (pointing game, energy inside the tumor, IoU).
* **Reproducible.** Every run saves its exact configuration, checkpoint, metrics and plots.

New to machine learning? Start with [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Results

| Model | Evaluation | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| ResNet-18 (ImageNet, full fine-tune) | Fold 1 only (preliminary) | 93.2% | 0.922 | 0.995 |
| ResNet-18 (ImageNet, full fine-tune) | 5-fold CV | *pending* | | |

Grad-CAM (fold 1): the heatmap's peak falls inside the tumor in only about 11% of test slices.
This suggests the model relies largely on location and surrounding anatomy. Investigating
this is part of the project; see [docs/DESIGN.md](docs/DESIGN.md).

## Dataset

Cheng et al., [Brain Tumor Dataset (Figshare)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427):
3,064 T1-weighted contrast-enhanced slices from 233 patients.

| Class | Slices | Patients |
|---|---|---|
| Glioma | 1,426 | 89 |
| Meningioma | 708 | 82 |
| Pituitary tumor | 930 | 62 |

The data is **not** included in this repository. Download it yourself (instructions below).

## Setup

Requires Python 3.10+. A GPU is optional. ResNet-18 trains on a laptop CPU in about 30 minutes
per fold.

```bash
git clone https://github.com/<your-username>/brain-tumor-mri-gradcam.git
cd brain-tumor-mri-gradcam
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest          # all tests should pass
```

Download `1512427.zip` from Figshare and arrange it like this:

```
data/raw/
    cvind.mat                    (from the downloaded zip)
    mat/1.mat ... mat/3064.mat   (extracted from the four brainTumorDataPublic_*.zip inside it)
```

## Usage

```bash
# 1. Convert .mat files to PNG images + metadata.csv (~1 min)
python scripts/prepare_data.py

# 2. Train and test with 5-fold cross-validation (~2.5 h on CPU, minutes on a GPU)
python scripts/train.py --config configs/resnet18_cpu.yaml
python scripts/train.py --config configs/resnet18_cpu.yaml --folds 1        # one fold only
python scripts/train.py --config configs/resnet18_cpu.yaml --set train.lr=1e-4   # change a setting

# 3. Grad-CAM heatmaps + localization scores
#    Compare models/layers on the validation fold; use --split test only for the final report.
#    Layers: last (7x7 grid), penultimate (14x14 grid), or a module name such as layer3.
python scripts/explain.py --run-dir outputs/runs/<run_name> --split val --layers last penultimate

# 4. Compare runs (choose settings by the VAL column, not test)
python scripts/compare_runs.py
```

For larger models on Google Colab's free GPU, see [docs/COLAB.md](docs/COLAB.md).

Each run writes to `outputs/runs/<name>_<timestamp>/`:

```
config.yaml, train.log, summary.json       settings, log, cross-validation summary
fold_k/best.pt                             best checkpoint (chosen on the validation fold)
fold_k/curves.png, confusion_matrix.png    training curves, test confusion matrix
fold_k/test_metrics.json, predictions.csv  test scores, per-image probabilities
fold_k/gradcam_<split>_<layer>/            heatmap examples and localization scores
```

## Method

| Step | Choice |
|---|---|
| Preprocessing | Per-image intensity normalisation (0.5-99.5 percentile), resize to 256, grayscale copied to 3 channels |
| Augmentation | Random crop (85-100%), horizontal flip, rotation ±10°, brightness/contrast jitter |
| Model | ImageNet-pretrained CNN from `timm` with a new 3-class head |
| Training | AdamW, cosine learning-rate schedule, head-only first epoch, class-weighted cross-entropy |
| Model selection | Best validation macro-F1 epoch; early stopping (patience 5) |
| Evaluation | Official patient-disjoint 5 folds: test = fold k, validation = fold k+1, train = the rest |
| Explainability | Grad-CAM on the last or second-to-last stage, scored against tumor masks |

Design decisions and how to extend the code are in [docs/DESIGN.md](docs/DESIGN.md).

## Project structure

```
configs/       experiment settings (YAML, inherit from base.yaml)
scripts/       command-line entry points
src/btmri/     the library: data/, models/, training/, explain/, config.py, viz.py
tests/         pytest suite (uses synthetic data; no download needed)
docs/          walkthrough, design notes, Colab guide
```

## Known limitations

* **Possible site confound:** most glioma scans come from patients with `MR`-prefixed IDs. No
  meningioma or pituitary scan has one, which suggests a different hospital or scanner. A model
  could partly learn scanner differences instead of tumor appearance.
* Single 2-D slices only; no 3-D context or other MRI sequences.
* Research code. **Not for clinical use.**

## Citation

If you use the dataset, cite:

> Cheng, J. et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region
> Augmentation and Partition." *PLoS ONE* 10(10): e0140381, 2015.
