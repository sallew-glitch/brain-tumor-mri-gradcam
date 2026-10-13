# How this project works (simple version)

## The goal in one sentence

Show the computer thousands of brain scans labelled "meningioma", "glioma" or "pituitary",
let it learn the difference, then test it on scans from **people it has never seen**, and
check **where in the image** it looked when deciding.

## The 4 steps

```
 1. PREPARE        2. TRAIN              3. TEST              4. EXPLAIN
 .mat files   ->   model studies    ->   model sits a    ->   heatmap: where did
 become PNGs       training scans        final exam on        it look? compared with
                   for many rounds       unseen patients      the doctor's outline
```

| Step | Command | Main file |
|---|---|---|
| 1. Prepare | `python scripts/prepare_data.py` | `src/btmri/data/convert.py` |
| 2+3. Train and test | `python scripts/train.py --config configs/resnet18_cpu.yaml` | `src/btmri/training/` |
| 4. Explain | `python scripts/explain.py --run-dir outputs/runs/<run>` | `src/btmri/explain/` |
| Compare runs | `python scripts/compare_runs.py` | |

## Step 1: Prepare

* Every scan is stored in a MATLAB file. We turn each one into a normal PNG picture.
* Every scan's brightness is stretched to 0-255, because different scanners use different scales.
* The doctor's tumor outline (mask) is also saved as a black-and-white PNG.
* `metadata.csv` is a table with one row per scan: its patient, its tumor type, its fold.

## Step 2: Split the patients (the most important rule)

* The 233 patients are split into **5 folds** (groups of ~47 patients). The dataset's
  authors made these groups.
* A patient's scans always stay together in one fold. Otherwise the model could "recognise
  the patient" in the exam instead of the tumor. That would be cheating without anyone noticing.
* In each round, one fold is the **exam** (test), one is the **practice test** (validation),
  and three are the **textbook** (training).

## Step 3: Train

* **The model:** ResNet-18, a network that has already learned to see edges, textures and
  shapes from 1.2 million everyday photos. We swap its last layer (which chose between 1000
  photo categories) for a new one that chooses between our 3 tumor types.
* **One epoch:** the model looks at every training scan once, in small batches of 32.
* **After each batch:** it compares its guesses with the true answers (the *loss*), then
  nudges its weights to be a little less wrong. The *learning rate* is the size of each nudge.
* **After each epoch:** it takes the practice test (validation). If that is its best score so
  far, we save a copy of the model (`best.pt`).
* **At the end:** the best saved copy takes the exam (test fold), once.

## Step 4: Repeat 5 times (cross-validation)

Each fold takes a turn being the exam, so every patient is tested exactly once. We report the
average of the 5 exam scores, plus how much they vary.

## Step 5: Explain (Grad-CAM)

For a test scan, Grad-CAM colours the regions that pushed the model towards its answer (red =
strong). We then check, with the doctor's outline: does the red land on the tumor?

## Where results go

`outputs/runs/<name>_<date>/fold_1/`:
* `curves.png`: how training went, epoch by epoch
* `confusion_matrix.png`: which tumor types get mixed up
* `test_metrics.json`: the exam scores
* `gradcam/examples.png`: heatmaps

## Glossary

| Word | Simple meaning |
|---|---|
| Weights | The millions of numbers inside the model; learning = adjusting them |
| Loss | A single number for "how wrong" the model is; training makes it smaller |
| Learning rate | How big each adjustment is |
| Epoch | One full pass over the training scans |
| Batch | The small group of scans (32) handled before each adjustment |
| Fold | One of the 5 groups of patients |
| Overfitting | Memorising the training scans instead of learning the general pattern |
| Underfitting | Not having learned enough yet |
| Tensor | A block of numbers with several directions (axes), like a stack of spreadsheets |
