# Running on Google Colab (free GPU)

Use Colab for the bigger models (ResNet-50, DenseNet-121, EfficientNet) and full 5-fold
runs. A Colab T4 GPU is roughly 20-50x faster than a laptop CPU for this workload.

## One-time preparation (on your laptop)

1. Push this repository to GitHub (a private repo is fine).
2. Zip the processed data (about 90 MB) and upload it to Google Drive:
   ```powershell
   Compress-Archive -Path data\processed -DestinationPath processed.zip
   ```
   Upload `processed.zip` to `My Drive/btmri/`.

## In a Colab notebook

`Runtime -> Change runtime type -> T4 GPU`, then run these cells:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/<your-account>/<your-repo>.git project
%cd project
!pip install -q -e .
!unzip -q /content/drive/MyDrive/btmri/processed.zip -d data/
!pytest -q
```

```bash
# Results are written straight to Drive so they survive the session ending.
!python scripts/train.py --config configs/resnet50.yaml --set output_dir=/content/drive/MyDrive/btmri/runs
```

```bash
!python scripts/explain.py --run-dir /content/drive/MyDrive/btmri/runs/<run_name>
!python scripts/compare_runs.py --runs-dir /content/drive/MyDrive/btmri/runs
```

## Tips

* Private repo: clone with a GitHub personal access token,
  `https://<token>@github.com/<account>/<repo>.git`, and never commit the token.
* Colab sessions disconnect after inactivity or a few hours. Run one model per session, or
  use `--folds 1 2` and `--folds 3 4 5` in separate sessions.
* To inspect a Colab-trained run on your laptop, download the run folder and pass
  `--processed-dir data/processed` to `explain.py`.
