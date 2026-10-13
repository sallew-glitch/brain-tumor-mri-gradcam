# Running on Google Colab (free GPU)

Use Colab for the bigger models (ResNet-50, DenseNet-121, EfficientNet) and full 5-fold runs.
A Colab T4 GPU is roughly 20-50x faster than a laptop CPU for this workload.

## 1. One-time preparation (on your laptop)

Zip the processed data (about 90 MB). From the project folder in PowerShell:

```powershell
tar -a -c -f processed.zip -C data processed
```

(`tar` is built into Windows 10/11. It makes a zip that Linux reads correctly; PowerShell's
`Compress-Archive` uses Windows-style paths that Colab's `unzip` can mishandle.)

Upload `processed.zip` to Google Drive into a folder named `btmri`, so it ends up at
`My Drive/btmri/processed.zip`.

## 2. Open the notebook in Colab

Open this link (it loads the notebook straight from GitHub):

<https://colab.research.google.com/github/sallew-glitch/brain-tumor-mri-gradcam/blob/main/notebooks/colab_experiments.ipynb>

Then:

1. `Runtime -> Change runtime type -> T4 GPU -> Save`.
2. `Runtime -> Run all`. Approve the Google Drive pop-up when it appears.

The notebook installs the code, unpacks the data, runs the tests, trains each model on fold 1,
runs Grad-CAM on the validation fold, and prints the comparison tables. Results are written to
`My Drive/btmri/runs`.

To keep your own edits to the notebook, use `File -> Save a copy in Drive`.

## Tips

* **Free-tier limits:** sessions end after ~90 minutes of inactivity or several hours in
  total, and GPU access can be temporarily unavailable after heavy use. Results already saved
  to Drive are kept. Re-run the notebook and skip models that already finished.
* **Private repo:** clone with a GitHub personal access token,
  `https://<token>@github.com/<account>/<repo>.git`, and never commit or share the token.
* **Code changes:** push from your laptop, then re-run the setup cell (it runs `git pull`).
* **Inspect Colab results on your laptop:** download the run folder from Drive into
  `outputs/runs/`; `compare_runs.py` and `explain.py` work on it as usual.
* GPU results differ very slightly from CPU results (different numeric routines), so compare
  models trained on the same kind of hardware.
