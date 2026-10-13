"""Convert the raw Figshare ``.mat`` files into PNG images + a metadata table.

The raw files are MATLAB v7.3 (HDF5) structs with fields ``image`` (int16), ``label``
(1=meningioma, 2=glioma, 3=pituitary), ``PID`` (patient id stored as char codes) and
``tumorMask``. Converting once up front means training never touches MATLAB formats and
every collaborator works from the same, simple representation:

    data/processed/
        images/<id>.png      8-bit grayscale MRI slice
        masks/<id>.png       tumor mask (0 or 255)
        metadata.csv         one row per slice: id, patient, label, official fold, paths
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from btmri.constants import CLASS_NAMES, METADATA_FILENAME


@dataclass
class MatRecord:
    image: np.ndarray  # (H, W) int16
    mask: np.ndarray  # (H, W) uint8 in {0, 1}
    label: int  # 0-based class index
    patient_id: str


def read_mat(path: str | Path) -> MatRecord:
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        # HDF5 stores MATLAB arrays column-major, so they come out transposed.
        image = np.array(cj["image"]).T
        mask = np.array(cj["tumorMask"]).T.astype(np.uint8)
        label = int(np.array(cj["label"]).item()) - 1
        patient_id = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
    return MatRecord(image=image, mask=mask, label=label, patient_id=patient_id)


def read_official_folds(cvind_path: str | Path) -> np.ndarray:
    """Fold number (1..5) for each image, in image-id order (index 0 -> 1.mat)."""
    with h5py.File(cvind_path, "r") as f:
        return np.array(f["cvind"]).flatten().astype(int)


def to_uint8(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Rescale intensities to 0..255, clipping the extreme 0.5% at each end.

    MRI intensities have no fixed scale (they vary by scanner and patient), so each slice is
    normalised on its own. Percentile clipping instead of plain min/max stops a handful of
    very bright pixels from making the whole image dark.
    """
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (image * 255).round().astype(np.uint8)


def convert_dataset(raw_dir: str | Path, out_dir: str | Path, size: int = 256) -> pd.DataFrame:
    """Convert every ``<raw_dir>/mat/*.mat`` file and write ``metadata.csv``.

    Args:
        raw_dir: folder containing ``mat/`` (the extracted slices) and ``cvind.mat``.
        out_dir: destination folder (created if missing).
        size: images are resized to ``size x size``. 256 is plenty for 224px models and keeps
            the dataset small; use 512 to keep full resolution.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    mat_files = sorted((raw_dir / "mat").glob("*.mat"), key=lambda p: int(p.stem))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {raw_dir / 'mat'}")
    folds = read_official_folds(raw_dir / "cvind.mat")
    if len(folds) != len(mat_files):
        raise ValueError(f"cvind.mat has {len(folds)} entries but found {len(mat_files)} .mat files")

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    for mat_path in tqdm(mat_files, desc="Converting .mat -> .png"):
        image_id = int(mat_path.stem)
        record = read_mat(mat_path)
        image_rel, mask_rel = f"images/{image_id}.png", f"masks/{image_id}.png"

        image = Image.fromarray(to_uint8(record.image)).resize((size, size), Image.BILINEAR)
        mask = Image.fromarray(record.mask * 255).resize((size, size), Image.NEAREST)
        image.save(out_dir / image_rel)
        mask.save(out_dir / mask_rel)

        rows.append(
            {
                "image_id": image_id,
                "patient_id": record.patient_id,
                "label": record.label,
                "class_name": CLASS_NAMES[record.label],
                "cv_fold": int(folds[image_id - 1]),
                "image_path": image_rel,
                "mask_path": mask_rel,
                "orig_height": record.image.shape[0],
                "orig_width": record.image.shape[1],
                "tumor_area_frac": float(record.mask.mean()),
            }
        )

    metadata = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    metadata.to_csv(out_dir / METADATA_FILENAME, index=False)
    return metadata


def load_metadata(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")
    return pd.read_csv(path, dtype={"patient_id": str})
