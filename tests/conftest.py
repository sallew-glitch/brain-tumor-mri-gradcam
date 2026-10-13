"""Shared test fixtures: a tiny synthetic dataset, so tests run in seconds without real data."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from btmri.constants import CLASS_NAMES, METADATA_FILENAME


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    """30 patients x 2 slices, 3 classes, 5 folds assigned by patient."""
    rows = []
    for p in range(30):
        label = p % 3
        for s in range(2):
            image_id = p * 2 + s + 1
            rows.append({
                "image_id": image_id,
                "patient_id": f"P{p:03d}",
                "label": label,
                "class_name": CLASS_NAMES[label],
                "cv_fold": p % 5 + 1,
                "image_path": f"images/{image_id}.png",
                "mask_path": f"masks/{image_id}.png",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def processed_dir(tmp_path, synthetic_metadata):
    """Writes the synthetic dataset to disk in the same layout as scripts/prepare_data.py."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rng = np.random.default_rng(0)
    for row in synthetic_metadata.itertuples():
        Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(tmp_path / row.image_path)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        Image.fromarray(mask).save(tmp_path / row.mask_path)
    synthetic_metadata.to_csv(tmp_path / METADATA_FILENAME, index=False)
    return tmp_path
