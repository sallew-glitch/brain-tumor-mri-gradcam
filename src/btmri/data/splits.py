"""Patient-disjoint train / validation / test splits.

Why patient-disjoint? Each patient contributes several neighbouring MRI slices that look
almost identical. If slices of the same patient land in both train and test, the model can
"recognise the patient" instead of the tumor and the test score is inflated. This is the
most common flaw in published results on this dataset, so every split here guarantees that
a patient appears in exactly one of train / val / test.

Scheme for test fold k (folds numbered 1..n):  test = k, val = k+1 (wrapping), train = rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class FoldSplit:
    test_fold: int
    val_fold: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_patient_folds(metadata: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Assign each row a fold 1..n so that patients never span folds and classes stay balanced."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = pd.Series(0, index=metadata.index, dtype=int)
    for fold, (_, test_idx) in enumerate(
        splitter.split(metadata, metadata["label"], groups=metadata["patient_id"]), start=1
    ):
        folds.iloc[test_idx] = fold
    return folds


def assert_patient_disjoint(*parts: pd.DataFrame) -> None:
    seen: set[str] = set()
    for part in parts:
        patients = set(part["patient_id"])
        overlap = seen & patients
        if overlap:
            raise AssertionError(f"Patients appear in more than one split: {sorted(overlap)[:5]}...")
        seen |= patients


def get_fold_split(
    metadata: pd.DataFrame,
    test_fold: int,
    n_folds: int = 5,
    fold_column: str = "cv_fold",
) -> FoldSplit:
    if not 1 <= test_fold <= n_folds:
        raise ValueError(f"test_fold must be in 1..{n_folds}, got {test_fold}")
    val_fold = test_fold % n_folds + 1
    folds = metadata[fold_column]
    split = FoldSplit(
        test_fold=test_fold,
        val_fold=val_fold,
        train=metadata[~folds.isin([test_fold, val_fold])].reset_index(drop=True),
        val=metadata[folds == val_fold].reset_index(drop=True),
        test=metadata[folds == test_fold].reset_index(drop=True),
    )
    assert_patient_disjoint(split.train, split.val, split.test)
    return split
