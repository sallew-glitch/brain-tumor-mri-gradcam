import numpy as np
import pytest
import torch

from btmri.config import AugmentConfig
from btmri.data.convert import to_uint8
from btmri.data.dataset import BrainTumorDataset
from btmri.data.splits import assert_patient_disjoint, get_fold_split, make_patient_folds
from btmri.data.transforms import build_eval_transform, build_train_transform


@pytest.mark.parametrize("test_fold", [1, 2, 3, 4, 5])
def test_fold_split_is_patient_disjoint_and_complete(synthetic_metadata, test_fold):
    split = get_fold_split(synthetic_metadata, test_fold)
    assert split.val_fold == test_fold % 5 + 1
    assert len(split.train) + len(split.val) + len(split.test) == len(synthetic_metadata)
    assert_patient_disjoint(split.train, split.val, split.test)


def test_leak_is_detected(synthetic_metadata):
    with pytest.raises(AssertionError):
        assert_patient_disjoint(synthetic_metadata.iloc[:3], synthetic_metadata.iloc[1:4])


def test_generated_folds_keep_patients_together(synthetic_metadata):
    folds = make_patient_folds(synthetic_metadata, n_folds=5, seed=0)
    assert set(folds) == {1, 2, 3, 4, 5}
    assert (synthetic_metadata.assign(f=folds).groupby("patient_id")["f"].nunique() == 1).all()


def test_to_uint8_range():
    image = np.linspace(-100, 5000, 10_000).reshape(100, 100)
    out = to_uint8(image)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert to_uint8(np.zeros((4, 4))).max() == 0  # constant image does not divide by zero


def test_dataset_items(processed_dir, synthetic_metadata):
    ds = BrainTumorDataset(
        synthetic_metadata, processed_dir, build_eval_transform(32), return_mask=True, mask_size=32
    )
    item = ds[0]
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert item["mask"].shape == (32, 32) and set(item["mask"].unique().tolist()) <= {0.0, 1.0}
    assert item["label"] == synthetic_metadata.loc[0, "label"]


def test_train_transform_output_shape(processed_dir, synthetic_metadata):
    ds = BrainTumorDataset(synthetic_metadata, processed_dir, build_train_transform(48, AugmentConfig()))
    assert ds[0]["image"].shape == (3, 48, 48)
