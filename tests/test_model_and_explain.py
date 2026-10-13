import numpy as np
import pytest
import torch

from btmri.config import ModelConfig
from btmri.explain.gradcam import GradCAM
from btmri.explain.localization import localization_metrics
from btmri.models import build_model, set_backbone_trainable
from btmri.training.metrics import compute_metrics


@pytest.fixture(scope="module")
def tiny_model():
    return build_model(ModelConfig(name="resnet18", pretrained=False), num_classes=3)


def test_model_output_shape(tiny_model):
    assert tiny_model(torch.randn(2, 3, 64, 64)).shape == (2, 3)


def test_freezing_leaves_only_head_trainable(tiny_model):
    set_backbone_trainable(tiny_model, False)
    trainable = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("fc.") for n in trainable)
    set_backbone_trainable(tiny_model, True)
    assert all(p.requires_grad for p in tiny_model.parameters())


def test_gradcam_shape_and_range(tiny_model):
    images = torch.randn(2, 3, 64, 64)
    cams, logits = GradCAM(tiny_model)(images)
    assert cams.shape == (2, 64, 64) and logits.shape == (2, 3)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6


def test_gradcam_works_with_frozen_backbone(tiny_model):
    set_backbone_trainable(tiny_model, False)
    try:
        cams, _ = GradCAM(tiny_model)(torch.randn(1, 3, 64, 64), torch.tensor([2]))
        assert cams.shape == (1, 64, 64)
    finally:
        set_backbone_trainable(tiny_model, True)


def test_localization_perfect_and_miss():
    mask = np.zeros((10, 10))
    mask[2:5, 2:5] = 1
    inside = localization_metrics(mask.copy(), mask)
    assert inside["pointing_hit"] == 1 and inside["energy_in_mask"] == 1 and inside["iou"] == 1
    assert inside["energy_ratio_vs_chance"] == pytest.approx(1 / 0.09)

    elsewhere = np.zeros((10, 10))
    elsewhere[7:9, 7:9] = 1
    miss = localization_metrics(elsewhere, mask)
    assert miss["pointing_hit"] == 0 and miss["energy_in_mask"] == 0 and miss["iou"] == 0


def test_metrics_on_known_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_prob = np.eye(3)[[0, 1, 1, 1, 2, 2]]  # one meningioma mistaken for glioma
    m = compute_metrics(y_true, y_prob, ("meningioma", "glioma", "pituitary"))
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["per_class"]["meningioma"]["sensitivity"] == pytest.approx(0.5)
    assert m["per_class"]["glioma"]["specificity"] == pytest.approx(3 / 4)
    assert m["confusion_matrix"][0] == [1, 1, 0]
