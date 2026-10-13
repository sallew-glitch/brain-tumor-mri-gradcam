from pathlib import Path

import pytest

from btmri.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_without_file():
    cfg = load_config()
    assert cfg.model.name == "resnet18"
    assert cfg.folds == [1, 2, 3, 4, 5]


def test_base_inheritance_and_overrides():
    cfg = load_config(CONFIG_DIR / "efficientnet_b0.yaml", ["train.epochs=3", "folds=[2]"])
    assert cfg.model.name == "efficientnet_b0"  # from the child file
    assert cfg.data.image_size == 224  # inherited from base.yaml
    assert cfg.train.epochs == 3 and cfg.folds == [2]  # CLI overrides
    assert isinstance(cfg.train.lr, float)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError, match="lrr"):
        load_config(overrides=["train.lrr=0.1"])


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    load_config(path)
