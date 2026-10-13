"""Step 2: train and evaluate with patient-disjoint cross-validation.

Usage:
    python scripts/train.py --config configs/resnet18_cpu.yaml
    python scripts/train.py --config configs/resnet18_cpu.yaml --folds 1 --set train.epochs=3
"""

import argparse

from btmri.config import load_config
from btmri.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="YAML file under configs/")
    parser.add_argument("--folds", type=int, nargs="+", help="test folds to run (default: from config)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.lr=1e-4 model.name=resnet50")
    args = parser.parse_args()

    overrides = list(args.set)
    if args.folds:
        overrides.append(f"folds={args.folds}")
    cfg = load_config(args.config, overrides)
    run_dir = run_experiment(cfg)
    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
