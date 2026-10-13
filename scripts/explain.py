"""Step 3: Grad-CAM heatmaps + localization scores against the tumor masks.

Usage:
    python scripts/explain.py --run-dir outputs/runs/resnet18_cpu_20261009-120000
    python scripts/explain.py --run-dir <run> --folds 1 --num-examples 24 --target true
"""

import argparse
from pathlib import Path

from btmri.explain.runner import explain_fold
from btmri.utils.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", help="default: every trained fold in the run")
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted",
                        help="explain the class the model predicted, or the true class")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--processed-dir", help="override data location (e.g. model trained on Colab)")
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run_dir)
    folds = args.folds or sorted(
        int(p.name.split("_")[1]) for p in run_dir.glob("fold_*") if (p / "best.pt").exists()
    )
    for fold in folds:
        explain_fold(run_dir, fold, args.num_examples, args.device, args.processed_dir, args.target)
    print(f"\nDone. See {run_dir}/fold_*/gradcam/")


if __name__ == "__main__":
    main()
