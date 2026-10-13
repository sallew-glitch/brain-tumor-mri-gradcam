"""Step 3: Grad-CAM heatmaps + localization scores against the tumor masks.

Usage:
    # Comparing models/layers: score on the VALIDATION fold, at both layers
    python scripts/explain.py --run-dir <run> --split val --layers last penultimate

    # Final report for the chosen model and layer
    python scripts/explain.py --run-dir <run> --split test --layers penultimate

Layers: "last" (7x7 grid), "penultimate" (14x14 grid), or a module name such as layer3.
"""

import argparse
from pathlib import Path

from btmri.explain.runner import SPLITS, explain_fold
from btmri.utils.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", help="default: every trained fold in the run")
    parser.add_argument("--split", choices=SPLITS, default="test",
                        help="use val to compare models/layers, test for the final report")
    parser.add_argument("--layers", nargs="+", default=["last"],
                        help="one or more of: last, penultimate, <module name>")
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
        for layer in args.layers:
            explain_fold(run_dir, fold, args.num_examples, args.device, args.processed_dir,
                         args.target, args.split, layer)
    print(f"\nDone. See {run_dir}/fold_*/gradcam_{args.split}_*/")


if __name__ == "__main__":
    main()
