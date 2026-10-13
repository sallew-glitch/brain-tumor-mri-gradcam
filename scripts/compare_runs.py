"""Compare finished experiments side by side (one row per run, mean over folds).

Pick settings by the VALIDATION column. The test columns are for reporting the final
result; choosing settings by test score lets the test set leak into your decisions.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --runs-dir outputs/runs --csv comparison.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

TEST_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")


def best_validation_score(run_dir: Path, folds: list[int], monitor: str) -> float:
    """Mean over folds of the best validation score, i.e. the score of the saved checkpoint."""
    scores = []
    for fold in folds:
        history = pd.read_csv(run_dir / f"fold_{fold}" / "history.csv")
        column = history[f"val_{monitor}"]
        scores.append(column.min() if monitor == "loss" else column.max())
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--csv", help="also save the table to this file")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(Path(args.runs_dir).glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        monitor = cfg["train"]["monitor"]
        row = {
            "run": run_dir.name,
            "model": cfg["model"]["name"],
            "folds": len(summary["folds"]),
            f"VAL_{monitor}": f"{best_validation_score(run_dir, summary['folds'], monitor):.4f}",
        }
        for metric in TEST_METRICS:
            row[f"test_{metric}"] = f"{summary['mean'][metric]:.4f} +/- {summary['std'][metric]:.4f}"
        rows.append(row)

    if not rows:
        print(f"No finished runs found in {args.runs_dir}")
        return
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
