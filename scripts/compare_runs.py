"""Compare finished experiments side by side (mean over folds).

Prints two tables:
  1. Classification: pick settings by the VAL column; test columns are for the final report.
  2. Grad-CAM localization, for every split/layer that explain.py has been run on.

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
CAM_METRICS = ("pointing_hit", "energy_ratio_vs_chance", "iou")


def best_validation_score(run_dir: Path, folds: list[int], monitor: str) -> float:
    """Mean over folds of the best validation score, i.e. the score of the saved checkpoint."""
    scores = []
    for fold in folds:
        history = pd.read_csv(run_dir / f"fold_{fold}" / "history.csv")
        column = history[f"val_{monitor}"]
        scores.append(column.min() if monitor == "loss" else column.max())
    return sum(scores) / len(scores)


def classification_row(run_dir: Path, summary: dict, cfg: dict) -> dict:
    monitor = cfg["train"]["monitor"]
    row = {
        "run": run_dir.name,
        "model": cfg["model"]["name"],
        "folds": len(summary["folds"]),
        f"VAL_{monitor}": f"{best_validation_score(run_dir, summary['folds'], monitor):.4f}",
    }
    for metric in TEST_METRICS:
        row[f"test_{metric}"] = f"{summary['mean'][metric]:.4f} +/- {summary['std'][metric]:.4f}"
    return row


def gradcam_rows(run_dir: Path, cfg: dict) -> list[dict]:
    """One row per (split, layer), averaged over the folds that have been explained."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for path in run_dir.glob("fold_*/gradcam_*/summary.json"):
        cam = json.loads(path.read_text())
        groups.setdefault((cam["split"], cam["layer"]), []).append(cam["overall"])
    rows = []
    for (split, layer), overalls in sorted(groups.items()):
        means = pd.DataFrame(overalls).mean()
        rows.append({
            "run": run_dir.name, "model": cfg["model"]["name"], "split": split, "layer": layer,
            "folds": len(overalls), **{m: round(float(means[m]), 3) for m in CAM_METRICS},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--csv", help="also save the classification table to this file")
    args = parser.parse_args()

    rows, cam_rows = [], []
    for summary_path in sorted(Path(args.runs_dir).glob("*/summary.json")):
        run_dir = summary_path.parent
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        rows.append(classification_row(run_dir, json.loads(summary_path.read_text()), cfg))
        cam_rows.extend(gradcam_rows(run_dir, cfg))

    if not rows:
        print(f"No finished runs found in {args.runs_dir}")
        return
    table = pd.DataFrame(rows)
    print("CLASSIFICATION\n" + table.to_string(index=False))
    if cam_rows:
        print("\nGRAD-CAM LOCALIZATION (higher is better)\n" + pd.DataFrame(cam_rows).to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
