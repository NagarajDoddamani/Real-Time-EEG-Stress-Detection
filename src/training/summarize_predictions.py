#!/usr/bin/env python3
"""Create readable participant/condition summaries from LOSO OOF predictions."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_rows(rows: list[dict[str, str]], model_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model") == model_name:
            grouped[(row["participant_id"], row["condition"], str(row["true_label"]))].append(row)
    output: list[dict[str, Any]] = []
    for (participant, condition, true_label), values in sorted(grouped.items(), key=lambda item: (item[0][0].casefold(), item[0][1])):
        label = int(true_label)
        predicted = [int(item["predicted_label"]) for item in values]
        baseline_count = predicted.count(0)
        stress_count = predicted.count(1)
        if baseline_count == stress_count:
            majority = "tie"
        else:
            majority = "baseline" if baseline_count > stress_count else "stress"
        probabilities = [float(item["probability_stress"]) for item in values]
        correct = sum(value == label for value in predicted)
        output.append({
            "participant_id": participant,
            "actual_condition": condition,
            "actual_label": label,
            "window_count": len(values),
            "correct_window_count": correct,
            "window_accuracy": correct / len(values),
            "predicted_baseline_windows": baseline_count,
            "predicted_stress_windows": stress_count,
            "majority_predicted_condition": majority,
            "mean_probability_stress": statistics.fmean(probabilities),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["participant_id", "actual_condition", "actual_label", "window_count",
              "correct_window_count", "window_accuracy", "predicted_baseline_windows",
              "predicted_stress_windows", "majority_predicted_condition", "mean_probability_stress"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create_summary(predictions_path: Path, comparison_path: Path,
                   marker_report_path: Path, output_dir: Path) -> dict[str, Any]:
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    marker_report = json.loads(marker_report_path.read_text(encoding="utf-8"))
    selected = comparison["selected_model"]
    rows = summarize_rows(predictions, selected)
    if not rows:
        raise ValueError(f"no out-of-fold predictions found for selected model {selected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "participant_condition_predictions.csv"
    write_csv(csv_path, rows)
    metrics = comparison["models"][selected]["pooled_out_of_fold"]
    blocked = [
        {"participant_id": item.get("participant_id", ""),
         "reason": "; ".join(item.get("exclusion_reasons", [])) or "marker validation failed"}
        for item in marker_report.get("participants", []) if not item.get("valid_for_windowing")
    ]
    summary = {
        "selected_model": selected,
        "prediction_method": "leave-one-participant-out out-of-fold predictions; each included participant was held out from its model fit",
        "participant_count": len({row["participant_id"] for row in rows}),
        "condition_summary_row_count": len(rows),
        "window_count": sum(row["window_count"] for row in rows),
        "overall_out_of_fold_metrics": metrics,
        "participant_condition_csv": str(csv_path),
        "participants_without_valid_condition_intervals": blocked,
        "limitations": [
            "condition markers identify experimental conditions, not clinical stress diagnoses",
            "LOSO model comparison is not external validation",
            "eight marker-invalid recordings are excluded from condition-labelled summary",
        ],
    }
    (output_dir / "participant_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Participant-level out-of-fold prediction summary", "",
        f"- Selected model: **{selected}**",
        "- Evaluation: **leave-one-participant-out**; each listed participant was held out during prediction.",
        f"- Participants included: **{summary['participant_count']}**",
        f"- Condition-summary rows: **{len(rows)}**",
        f"- Windows summarized: **{summary['window_count']}**", "",
        "These are experimental baseline/stress condition classifications, not clinical diagnoses. The metric below is pooled over out-of-fold windows; it is not external validation.", "",
        "## Pooled out-of-fold metrics", "",
        f"- Accuracy: **{metrics['accuracy']:.3f}**",
        f"- Macro precision: **{metrics['precision_macro']:.3f}**",
        f"- Macro recall: **{metrics['recall_macro']:.3f}**",
        f"- Macro F1: **{metrics['f1_macro']:.3f}**",
        f"- ROC-AUC: **{metrics['roc_auc']:.3f}**",
        f"- Confusion matrix (actual rows / predicted columns; baseline, stress): `{metrics['confusion_matrix']}`", "",
        "## Predictions by participant and recorded condition", "",
        "`window_accuracy` is the fraction of windows in that recorded condition predicted as its corresponding configured label.", "",
        "| Participant | Recorded condition | Windows | Correct | Accuracy | Predicted baseline | Predicted stress | Majority output | Mean stress probability |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['participant_id']} | {row['actual_condition']} | {row['window_count']} | {row['correct_window_count']} | {row['window_accuracy']:.3f} | {row['predicted_baseline_windows']} | {row['predicted_stress_windows']} | {row['majority_predicted_condition']} | {row['mean_probability_stress']:.3f} |")
    lines += ["", "## Excluded from condition-labelled prediction summary", "",
              "These recordings have missing or duplicate condition markers, so their windows cannot be assigned reliable baseline/stress ground truth.", ""]
    lines += [f"- **{item['participant_id']}**: {item['reason']}" for item in blocked] or ["- None"]
    lines.append("")
    (output_dir / "participant_prediction_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=root / "models" / "evaluation" / "out_of_fold_predictions.csv")
    parser.add_argument("--comparison", type=Path, default=root / "models" / "evaluation" / "model_comparison.json")
    parser.add_argument("--marker-report", type=Path, default=root / "data" / "processed" / "marker_validation.json")
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "evaluation")
    args = parser.parse_args()
    try:
        summary = create_summary(args.predictions, args.comparison, args.marker_report, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Summary generation failed: {exc}\n")
    print(f"Wrote participant/condition predictions for {summary['participant_count']} participants "
          f"({summary['window_count']} out-of-fold windows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
