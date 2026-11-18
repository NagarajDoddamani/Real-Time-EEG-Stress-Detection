#!/usr/bin/env python3
"""Compare EEG condition classifiers with leave-one-participant-out CV."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


METADATA_COLUMNS = {
    "participant_id", "condition", "label", "source_filename", "start_sample_index",
    "end_sample_index_exclusive", "normalized_filename",
}
LABELS = [0, 1]


def validate_training_inputs(config: dict[str, Any], marker_report: dict[str, Any],
                             window_rows: list[dict[str, str]], feature_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    fs = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or fs is None or float(fs) != 500.0:
        errors.append("sampling frequency is not verified as the required 500 Hz dataset setting")
    participants = {p["participant_id"] for p in marker_report.get("participants", [])
                    if p.get("valid_for_windowing")}
    if not participants:
        errors.append("marker report contains no valid participants")
    if not window_rows:
        errors.append("window index is empty")
    if not feature_rows:
        errors.append("feature table is empty")
    if window_rows and feature_rows:
        required = METADATA_COLUMNS - {"normalized_filename"}
        if not required.issubset(window_rows[0]) or not required.issubset(feature_rows[0]):
            errors.append("window/feature files are missing participant, label, condition, or sample-index metadata")
        else:
            def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
                return (row["participant_id"], row["condition"], str(row["label"]),
                        str(row["start_sample_index"]), str(row["end_sample_index_exclusive"]))
            if len({key(r) for r in window_rows}) != len(window_rows):
                errors.append("window index has duplicate window keys")
            if len({key(r) for r in feature_rows}) != len(feature_rows):
                errors.append("feature file has duplicate window keys")
            if {key(r) for r in window_rows} != {key(r) for r in feature_rows}:
                errors.append("feature rows do not correspond exactly to window index rows")
            label_map = config.get("label_map", {})
            if any(str(row["label"]) != str(label_map.get(row["condition"], "invalid"))
                   for row in feature_rows):
                errors.append("condition/label pairs disagree with configured label_map")
            if {r["participant_id"] for r in feature_rows} - participants:
                errors.append("feature table contains participant(s) without validated marker intervals")
            if any(int(r["end_sample_index_exclusive"]) - int(r["start_sample_index"]) != 2000
                   for r in window_rows):
                errors.append("window index contains non-2000-sample windows")
            expected = config.get("label_map", {})
            if any(str(r["label"]) != str(expected.get(r["condition"], "invalid")) for r in feature_rows):
                errors.append("condition labels disagree with configured label_map")
    return errors


def model_factories() -> dict[str, Any]:
    factories: dict[str, Any] = {
        "logistic_regression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=1,
        ),
        "svm_rbf": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42)),
        ]),
    }
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier
        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    return factories


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(metrics.roc_auc_score(y_true, y_score))
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision_macro": float(metrics.precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(metrics.recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(metrics.f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": metrics.classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0,
        ),
    }


def _positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=float)
    decision = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -50, 50)))


def evaluate_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  feature_rows: list[dict[str, str]], factories: dict[str, Any]
                  ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    participant_ids = sorted(set(groups.tolist()), key=str.casefold)
    if len(participant_ids) < 3:
        raise ValueError("LOSO evaluation requires at least three participants")
    for participant in participant_ids:
        participant_y = set(y[groups == participant].tolist())
        if participant_y != set(LABELS):
            raise ValueError(f"{participant} does not contain both classes; cannot compute balanced per-participant evaluation")

    comparison: dict[str, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name, factory in factories.items():
        oof_y = np.full(len(y), -1, dtype=int)
        oof_score = np.full(len(y), np.nan, dtype=float)
        participant_metrics: dict[str, Any] = {}
        for fold_index, held_out in enumerate(participant_ids, start=1):
            test_mask = groups == held_out
            train_mask = ~test_mask
            train_participants = set(groups[train_mask].tolist())
            if held_out in train_participants or not train_participants:
                raise RuntimeError("participant leakage detected in LOSO split")
            if set(y[train_mask].tolist()) != set(LABELS):
                raise ValueError(f"training fold {held_out} does not contain both classes")
            model = factory()
            # Every scaler/model is fitted on this fold's training participants only.
            model.fit(X[train_mask], y[train_mask])
            y_pred = np.asarray(model.predict(X[test_mask]), dtype=int)
            y_score = _positive_scores(model, X[test_mask])
            oof_y[test_mask] = y_pred
            oof_score[test_mask] = y_score
            fold_metrics = metric_bundle(y[test_mask], y_pred, y_score)
            participant_metrics[held_out] = fold_metrics
            fold_rows.append({"model": model_name, "fold": fold_index, "held_out_participant": held_out,
                              "window_count": int(np.sum(test_mask)), **{k: v for k, v in fold_metrics.items()
                              if k != "confusion_matrix" and k != "classification_report"},
                              "confusion_matrix": json.dumps(fold_metrics["confusion_matrix"])})

        if np.any(oof_y < 0) or not np.isfinite(oof_score).all():
            raise RuntimeError(f"{model_name}: failed to produce one out-of-fold result per window")
        pooled = metric_bundle(y, oof_y, oof_score)
        mean_participant_f1 = float(np.mean([m["f1_macro"] for m in participant_metrics.values()]))
        comparison[model_name] = {
            "pooled_out_of_fold": pooled,
            "mean_participant_macro_f1": mean_participant_f1,
            "mean_participant_accuracy": float(np.mean([m["accuracy"] for m in participant_metrics.values()])),
            "per_participant": participant_metrics,
            "fold_count": len(participant_ids),
        }
        for index, row in enumerate(feature_rows):
            prediction_rows.append({
                "model": model_name, "participant_id": row["participant_id"],
                "condition": row["condition"], "start_sample_index": row["start_sample_index"],
                "end_sample_index_exclusive": row["end_sample_index_exclusive"],
                "true_label": int(y[index]), "predicted_label": int(oof_y[index]),
                "probability_stress": float(oof_score[index]),
            })
    return comparison, fold_rows, prediction_rows


def train(features_path: Path, config_path: Path, marker_report_path: Path,
          window_index_path: Path, feature_manifest_path: Path,
          output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_frequency_status") != "verified_by_user_for_existing_recordings":
        raise ValueError("feature manifest does not mark the user-confirmed existing recordings as 500 Hz")
    if float(manifest.get("sampling_frequency_hz", -1)) != float(config.get("sampling_frequency_hz", -2)):
        raise ValueError("feature manifest and training configuration sampling rates differ")
    marker_report = json.loads(marker_report_path.read_text(encoding="utf-8"))
    with window_index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        window_rows = list(csv.DictReader(stream))
    with features_path.open("r", encoding="utf-8-sig", newline="") as stream:
        feature_rows = list(csv.DictReader(stream))
    errors = validate_training_inputs(config, marker_report, window_rows, feature_rows)
    if errors:
        raise ValueError("training input validation failed: " + "; ".join(errors))

    feature_names = manifest.get("feature_names", [])
    if not feature_names or any(name not in feature_rows[0] for name in feature_names):
        raise ValueError("feature manifest schema does not match training feature CSV")
    X = np.asarray([[float(row[name]) for name in feature_names] for row in feature_rows], dtype=np.float64)
    y = np.asarray([int(row["label"]) for row in feature_rows], dtype=int)
    groups = np.asarray([row["participant_id"] for row in feature_rows], dtype=object)
    if not np.isfinite(X).all() or set(y.tolist()) != set(LABELS):
        raise ValueError("training matrix contains invalid values or is missing one of the two labels")

    factories = model_factories()
    comparison, fold_rows, prediction_rows = evaluate_loso(X, y, groups, feature_rows, factories)
    selected_name = max(comparison, key=lambda name: (
        comparison[name]["mean_participant_macro_f1"],
        comparison[name]["pooled_out_of_fold"]["f1_macro"],
        comparison[name]["pooled_out_of_fold"]["balanced_accuracy"],
    ))
    final_model = factories[selected_name]()
    final_model.fit(X, y)

    evaluation_dir = output_dir / "evaluation"
    expected_paths = [output_dir / "stress_model.pkl", output_dir / "feature_names.json",
                      output_dir / "model_metadata.json", evaluation_dir / "model_comparison.json",
                      evaluation_dir / "cross_validation_results.csv", evaluation_dir / "out_of_fold_predictions.csv",
                      evaluation_dir / "confusion_matrix.png", evaluation_dir / "classification_report.json"]
    present = [str(path) for path in expected_paths if path.exists()]
    if present and not overwrite:
        raise FileExistsError("model outputs already exist; use --overwrite to replace: " + ", ".join(present))
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_path = output_dir / "stress_model.pkl"
    joblib.dump(final_model, model_path)
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (evaluation_dir / "model_comparison.json").write_text(json.dumps({
        "selection_metric": "mean participant-level macro F1 from LOSO out-of-fold predictions",
        "selected_model": selected_name,
        "models": comparison,
        "xgboost_available": "xgboost" in factories,
        "participants": sorted(set(groups.tolist()), key=str.casefold),
        "window_count": int(len(y)),
        "class_window_counts": {str(label): int(np.sum(y == label)) for label in LABELS},
    }, indent=2, allow_nan=False), encoding="utf-8")
    _write_rows(evaluation_dir / "cross_validation_results.csv", fold_rows)
    _write_rows(evaluation_dir / "out_of_fold_predictions.csv", prediction_rows)

    selected_oof = comparison[selected_name]["pooled_out_of_fold"]
    (evaluation_dir / "classification_report.json").write_text(json.dumps({
        "selected_model": selected_name,
        "pooled_out_of_fold": selected_oof["classification_report"],
        "per_participant": {pid: result["classification_report"]
                            for pid, result in comparison[selected_name]["per_participant"].items()},
    }, indent=2, allow_nan=False), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.asarray(selected_oof["confusion_matrix"]), display_labels=["baseline", "stress"]).plot(
        ax=axis, values_format="d", colorbar=False,
    )
    axis.set_title(f"{selected_name}: pooled LOSO out-of-fold predictions")
    figure.tight_layout()
    figure.savefig(evaluation_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    try:
        package_versions = {name: importlib.metadata.version(name) for name in
                            ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib")}
    except importlib.metadata.PackageNotFoundError:
        package_versions = {}
    participant_ids = sorted(set(groups.tolist()), key=str.casefold)
    metadata = {
        "model_algorithm": selected_name,
        "candidate_algorithms": list(factories),
        "sampling_frequency_hz": 500,
        "sampling_frequency_status": "user_confirmed_for_existing_21_recordings; live stream still unverified",
        "window_duration_seconds": 4.0,
        "window_samples": 2000,
        "window_step_samples": 1000,
        "channel_names_and_order": manifest["channel_order"],
        "feature_names_and_order": feature_names,
        "feature_definitions": manifest["feature_definitions"],
        "frequency_bands_hz": manifest["frequency_bands_hz"],
        "psd_configuration": manifest["psd_configuration"],
        "signal_units": None,
        "signal_units_status": "unverified; absolute features are uncalibrated numeric scale",
        "filters_applied": [],
        "filter_note": "No new filters applied; export-time filtering of source CSVs remains unverified.",
        "participant_count": len(participant_ids),
        "training_participant_ids": participant_ids,
        "validation_method": "Leave-One-Participant-Out cross-validation; same folds used for model comparison",
        "selection_metric": "mean participant-level macro F1",
        "selected_model_cv_metrics": selected_oof,
        "all_model_cv_metrics": comparison,
        "class_window_counts": {str(label): int(np.sum(y == label)) for label in LABELS},
        "python_version": platform.python_version(),
        "library_versions": package_versions,
        "model_created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "external_validation": False,
        "label_interpretation": "experimental baseline/stress conditions; not a clinical diagnosis",
        "deployment_ready": False,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty evaluation table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=root / "data" / "features" / "training_500Hz_19ch" / "training_features.csv")
    parser.add_argument("--window-index", type=Path, default=root / "data" / "features" / "training_500Hz_19ch" / "window_index.csv")
    parser.add_argument("--feature-manifest", type=Path, default=root / "data" / "features" / "training_500Hz_19ch" / "feature_manifest.json")
    parser.add_argument("--config", type=Path, default=root / "configs" / "eeg_config.json")
    parser.add_argument("--marker-report", type=Path, default=root / "data" / "processed" / "marker_validation.json")
    parser.add_argument("--output-dir", type=Path, default=root / "models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = train(args.features, args.config, args.marker_report, args.window_index,
                         args.feature_manifest, args.output_dir, args.overwrite)
    except (OSError, ValueError, KeyError, ImportError, TypeError) as exc:
        parser.exit(2, f"Training stopped: {exc}\n")
    print(f"Selected {metadata['model_algorithm']} by mean participant macro F1; "
          f"LOSO participants={metadata['participant_count']}; model={metadata['model_path']}")
    print("XGBoost was not included because the xgboost package is not installed." if "xgboost" not in metadata["candidate_algorithms"] else "XGBoost included in LOSO comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
