# Offline EEG condition-classifier training

The training workflow uses the prepared normalized recordings under `data/processed/eligible`, the current manifest, and the marker validation report. It does not edit files under `data/raw`.

## Rebuild features

```powershell
python -m src.features.build_feature_dataset
```

The default output is `data/features/training_500Hz_19ch/`. It contains all 19 configured scalp channels, 4-second windows (2,000 samples), a 1,000-sample step, event-derived labels, 5 time-domain features per channel, configured Welch absolute/relative band powers, and normalized spectral entropy. Invalid marker participants are listed as skipped in `window_coverage.csv`.

The configured bands are delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz and gamma 30–80 Hz. Welch uses a 2-second Hann segment, 50% segment overlap, constant detrending, density scaling and mean averaging. No additional filters or unit conversions are applied.

## Train and compare

```powershell
python -m src.training.train_models
```

This compares Logistic Regression, Random Forest and RBF-SVM using leave-one-participant-out folds. Scalers are fitted inside each training fold. Model selection uses mean participant-level macro F1. The selected pipeline is then fitted on all marker-valid participants and saved under `models/`.

XGBoost is included only if `xgboost` is already installed. Install core packages from `requirements.txt` if needed. No package is installed automatically by these scripts.

## Replay a recorded segment at the configured rate

The recorded replay feeds source samples in order as small chunks and paces them using the configured rate. For example, to replay 12 seconds starting at the validated baseline marker for `ananya_d`:

```powershell
python -m src.realtime.recorded_replay `
  --input data/processed/eligible/ananya_d.csv `
  --start-sample 7780 `
  --duration-seconds 12 `
  --chunk-samples 64 `
  --replay-speed 1 `
  --model models/stress_model.pkl `
  --training-metadata models/model_metadata.json `
  --allow-exploratory-model `
  --output data/processed/replay_test_500Hz/ananya_d_baseline_12s.csv
```

Use the source sample index for the desired interval. Replay verifies the model feature/channel order and writes a CSV plus JSON run metadata. This exercises local buffering, feature extraction and model inference; it does not publish or receive an LSL stream. The current model remains marked not deployment-ready while signal units/export scaling are unverified.

Labels represent the recorded baseline and stress experimental conditions, not clinical diagnoses. The 500 Hz rate is recorded as user-confirmed for the existing recordings; the live LSL rate still needs independent verification. Physical units and export scaling remain unknown, so absolute amplitude and power values are not calibrated physical measurements. EEGStudio's displayed filter settings are not assumed to have been applied to the exported raw data.
