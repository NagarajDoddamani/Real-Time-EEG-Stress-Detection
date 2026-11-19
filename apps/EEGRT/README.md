# EEGRT desktop application

EEGRT is a standalone Electron application for LSL stream discovery, local EEG
recording, model inference, and downloadable session reports. It does not use a
database. Session metadata and predictions are kept in local JSON/CSV files.

## Run

From this directory, install the Python dependencies into the supported Python
environment, then launch Electron:

```powershell
python -m pip install -r requirements.txt
npm start
```

Start the EEGStudio LSL Outlet first, then use **Refresh streams**. Select the
stream explicitly, enter participant details, and start the live session. Stop
the session to finalize its local archive and enable the PDF report.

## Local files

Sessions are written under `data/live/combined-app/`:

- `recordings/`: original received channel values and LSL timestamps.
- `metadata/`: stream, participant, validation, model, and summary metadata.
- `predictions/`: one row per model window.
- `events/`: optional separately selected event stream.
- `logs/`: acquisition diagnostics.

The Session Reports view reads these local files. It does not depend on a
network connection or database.

## Signal and model safeguards

The application preserves incoming samples and timestamps without filtering,
normalization, or resampling. It requires stream channel labels and a compatible
observed rate before model inference. If labels are absent, generic `Ch 01`-style
columns are used only to preserve and display the recording; inference remains
blocked because the model channel mapping cannot be verified. Signal units and
whether EEGStudio's configured filters affect the LSL signal remain unverified.
Predictions estimate experimental conditions and are not clinical diagnoses.

LSL discovery confirms only that a stream is visible. A physical acquisition
test is still required to verify received channel metadata, timestamps, values,
and end-to-end predictions with the Mitsar device.
