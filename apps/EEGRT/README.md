# EEGRT — standalone real-time EEG application

This folder is the complete application. It does not import Python code,
models, settings, or UI assets from the repository's old `src/`, `models/`,
`configs/`, `apps/device1-recorder/`, or `apps/device2-dashboard/` folders.

## Run

From PowerShell:

```powershell
cd C:\path\to\Real-Time-EEG-Stress-Detection\apps\EEGRT
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # only if this folder does not already have .env
notepad .env
npm install                   # only for initial setup
npm start
```

The `.env` file contains local Aura credentials and is ignored by Git. The
Python environment and Electron app are resolved from this folder. Local EEG
recordings and logs are written under `data/` within this folder.

## Single-folder contents

- `dev1_discover.ps1`, `dev1_record.ps1`: standalone LSL discovery and raw
  recording utilities for the acquisition side.
- `main.js`, `preload.js`, `index.html`, `renderer.js`, `styles.css`: the
  combined Electron UI and its secure IPC bridge.
- `eeg_cli.py`, `eeg_pipeline.py`, `eeg_recording.py`, `eeg_features.py`:
  acquisition, rolling-window validation, inference, local recording, and the
  exact feature implementation used by the saved model.
- `eeg_service.py`, `eeg_database.py`: participant/session/result storage and
  report retrieval through Neo4j Aura. Raw EEG samples are not sent to Aura.
- `stress_model.pkl`, `model_metadata.json`, `feature_names.json`,
  `eeg_config.json`: the bundled model and its inference contract.
- `requirements.txt`, `package.json`, `package-lock.json`: standalone runtime
  dependencies.

## Live session

1. Start the Mitsar device and EEGStudio, then enable EEGStudio's LSL Outlet.
2. Start this app with `npm start`.
3. Discover and select the stream; enter participant ID and full name.
4. Start the session. Incoming samples are recorded locally; valid windows are
   passed through the bundled feature/model pipeline and shown in the UI.
5. Stop the session to save summary and prediction results in Aura. Use the
   session archive to review results and download a PDF.

The model requires the exact 19 channel labels and 500 Hz. It refuses to infer
when channel labels or observed rate cannot be validated. Your screenshot's
23-channel stream had blank labels, so the graph may show generic channel names
and predictions will remain blocked until EEGStudio exposes a reliable channel
mapping. Signal units and EEGStudio raw/processed/filter behavior remain
unverified. Outputs are experimental-condition estimates, not clinical
diagnoses.

## Device 1 command-line utilities

```powershell
.\dev1_discover.ps1
.\dev1_record.ps1 -StreamIndex 0
```

The main workflow is the combined Electron UI. `dev1_record.ps1` is a separate
diagnostic recorder and does not run model inference.
