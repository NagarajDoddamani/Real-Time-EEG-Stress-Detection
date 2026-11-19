# NeuroMonitor — combined EEG desktop application

The primary Electron app now lives in `apps/EEGRT/`. This earlier copy is
retained to preserve project history; make future changes in `EEGRT/`.

This Electron app replaces the earlier two-computer relay workflow. Run it on
the computer that can see EEGStudio's LSL Outlet. It discovers the outlet,
collects participant details, receives the stream directly, applies the
existing trained SVM pipeline, displays the incoming traces and estimates, and
saves session/results metadata to Neo4j Aura.

Raw EEG samples are recorded incrementally to local CSV files. The app does
not send raw signal chunks to Aura. Neo4j stores participant identity, session
metadata, prediction windows, and final session summary. The Electron app can
generate a PDF report from a completed session or retrieve saved results for a
previous session.

## Setup

On the EEG acquisition computer:

1. Start the SmartBCI x24 and EEGStudio.
2. Start EEGStudio's LSL Outlet.
3. Configure the project Python environment and `.env` with the Aura
   connection settings. The `.env` file must stay local and untracked.
4. Install the Python dependencies and Electron dependencies if needed.

```powershell
cd C:\path\to\Real-Time-EEG-Stress-Detection
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # only if this machine does not already have .env
notepad .env
cd apps\device2-dashboard
npm install                  # only for initial setup
$env:EEG_PYTHON = (Resolve-Path ..\..\.venv\Scripts\python.exe).Path
npm start
```

If using the system Python instead of `.venv`, omit the `EEG_PYTHON` assignment.
The Python executable used by Electron must have the same installed dependencies
and must be able to read the project `.env`.

## Session workflow

1. Refresh streams and explicitly select the EEGStudio EEG outlet.
2. Enter a participant ID and full name.
3. Start the live session. The app validates connected channel labels and the
   stream sampling rate against the trained model before predicting.
4. Stop the session to flush local recording files and save the Aura summary.
5. Download the report, or retrieve a saved report from the session archive.

The current model requires the exact 19 trained EEG channels and 500 Hz. If
EEGStudio publishes blank labels or does not expose a verified mapping, the app
will block inference rather than guess channels. Verify the inlet metadata and
electrode ordering before relying on predictions. Signal units and EEGStudio
filtering/raw-vs-processed behavior remain unverified.

The predictions estimate recorded baseline/stress experimental conditions.
They are preliminary and are not clinical diagnoses. Music control is not
connected.
