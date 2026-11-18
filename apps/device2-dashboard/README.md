# Device 2 — EEG session analysis desktop app

This Electron desktop UI lists active and completed EEG sessions with stored
sample chunks in Aura. Select a participant/session and start processing; the
Python service polls for the next one-second (500-sample) chunk, beginning at
chunk zero so data stored before Device 2 starts is not skipped. It validates
channel mapping, sample count, finite values, and timestamps before extracting
training-compatible features and running the saved SVM. Model windows are four
seconds with a two-second step, so a new prediction normally appears every two
seconds after the initial four-second warm-up.

The UI shows participant details read from the session, stacked EEG traces,
window-level model stress-score trend, prediction history, and a stop-and-
summarize action. Stopping Device 2 does not stop Device 1 recording. A PDF
report can be saved after processing stops. Music automation is intentionally
not implemented; it remains a separately validated later module.

## Setup

Install Node.js and Electron on Device 2, then copy the repository and its
trained artifacts to that computer. The project Python environment must have
the packages in root `requirements.txt`; Device 2 also needs a local `.env`
with its own Aura connection configuration. Never copy credentials through
source control or package them in the Electron app.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
cd apps/device2-dashboard
npm install
npm start
```

The desktop app launches Python as a child process from the repository root.
Set `EEG_PYTHON` to the desired Python executable if `python` is not on PATH.

## Data and interpretation

The app uses `src/dashboard/device2_service.py`, `Neo4jReader`,
`RollingWindowBuffer`, and `TrainedModel`; feature definitions are not copied
into the UI. Every displayed class is an estimate of the recorded experimental
baseline/stress condition, not a clinical stress diagnosis. The current model
has preliminary participant-wise validation only and no external validation.
Signals are accepted only when the stored channel set/order can be mapped and
the saved sample rate is compatible with the trained 500 Hz configuration.

The existing bounded prototype stores up to the requested number of raw
seconds in Neo4j. Device 2 refuses a session when the stored sample count and
retrieved chunks disagree. It does not fill missing samples or resample.
For a 10-20 minute run, configure Device 1's `--db-raw-seconds` to 600-1200
seconds. The stress-score graph shows the SVM's uncalibrated model output; it
must not be interpreted as a clinical probability or diagnosis.
