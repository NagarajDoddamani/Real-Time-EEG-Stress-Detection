# LSL streaming, model inference, and Neo4j storage

This module uses the existing saved `models/stress_model.pkl` and its metadata. It does not train or alter a model. The current artifact is the `svm_rbf` pipeline with scaling embedded, trained on 13 participants and evaluated by leave-one-participant-out cross-validation (1,103 held-out windows; accuracy 0.534, macro F1 0.532). This is preliminary prediction of experimental baseline/stress conditions, not a clinical diagnosis or external validation.

## Data storage design

- Original LSL samples and source timestamps are incrementally written to `data/live/recordings/` CSV in stream channel order.
- Metadata and validation results go to `data/live/metadata/` JSON.
- Window predictions and supported probabilities go to `data/live/predictions/` CSV.
- When Neo4j is enabled, it stores session/stream metadata, links to local files, and one `EEGWindowPrediction` node per prediction. High-rate raw samples remain in CSV; they are not expanded into graph nodes.
- Graph labels are `EEGSession`, `EEGStream`, and `EEGWindowPrediction`; relations are `RECEIVED_FROM` and `HAS_PREDICTION`. IDs use session UUID and session/window index so writes are repeatable.

## Windows setup

From the project root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pylsl` also needs the Lab Streaming Layer native library on some Windows setups. Follow the official pylsl installation notes for the current package/platform combination if import reports a missing native library. Do not install a second runtime DLL from an untrusted source.

Neo4j is optional for recording and inference. A local `.env` file is git-ignored and loaded by the Neo4j adapter. Copy `.env.example` to `.env`, then fill in the Aura connection URI, username, password, and database name shown in Aura's credentials file. Do not commit `.env`. Shell environment variables take precedence over `.env` values.

```powershell
Copy-Item .env.example .env
notepad .env
```

The Neo4j server must already be running and reachable. Passing `--neo4j` verifies connectivity before recording. If the database is not available, run without `--neo4j` and the local CSV/JSON output still works.

## EEGStudio physical stream check

1. Start SmartBCI x24 and EEGStudio Acquisition 1.30.4, confirm the correct examination/device, montage and 500 Hz setting in the application.
2. Enable EEGStudio's LSL Outlet using its documented UI. The screenshots/context show an outlet menu, but do not yet prove that it is publishing.
3. In a second PowerShell window, activate the same virtual environment and run:

```powershell
python -m src.acquisition.cli discover
python -m src.acquisition.cli inspect --stream-index 0
```

4. Select the EEG stream by the index printed by `discover`; never assume ordering. Inspect name, type, source ID, channel count, format, and actual labels. A separate marker/event stream must be identified separately from its metadata; this CLI does not fabricate markers.
5. Record the original samples for 30 seconds (replace index only after inspection):

```powershell
python -m src.acquisition.cli record --stream-index 0 --duration 30
```

6. Once labels are confirmed and all expected channels are available, test inference:

```powershell
python -m src.acquisition.cli live --stream-index 0 --neo4j
```

Omit `--neo4j` for local-only testing. Stop with Ctrl+C; files are flushed on exit. `predict --duration 60` can be used for a bounded prediction run.

## Simulated tests

Run without a physical device:

```powershell
python -m unittest discover -s tests -v
python -m src.acquisition.cli --help
python -m src.acquisition.cli live --help
```

Automated tests exercise metadata/channel mapping with fake stream metadata, timestamps and gap detection, window lengths/overlap, exact feature calculation via the existing offline extractor, model artifact loading/prediction, and local CSV/JSON output. They are simulated tests; they do not test the Mitsar device, EEGStudio outlet, native LSL runtime, or a live Neo4j server.

### Two-terminal LSL-to-Neo4j simulation

The synthetic publisher generates arbitrary-unit sine waves; the recorded-data publisher replays a CSV prefix at the configured rate. Both are explicitly marked non-hardware. For a one-minute replay of one raw file, start Terminal 1 first; its startup delay lets the receiver connect before sample transmission begins:

```powershell
python -m src.acquisition.recorded_publisher --input data/raw/Chaitanya.csv --duration 60 --startup-delay 30
```

While it is publishing, in Terminal 2:

```powershell
python -m src.acquisition.cli discover
python -m src.acquisition.cli inspect --stream-index <SIMULATED_INDEX>
python -m src.acquisition.cli live --stream-index <REPLAY_INDEX> --duration 90 --disconnect-timeout 30 --db-raw-seconds 60 --neo4j
```

The receiver prints a generated session ID after closing. Retrieve stored records and up to the first minute of raw EEG in either terminal:

```powershell
python -m src.acquisition.cli db-list --limit 5
python -m src.acquisition.cli db-show --session-id <SESSION_ID>
python -m src.acquisition.cli db-samples --session-id <SESSION_ID> --duration-seconds 60
```

These commands read/write the Aura database from the git-ignored `.env`. The replay stream is explicitly identified as recorded CSV data in session metadata. Predictions from replay only demonstrate software plumbing; they do not assess stress detection.

## Output and interpretation

The receiver stores every incoming channel in the order declared by the stream metadata, without filtering, resampling, normalization, or artifact correction. Model features explicitly select the 19 trained EEG channels by label and reorder them into the saved model's documented order. Missing/duplicate labels stop inference. Nominal rate and the median timestamp-derived rate are both reported; inference stops if either cannot be reconciled with the model's 500 Hz rate (2% configurable tolerance). Units and raw/processed status remain unknown and are recorded as such. No unit conversion is performed.

The feature code is the same `src/features/window_features.py` implementation used offline. The model metadata currently records `deployment_ready: false`; live results are experimental and the metadata does not establish that the incoming signal scale or acquisition-side filters match training. Do not interpret probability as calibrated clinical confidence.

## Limitations and troubleshooting

- `discover` reporting no streams means the outlet is not visible to the local LSL resolver; confirm EEGStudio's outlet status, Windows firewall/network isolation, and that the receiver is on the same host/network.
- If labels are empty or differ (including case-independent exact names), inference stops. Inspect actual stream channel XML; do not choose the first 19 entries.
- A nominal rate of zero or observed rate that differs from 500 Hz blocks model predictions; no silent resampling is implemented.
- Marker stream discovery is descriptive. If a separate stream exists, pass its independently inspected index with `--event-stream-index`; event values/timestamps are then saved to a separate CSV and, when enabled, Neo4j event nodes linked to that session. Events are not inferred or synthesized, and automatic stream selection is intentionally disabled.
- The optional Neo4j writer performs writes on a background thread. Local CSV and JSON are the durable signal record. Review application logs and graph connectivity if writes fail.
- The current saved SVM has modest participant-wise CV results and no external validation. A streaming test confirms software compatibility only, not scientific validity.
- The actual EEGStudio LSL stream name, type, channel order/count, sample format/rate, units, raw-versus-processed status, marker publication, and hardware connectivity remain to be verified on the device.
