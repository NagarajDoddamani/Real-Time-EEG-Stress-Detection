# Device 1 — EEGStudio LSL recorder

This computer runs EEGStudio and the LSL Outlet. The Python recorder discovers
the stream and saves original samples locally while writing session metadata
and one-second sample chunks to Neo4j Aura. The participant ID and optional
display name are saved on the session before the first sample chunk is sent,
so Device 2 can identify the session even if Device 1 started first.

It does not run the model. Keep this project checkout and a local `.env` on
Device 1; copy `.env.example` to `.env` and fill it locally. Do not send the
credential file or commit `.env`.

## Run

From the repository root in PowerShell:

```powershell
python -m src.acquisition.cli discover
```

Use the printed stream index after checking the name, channel count, and labels:

```powershell
python -m src.acquisition.cli record --stream-index 0 --duration 1200 --neo4j --db-raw-seconds 1200 --participant-id P001 --participant-name "Participant 001" --output data/live/device1
```

Use `--duration 600` for 10 minutes or `--duration 1200` for 20 minutes. Keep
`--db-raw-seconds` at least as long as Device 2 needs to process. A session can
also run until Ctrl+C by omitting `--duration`. Each completed 500-sample,
one-second chunk is queued to Aura; Device 2 polls and consumes those chunks in
source order. `record` stores samples without running model inference. Local
CSV remains the complete durable recording. The graph sample path is currently
a prototype and should be load-tested before relying on it for long studies.

The PowerShell launcher asks for a participant code when one is not supplied:

```powershell
.\apps\device1-recorder\record.ps1 -StreamIndex 0 -DurationSeconds 1200 -StoreRawSeconds 1200
```

The device, EEGStudio connection, actual LSL labels/rate, units, filtering, and
event-stream behavior must be verified on the physical setup. Simulated CSV
replay is not a hardware test.
