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

To see all discovered stream entries together in a Python desktop window:

```powershell
.\apps\device1-recorder\ui.ps1
```

Enter the participant code and the user's full name in the **Record Session**
fields near the top of the window. Both are required for an Aura recording;
Device 2 uses the saved name in its selectable session list and user panel.
Use **Refresh streams**, then **Inspect all** to retrieve connected inlet
labels for every entry. Select the intended stream row before recording. If
two rows have the same stream UID, they are duplicate resolver entries for
the same LSL outlet; the recorder selects by UID rather than relying on row
order. If their UIDs differ, inspect and select the correct row explicitly.
Click **Start live preview** to see the incoming channel traces. Starting a
recording also starts this read-only preview automatically. When EEGStudio
does not publish channel labels, the graph uses generic `Ch 01`, `Ch 02`, …
labels; it does not guess electrode names. The display autoscale is visual
only, signal values are not changed, and the physical units remain unknown.

To store to Neo4j Aura, install the repository dependencies in the same Python
environment that launches the UI, and make sure that environment can read the
project `.env`:

```powershell
python -m pip install -r requirements.txt
```

The UI's Aura status distinguishes recorder startup from database readback.
It reports the session ID, then polls stored chunks and reports how many
chunks and samples it has read back from Aura. If it says **Aura not verified**
or reports a driver/connection error, do not treat that run as a successful
database test. The local waveform preview can still run when the Neo4j driver
is unavailable. A completed, nonzero readback count is evidence that this
session's chunks were stored and retrieved; the UI does not claim that until
it reads them back.

From the repository root in PowerShell:

```powershell
python -m src.acquisition.cli discover
```

Use the printed stream index after checking the name, channel count, and labels:

```powershell
python -m src.acquisition.cli record --stream-index 0 --duration 1200 --neo4j --db-raw-seconds 1200 --participant-id P001 --participant-name "Participant 001" --output data/live/device1
```

The PowerShell launcher defaults to manual stopping (`DurationSeconds = 0`):
it keeps recording until you press Ctrl+C in Device 1's terminal. To stop
automatically after 10 or 20 minutes, pass `-DurationSeconds 600` or
`-DurationSeconds 1200`. `StoreRawSeconds` is a separate maximum for how many
seconds are sent to Aura; its default is 1200. Keep it at least as long as
Device 2 needs to process. Each completed 500-sample, one-second chunk is
queued to Aura; Device 2 polls and consumes those chunks in source order.
`record` stores samples without running model inference. Local CSV remains the
complete durable recording. The graph sample path is currently a prototype and
should be load-tested before relying on it for long studies.

The PowerShell launcher asks for the participant code and user's full name when
they are not supplied:

```powershell
# Manual start and stop (stop this recording with Ctrl+C in this terminal)
.\apps\device1-recorder\record.ps1 -StreamIndex 0 -StoreRawSeconds 1200

# Or automatically stop after 20 minutes
.\apps\device1-recorder\record.ps1 -StreamIndex 0 -DurationSeconds 1200 -StoreRawSeconds 1200
```

The device, EEGStudio connection, actual LSL labels/rate, units, filtering, and
event-stream behavior must be verified on the physical setup. Simulated CSV
replay is not a hardware test.
