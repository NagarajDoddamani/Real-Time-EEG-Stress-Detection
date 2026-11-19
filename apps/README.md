# EEG desktop application

The Electron application in `EEGRT/` is the primary combined live system:

```text
EEGStudio LSL Outlet
  -> Electron stream discovery and participant form
  -> Python LSL receiver and local raw recording
  -> existing validated channel mapping, rolling windows, feature extraction, SVM
  -> live waveform and experimental-condition estimates in Electron
  -> Neo4j Aura participant, session summary, and prediction results
  -> downloadable PDF report
```

Raw EEG is written to local recording files and is not sent to Neo4j. Aura is
used for participant/session metadata and prediction results. The former
`device1-recorder/` scripts remain as standalone diagnostics and should not be
used as a separate production workflow when running the combined app.

The Electron app requires the repository Python environment, the trained model
artifacts, and a local `.env` configured for Aura. Start it with `npm start`
from `apps/EEGRT/`. The earlier `device2-dashboard/` copy is retained for
continuity; make future UI changes in `EEGRT/`.

Music playback and music-impact analysis remain a later development phase.
