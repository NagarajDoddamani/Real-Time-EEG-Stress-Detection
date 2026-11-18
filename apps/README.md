# Two-device project layout

```text
Device 1 (EEG acquisition computer)
  EEGStudio LSL Outlet
    -> src.acquisition.cli record + participant ID/display label
    -> local full CSV/JSON + one-second (500-sample) Aura chunks

Device 2 (analysis computer)
  Electron desktop UI
    -> select participant/session from Aura
    -> src.dashboard.device2_service polls each second from chunk zero
    -> validate channel names/order, timestamps and sample counts
    -> existing rolling 4 s windows / 2 s step / feature extractor / SVM
    -> live EEG traces + model-score trend + manual stop + PDF summary
```

Both computers need a local project checkout, Python environment, and their
own untracked `.env` containing access to the same Aura instance. Device 1
needs EEGStudio and the LSL runtime. Device 2 needs Node.js/Electron and the
Python inference dependencies. The source tree and trained artifact are shared
project code; acquisition and analysis entry points are separated by app.

Music playback and music-impact experiments are not implemented. Keep them as
a future module triggered only after the prediction flow is validated.
