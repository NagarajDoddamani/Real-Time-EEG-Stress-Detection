# Legacy LSL utilities

The supported user interface is now the combined Electron app in
`apps/device2-dashboard/`. It runs on the EEGStudio computer and handles stream
discovery, participant entry, live acquisition, model inference, visualization,
session storage, and report export in one workflow.

Launch it from the repository root:

```powershell
cd apps\device2-dashboard
npm start
```

The historical `record.ps1` and Python stream-monitor source remain available
for diagnostics, but they are not the recommended application workflow. The
standalone recorder does not run the trained model and should not be used to
start sessions intended for the Electron dashboard.

Raw EEG is saved incrementally to local files. Neo4j Aura stores participant
identity, session metadata, and prediction results only; it does not store raw
EEG sample chunks. Signal units, LSL channel labels, and live stream behavior
must still be verified against the physical Mitsar/EEGStudio installation.
