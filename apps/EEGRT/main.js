const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');

const appRoot = __dirname;
const activeLiveSessions = new Map();

function pythonExecutable() {
  const bundledVenv = path.join(appRoot, '.venv', 'Scripts', 'python.exe');
  return process.env.EEG_PYTHON || (require('node:fs').existsSync(bundledVenv)
    ? bundledVenv : (process.platform === 'win32' ? 'python' : 'python3'));
}

function runPythonScript(scriptName, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable(), [path.join(appRoot, scriptName), ...args], {
      cwd: appRoot,
      windowsHide: true,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', code => {
      const outputLines = stdout.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
      let payload;
      for (let index = outputLines.length - 1; index >= 0; index -= 1) {
        try { payload = JSON.parse(outputLines[index]); break; }
        catch { /* Ignore non-JSON diagnostic lines, such as liblsl startup logs. */ }
      }
      if (!payload) return reject(new Error(stderr.trim() || stdout.trim() || `Python exited ${code}`));
      if (code !== 0 || !payload.ok) return reject(new Error(payload.error || stderr.trim() || `Python exited ${code}`));
      resolve(payload);
    });
  });
}

function runService(args) {
  return runPythonScript('eeg_service.py', args);
}

ipcMain.handle('sessions:list', async (_event, limit) => {
  const safeLimit = Number.isInteger(limit) ? Math.min(100, Math.max(1, limit)) : 50;
  return runService(['sessions', '--limit', String(safeLimit)]);
});

ipcMain.handle('lsl:discover', async () => runPythonScript('eeg_cli.py', ['discover', '--json']));

ipcMain.handle('session:report', async (_event, sessionId) => {
  if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9-]{8,80}$/.test(sessionId)) {
    throw new Error('Select a valid session ID first.');
  }
  return runService(['session-report', '--session-id', sessionId]);
});

ipcMain.handle('live:start', async (event, details) => {
  const participantId = String(details?.participantId || '').trim();
  const participantName = String(details?.participantName || '').trim();
  const streamUid = String(details?.streamUid || '').trim();
  if (!participantId || participantId.length > 100) throw new Error('Enter a participant ID (up to 100 characters).');
  if (!participantName || participantName.length > 160) throw new Error('Enter the participant full name (up to 160 characters).');
  if (!streamUid || streamUid.length > 512) throw new Error('Select an available EEGStudio LSL stream.');
  const owner = event.sender.id;
  if (activeLiveSessions.has(owner)) throw new Error('A live EEG session is already active.');
  const controlDir = path.join(appRoot, 'data', 'live', 'control');
  await fs.mkdir(controlDir, { recursive: true });
  const stopFile = path.join(controlDir, `stop_${Date.now()}_${Math.random().toString(16).slice(2)}.flag`);
  try { await fs.unlink(stopFile); } catch { /* unique filename */ }
  const child = spawn(pythonExecutable(), [path.join(appRoot, 'eeg_cli.py'), 'live',
    '--stream-uid', streamUid,
    '--participant-id', participantId,
    '--participant-name', participantName,
    '--output', path.join(appRoot, 'data', 'live', 'combined-app'),
    '--stop-file', stopFile,
    '--json-events', '--allow-unmapped'], {
    cwd: appRoot,
    windowsHide: true,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  activeLiveSessions.set(owner, { child, stopFile });
  const sendLiveEvent = (channel, payload) => {
    if (!event.sender.isDestroyed()) event.sender.send(channel, payload);
  };
  let stdoutBuffer = '';
  let reportedError = '';
  child.stdout.setEncoding('utf8').on('data', chunk => {
    stdoutBuffer += chunk;
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const payload = JSON.parse(line);
        if (payload.type === 'error' && payload.error) reportedError = payload.error;
        sendLiveEvent('live:event', payload);
      }
      catch { /* Suppress unstructured liblsl/sklearn diagnostics; actionable errors use JSON events. */ }
    }
  });
  let stderr = '';
  child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
  child.on('error', error => sendLiveEvent('live:event', { type: 'error', error: error.message }));
  child.on('close', async code => {
    if (stdoutBuffer.trim()) {
      try {
        const payload = JSON.parse(stdoutBuffer);
        if (payload.type === 'error' && payload.error) reportedError = payload.error;
        sendLiveEvent('live:event', payload);
      }
      catch { /* Suppress unstructured diagnostics so they cannot flood the dashboard. */ }
    }
    activeLiveSessions.delete(owner);
    try { await fs.unlink(stopFile); } catch { /* already removed */ }
    const conciseStderr = stderr.split(/\r?\n/).map(line => line.trim())
      .filter(line => /fatal|error|exception|traceback|unable|failed/i.test(line))
      .slice(-5).join('\n');
    sendLiveEvent('live:exit', {
      exitCode: code,
      detail: code === 0 ? '' : (reportedError || conciseStderr || `Acquisition process exited with code ${code}.`)
    });
  });
  event.sender.once('destroyed', async () => {
    if (activeLiveSessions.get(owner)?.child === child) {
      try { await fs.writeFile(stopFile, 'stop\n', 'utf8'); } catch { child.kill(); }
    }
  });
  return { started: true };
});

ipcMain.handle('live:stop', async event => {
  const active = activeLiveSessions.get(event.sender.id);
  if (!active) return { stopped: false };
  await fs.writeFile(active.stopFile, 'stop\n', 'utf8');
  return { stopped: true };
});

function htmlEscape(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

ipcMain.handle('report:pdf', async (_event, report) => {
  if (!report || typeof report !== 'object' || !report.session || !Array.isArray(report.predictions)) {
    throw new Error('There is no completed monitoring report to export.');
  }
  const safeId = String(report.session.session_id || 'session').replace(/[^a-zA-Z0-9_-]/g, '_');
  const choice = await dialog.showSaveDialog({
    title: 'Save EEG session report',
    defaultPath: path.join(app.getPath('downloads'), `EEG_Report_${safeId}.pdf`),
    filters: [{ name: 'PDF document', extensions: ['pdf'] }]
  });
  if (choice.canceled || !choice.filePath) return { canceled: true };
  const session = report.session;
  const summary = report.summary || {};
  const rows = report.predictions.map(item => `<tr><td>${htmlEscape(item.window_index)}</td><td>${htmlEscape(Number(item.elapsed_seconds || 0).toFixed(1))}</td><td>${htmlEscape(item.predicted_condition)}</td><td>${item.stress_probability == null ? 'Not available' : htmlEscape(Number(item.stress_probability).toFixed(4))}</td></tr>`).join('');
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
    body{font:12px Arial,sans-serif;color:#172033;margin:28px}h1{margin:0 0 5px;color:#10203c}h2{margin:24px 0 9px;color:#234}p{color:#566276;line-height:1.5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.cell{padding:9px;border:1px solid #dbe2ec;border-radius:5px}.label{font-size:9px;text-transform:uppercase;color:#65748b;letter-spacing:1px}.value{font-weight:bold;margin-top:4px}table{border-collapse:collapse;width:100%;font-size:10px}th,td{text-align:left;padding:6px;border-bottom:1px solid #e1e6ef}th{background:#eef2f7}footer{margin-top:24px;color:#768196;font-size:9px}
    </style></head><body><h1>EEG Session Report</h1><p>Baseline/stress experimental-condition model estimates. These are not clinical diagnoses and the model has no external validation.</p>
    <h2>Participant and session</h2><div class="grid">
    <div class="cell"><div class="label">Participant ID</div><div class="value">${htmlEscape(session.participant_id || 'Not recorded')}</div></div>
    <div class="cell"><div class="label">Participant name / label</div><div class="value">${htmlEscape(session.participant_name || 'Not recorded')}</div></div>
    <div class="cell"><div class="label">Session ID</div><div class="value">${htmlEscape(session.session_id)}</div></div>
    <div class="cell"><div class="label">Session started</div><div class="value">${htmlEscape(session.started_at_utc || 'Unknown')}</div></div>
    <div class="cell"><div class="label">Processed duration</div><div class="value">${htmlEscape(Number(summary.processed_duration_seconds ?? summary.duration_seconds ?? 0).toFixed(1))} seconds</div></div>
    <div class="cell"><div class="label">Samples / rate</div><div class="value">${htmlEscape(summary.sample_count || 0)} / ${htmlEscape(session.nominal_sampling_rate_hz || 500)} Hz</div></div>
    <div class="cell"><div class="label">Channels</div><div class="value">${htmlEscape((session.channel_labels_used || []).join(', '))}</div></div>
    <div class="cell"><div class="label">Model / participant-wise CV</div><div class="value">${htmlEscape(report.model_algorithm || 'Saved model')} / accuracy ${report.model?.validation_accuracy == null ? 'not reported' : htmlEscape((Number(report.model.validation_accuracy) * 100).toFixed(1) + '%')} / macro F1 ${report.model?.validation_macro_f1 == null ? 'not reported' : htmlEscape(Number(report.model.validation_macro_f1).toFixed(3))}</div></div>
    </div><h2>Summary</h2><div class="grid">
    <div class="cell"><div class="label">Most frequent window class</div><div class="value">${htmlEscape(summary.majority_window_condition || 'None')}</div></div>
    <div class="cell"><div class="label">Mean model stress score</div><div class="value">${summary.average_stress_score == null ? 'Not available' : htmlEscape((Number(summary.average_stress_score) * 100).toFixed(1) + '%')}</div></div>
    <div class="cell"><div class="label">Stress / baseline windows</div><div class="value">${htmlEscape(summary.stress_window_count || 0)} / ${htmlEscape(summary.baseline_window_count || 0)}</div></div>
    <div class="cell"><div class="label">Samples processed</div><div class="value">${htmlEscape(summary.sample_count || 0)}</div></div></div>
    <h2>Window-by-window results</h2><table><thead><tr><th>Window</th><th>Elapsed (s)</th><th>Predicted condition</th><th>Stress score*</th></tr></thead><tbody>${rows}</tbody></table>
    <footer>* The model output score is not a calibrated clinical probability. Data units are unverified. Generated ${htmlEscape(new Date().toISOString())}.</footer></body></html>`;
  const reportWindow = new BrowserWindow({ show: false, webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false } });
  try {
    await reportWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
    const pdf = await reportWindow.webContents.printToPDF({ pageSize: 'A4', printBackground: true });
    await fs.writeFile(choice.filePath, pdf);
    return { canceled: false, filePath: choice.filePath };
  } finally { reportWindow.destroy(); }
});

function createWindow() {
  const window = new BrowserWindow({
    title: 'EEGRT | Real-Time EEG',
    width: 1480,
    height: 980,
    minWidth: 1000,
    minHeight: 740,
    backgroundColor: '#0b1020',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  window.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', () => {
  for (const { child, stopFile } of activeLiveSessions.values()) {
    fs.writeFile(stopFile, 'stop\n', 'utf8').catch(() => child.kill());
  }
});
