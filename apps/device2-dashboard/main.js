const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');
const activeMonitors = new Map();

function pythonExecutable() {
  return process.env.EEG_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
}

function runService(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable(), ['-m', 'src.dashboard.device2_service', ...args], {
      cwd: projectRoot,
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
      let payload;
      try { payload = JSON.parse(stdout.trim()); }
      catch { return reject(new Error(stderr.trim() || stdout.trim() || `Python exited ${code}`)); }
      if (code !== 0 || !payload.ok) return reject(new Error(payload.error || stderr.trim() || `Python exited ${code}`));
      resolve(payload);
    });
  });
}

ipcMain.handle('sessions:list', async (_event, limit) => {
  const safeLimit = Number.isInteger(limit) ? Math.min(100, Math.max(1, limit)) : 50;
  return runService(['sessions', '--limit', String(safeLimit)]);
});

ipcMain.handle('session:predict', async (_event, sessionId) => {
  if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9-]{8,80}$/.test(sessionId)) {
    throw new Error('Select a valid session ID first.');
  }
  return runService(['predict-session', '--session-id', sessionId]);
});

ipcMain.handle('monitor:start', (event, sessionId) => {
  if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9-]{8,80}$/.test(sessionId)) {
    throw new Error('Select a valid session ID first.');
  }
  const owner = event.sender.id;
  if (activeMonitors.has(owner)) throw new Error('A session monitor is already running in this window.');
  const child = spawn(pythonExecutable(), ['-m', 'src.dashboard.device2_service', 'monitor-session',
    '--session-id', sessionId, '--poll-interval', '1'], {
    cwd: projectRoot,
    windowsHide: true,
    env: process.env,
    stdio: ['pipe', 'pipe', 'pipe']
  });
  activeMonitors.set(owner, child);
  event.sender.once('destroyed', () => {
    if (activeMonitors.get(owner) === child) {
      try { child.stdin.write('{"command":"stop"}\n'); } catch { child.kill(); }
    }
  });
  let stdoutBuffer = '';
  child.stdout.setEncoding('utf8').on('data', chunk => {
    stdoutBuffer += chunk;
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      try { event.sender.send('monitor:event', JSON.parse(line)); }
      catch { event.sender.send('monitor:event', { type: 'error', error: 'Invalid message from analysis service.' }); }
    }
  });
  let stderr = '';
  child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
  child.on('error', error => event.sender.send('monitor:event', { type: 'error', error: error.message }));
  child.on('close', code => {
    if (stdoutBuffer.trim()) {
      try { event.sender.send('monitor:event', JSON.parse(stdoutBuffer)); }
      catch { event.sender.send('monitor:event', { type: 'error', error: stdoutBuffer.trim() }); }
    }
    activeMonitors.delete(owner);
    event.sender.send('monitor:exit', { exitCode: code, detail: code === 0 ? '' : stderr.trim() });
  });
  return { started: true, sessionId };
});

ipcMain.handle('monitor:stop', event => {
  const child = activeMonitors.get(event.sender.id);
  if (!child) return { stopped: false };
  child.stdin.write('{"command":"stop"}\n');
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
    <div class="cell"><div class="label">Processed duration</div><div class="value">${htmlEscape(Number(summary.processed_duration_seconds || 0).toFixed(1))} seconds</div></div>
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
    width: 1180,
    height: 820,
    minWidth: 900,
    minHeight: 650,
    backgroundColor: '#0b1020',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  window.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', () => {
  for (const child of activeMonitors.values()) {
    try { child.stdin.write('{"command":"stop"}\n'); } catch { child.kill(); }
  }
});
