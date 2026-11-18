const ui = {
  sessionList: document.getElementById('sessionList'),
  refresh: document.getElementById('refreshButton'),
  start: document.getElementById('startButton'),
  stop: document.getElementById('stopButton'),
  pdf: document.getElementById('pdfButton'),
  status: document.getElementById('statusMessage'),
  eegCanvas: document.getElementById('eegCanvas'),
  stressCanvas: document.getElementById('stressCanvas')
};

const palette = ['#64e2c3','#8bb8ff','#d4a6ff','#ffbc79','#ff8298','#6ed7ef','#b6e573','#f18bc9',
  '#8bdb99','#bba7ff','#e0cf77','#68b4ba','#f19f72','#a9c2e8','#d1869d','#78d2ae','#b9a0e8','#d1a36d','#78a6d1'];
let sessions = [];
let selected = null;
let monitorActive = false;
let listing = false;
let predictions = [];
let waveformRows = [];
let channelLabels = [];
let latestSummary = null;
let selectedSessionForReport = null;
let lastReport = null;
let activeModelAlgorithm = 'svm_rbf';
let activeModelMetadata = {};
let removeMonitorListener = null;
let removeExitListener = null;

function setStatus(text, kind = '') {
  ui.status.textContent = text;
  ui.status.className = `status-message ${kind}`;
}

function setDot(elementId, kind) {
  const dot = document.getElementById(elementId);
  dot.className = kind;
}

function shortId(id) { return id ? `${id.slice(0, 8)}...${id.slice(-5)}` : '--'; }

function dateText(value) {
  if (!value) return 'Not recorded';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function clockText(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}` : `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
}

function setProfile(session) {
  selectedSessionForReport = session;
  const displayName = session.participant_name || session.participant_id || 'Participant details not recorded';
  document.getElementById('participantName').textContent = displayName;
  document.getElementById('participantId').textContent = session.participant_id ? `Participant code: ${session.participant_id}` : 'No participant code saved on Device 1';
  document.getElementById('participantAvatar').textContent = displayName.split(/\s+/).filter(Boolean).slice(0, 2).map(piece => piece[0].toUpperCase()).join('') || 'EEG';
  document.getElementById('sessionShortId').textContent = shortId(session.session_id);
  document.getElementById('streamName').textContent = session.stream_name || 'EEG stream';
  document.getElementById('channelCount').textContent = `${session.channel_count ?? '--'} channels`;
  document.getElementById('sessionStart').textContent = dateText(session.started_at_utc);
}

function selectSession(session) {
  if (monitorActive) return;
  selected = session;
  resetSessionDisplay();
  lastReport = null;
  ui.pdf.disabled = true;
  setProfile(session);
  ui.start.disabled = false;
  ui.stop.disabled = true;
  ui.pdf.disabled = !lastReport;
  setStatus(session.ended_at_utc
    ? 'Saved session selected. Processing will replay its stored chunks from the beginning at the dashboard cadence.'
    : 'Active session selected. Device 2 will consume newly stored chunks and any earlier chunks in order.', 'success');
  renderSessions();
}

function renderSessions() {
  document.getElementById('sessionCount').textContent = String(sessions.length);
  if (!sessions.length) {
    ui.sessionList.innerHTML = '<div class="empty-state">No stored EEG chunks found yet. Start Device 1 recording, then refresh.</div>';
    return;
  }
  ui.sessionList.replaceChildren(...sessions.map(session => {
    const button = document.createElement('button');
    button.className = `session-item ${selected?.session_id === session.session_id ? 'selected' : ''}`;
    button.disabled = monitorActive;
    const name = document.createElement('strong');
    name.className = 'session-person';
    name.textContent = session.participant_name || session.participant_id || 'Participant not identified';
    const identity = document.createElement('span');
    identity.className = 'session-identity';
    identity.textContent = session.participant_id ? `ID ${session.participant_id}` : 'Add participant code on Device 1';
    const meta = document.createElement('span');
    meta.className = 'session-meta';
    const source = session.device || session.stream_name || 'Source not recorded';
    meta.textContent = `${Number(session.raw_sample_count).toLocaleString()} samples  |  ${session.observed_sampling_rate_hz ?? session.nominal_sampling_rate_hz ?? '?'} Hz  |  ${source}`;
    const state = document.createElement('span');
    state.className = `session-state ${session.ended_at_utc ? 'ended' : 'active'}`;
    state.textContent = session.ended_at_utc ? 'STORED SESSION' : 'RECORDING';
    button.append(name, identity, meta, state);
    button.addEventListener('click', () => selectSession(session));
    return button;
  }));
}

async function refreshSessions() {
  if (listing || monitorActive) return;
  listing = true;
  ui.refresh.disabled = true;
  try {
    const response = await window.eegApp.listSessions(100);
    sessions = response.sessions || [];
    const liveCount = sessions.filter(item => /mitsar/i.test(String(item.device || '')) &&
      !/replay|simulat/i.test(String(item.device || ''))).length;
    document.getElementById('dbStatus').textContent = sessions.length
      ? `Aura connected · ${sessions.length} sessions · ${liveCount} Mitsar source`
      : 'Aura connected · no stored EEG chunks';
    setDot('dbDot', 'dot online');
    setDot('sideDot', monitorActive ? 'side-live' : 'side-ready');
    renderSessions();
    if (selected) {
      const update = sessions.find(item => item.session_id === selected.session_id);
      if (update) { selected = update; setProfile(update); }
    }
  } catch (error) {
    document.getElementById('dbStatus').textContent = 'Aura unavailable';
    setDot('dbDot', 'dot offline');
    setStatus(error.message, 'error');
  } finally {
    listing = false;
    ui.refresh.disabled = false;
  }
}

function setMonitorState(active) {
  monitorActive = active;
  ui.start.disabled = active || !selected;
  ui.stop.disabled = !active;
  ui.refresh.disabled = active;
  document.getElementById('sideStatus').textContent = active ? 'MONITORING' : 'READY';
  setDot('sideDot', active ? 'side-live' : 'side-ready');
  document.getElementById('waveformStatus').textContent = active ? 'LIVE DATABASE FEED' : 'PAUSED';
  renderSessions();
}

function resetSessionDisplay() {
  predictions = [];
  waveformRows = [];
  channelLabels = [];
  latestSummary = null;
  lastReport = null;
  ui.pdf.disabled = true;
  document.getElementById('sampleMetric').textContent = '0';
  document.getElementById('sampleSub').textContent = 'Awaiting database chunks';
  document.getElementById('rateMetric').innerHTML = '-- <em>Hz</em>';
  document.getElementById('elapsedMetric').textContent = '00:00';
  document.getElementById('windowMetric').textContent = '0';
  document.getElementById('latestPrediction').textContent = 'Waiting for EEG window';
  document.getElementById('latestPredictionTime').textContent = 'A complete four-second window is required first.';
  document.getElementById('latestScore').innerHTML = '--<small>%</small>';
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('historyCount').textContent = '0';
  document.getElementById('predictionList').innerHTML = '<div class="empty-state compact">Waiting for complete model windows.</div>';
  document.getElementById('trendEnd').textContent = '00:00';
  drawEeg();
  drawStress();
}

async function startMonitor() {
  if (!selected || monitorActive) return;
  resetSessionDisplay();
  setMonitorState(true);
  setStatus('Connecting to the selected database session...', 'working');
  try {
    const result = await window.eegApp.startMonitor(selected.session_id);
    if (!result.started) throw new Error('The monitor did not start.');
  } catch (error) {
    setMonitorState(false);
    setStatus(error.message, 'error');
  }
}

function handleMonitorEvent(event) {
  if (!event) return;
  if (event.type === 'error' || event.ok === false) {
    setMonitorState(false);
    setStatus(event.error || 'The database monitor stopped because of an error.', 'error');
    return;
  }
  if (event.type === 'monitor_started') {
    setProfile({ ...selected, ...event.session });
    activeModelAlgorithm = event.model?.algorithm || 'saved model';
    activeModelMetadata = event.model || {};
    channelLabels = event.session.channel_labels_used?.length
      ? event.session.channel_labels_used
      : (event.session.channel_labels || []);
    if (event.inference_enabled) {
      setStatus(`Monitoring ${event.session.participant_id || 'participant'}; reading stored chunks in order and running model windows.`, 'success');
    } else {
      setStatus(`Live database data is being retrieved and graphed. Predictions are blocked: ${event.inference_block_reason || 'model input could not be validated.'}`, 'error');
    }
    return;
  }
  if (event.type === 'sample_chunk') {
    for (const row of event.samples) waveformRows.push(row);
    if (waveformRows.length > 2500) waveformRows.splice(0, waveformRows.length - 2500);
    document.getElementById('sampleMetric').textContent = Number(event.samples_processed).toLocaleString();
    document.getElementById('sampleSub').textContent = `${event.chunk_index + 1} one-second chunks received`;
    document.getElementById('rateMetric').innerHTML = `${event.observed_rate_hz == null ? '--' : Number(event.observed_rate_hz).toFixed(1)} <em>Hz</em>`;
    document.getElementById('elapsedMetric').textContent = clockText(event.elapsed_seconds);
    document.getElementById('trendEnd').textContent = clockText(event.elapsed_seconds);
    document.getElementById('waveformStatus').textContent = `${channelLabels.length} CHANNELS / LIVE`;
    document.getElementById('waveEnd').textContent = clockText(event.elapsed_seconds);
    document.getElementById('waveStart').textContent = clockText(Math.max(0, event.elapsed_seconds - 5));
    drawEeg();
    return;
  }
  if (event.type === 'prediction') {
    predictions.push(event.prediction);
    document.getElementById('windowMetric').textContent = String(event.prediction_count);
    renderLatest(event.prediction);
    renderHistory();
    drawStress();
    return;
  }
  if (event.type === 'monitor_stopped') {
    latestSummary = event;
    setMonitorState(false);
    renderSummary(event);
    lastReport = {
      session: selectedSessionForReport || selected,
      summary: event,
      predictions,
      model: activeModelMetadata,
      model_algorithm: activeModelAlgorithm
    };
    ui.pdf.disabled = predictions.length === 0;
    if (!event.inference_enabled) {
      setStatus(`Raw database retrieval finished (${Number(event.sample_count).toLocaleString()} samples). Predictions were not produced: ${event.inference_block_reason || 'model input could not be validated.'}`, 'error');
    } else {
      const msg = event.finish_reason === 'device1_session_ended'
        ? 'Device 1 recording ended. Final report is ready.'
        : 'Processing stopped. Summary is ready; Device 1 recording was not stopped.';
      setStatus(msg, 'success');
    }
  }
}

function renderLatest(prediction) {
  const condition = prediction.predicted_condition || 'unknown';
  document.getElementById('latestPrediction').textContent = condition === 'stress' ? 'Predicted stress condition' : 'Predicted baseline condition';
  document.getElementById('latestPrediction').className = `latest-${condition}`;
  document.getElementById('latestPredictionTime').textContent = `Window ${prediction.window_index}  |  ${clockText(prediction.elapsed_seconds)} into recording`;
  const score = prediction.stress_probability;
  if (score == null || !Number.isFinite(Number(score))) {
    document.getElementById('latestScore').innerHTML = '--<small>%</small>';
    document.getElementById('scoreBar').style.width = '0%';
  } else {
    const percent = Math.max(0, Math.min(100, Number(score) * 100));
    document.getElementById('latestScore').innerHTML = `${percent.toFixed(1)}<small>%</small>`;
    document.getElementById('scoreBar').style.width = `${percent}%`;
  }
}

function renderHistory() {
  document.getElementById('historyCount').textContent = String(predictions.length);
  const latest = predictions.slice(-12).reverse();
  document.getElementById('predictionList').replaceChildren(...latest.map(item => {
    const row = document.createElement('div'); row.className = 'history-row';
    const number = document.createElement('span'); number.className = 'history-index'; number.textContent = String(item.window_index).padStart(3, '0');
    const label = document.createElement('span'); label.className = item.predicted_condition === 'stress' ? 'history-stress' : 'history-baseline'; label.textContent = item.predicted_condition === 'stress' ? 'STRESS CONDITION' : 'BASELINE CONDITION';
    const time = document.createElement('span'); time.className = 'history-time'; time.textContent = clockText(item.elapsed_seconds);
    row.append(number, label, time); return row;
  }));
}

function renderSummary(summary) {
  if (summary.window_count > 0) {
    const avg = summary.average_stress_score == null ? 'score unavailable' : `mean model score ${(summary.average_stress_score * 100).toFixed(1)}%`;
    document.getElementById('latestPrediction').textContent = summary.majority_window_condition === 'tie_or_no_windows'
      ? 'No majority window condition' : `Session summary: ${summary.majority_window_condition}`;
    document.getElementById('latestPredictionTime').textContent = `${summary.stress_window_count} stress / ${summary.baseline_window_count} baseline windows · ${avg} (uncalibrated)`;
    if (summary.average_stress_score != null) {
      document.getElementById('latestScore').innerHTML = `${(summary.average_stress_score * 100).toFixed(1)}<small>%</small>`;
      document.getElementById('scoreBar').style.width = `${Math.max(0, Math.min(100, summary.average_stress_score * 100))}%`;
    }
  } else {
    document.getElementById('latestPrediction').textContent = 'No complete windows processed';
    document.getElementById('latestPredictionTime').textContent = `${summary.sample_count} samples were retrieved.`;
  }
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawEeg() {
  const canvas = ui.eegCanvas;
  const result = canvasContext(canvas); if (!result) return;
  const { context: ctx, width, height } = result;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0a1220'; ctx.fillRect(0, 0, width, height);
  const labelWidth = 47;
  const plotWidth = width - labelWidth - 10;
  const rowHeight = height / Math.max(channelLabels.length, 1);
  ctx.font = '10px monospace';
  for (let i = 0; i < channelLabels.length; i++) {
    const top = i * rowHeight;
    const center = top + rowHeight / 2;
    ctx.strokeStyle = '#1e2b40'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(labelWidth, center); ctx.lineTo(width, center); ctx.stroke();
    ctx.fillStyle = palette[i % palette.length]; ctx.fillText(channelLabels[i] || `Ch ${String(i + 1).padStart(2, '0')}`, 4, center + 3);
    if (!waveformRows.length) continue;
    const series = waveformRows.map(row => Number(row[i])).filter(Number.isFinite);
    if (!series.length) continue;
    let low = Math.min(...series), high = Math.max(...series);
    if (high - low < 1e-12) { low -= 1; high += 1; }
    const mid = (high + low) / 2;
    const scale = (rowHeight * 0.34) / ((high - low) / 2);
    ctx.strokeStyle = palette[i % palette.length]; ctx.lineWidth = 1.15;
    ctx.beginPath();
    series.forEach((value, sampleIndex) => {
      const x = labelWidth + (sampleIndex / Math.max(series.length - 1, 1)) * plotWidth;
      const y = center - (value - mid) * scale;
      if (sampleIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  if (!channelLabels.length) {
    ctx.fillStyle = '#76859d'; ctx.font = '12px sans-serif'; ctx.fillText('Waiting for selected channel data...', 16, height / 2);
  }
}

function drawStress() {
  const canvas = ui.stressCanvas;
  const result = canvasContext(canvas); if (!result) return;
  const { context: ctx, width, height } = result;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0a1220'; ctx.fillRect(0, 0, width, height);
  const pad = { left: 42, right: 14, top: 13, bottom: 25 };
  const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
  ctx.font = '10px monospace'; ctx.fillStyle = '#718099'; ctx.strokeStyle = '#26344a'; ctx.lineWidth = 1;
  [0, 0.5, 1].forEach(value => {
    const y = pad.top + (1 - value) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(`${Math.round(value * 100)}%`, 7, y + 3);
  });
  const data = predictions.filter(item => Number.isFinite(Number(item.stress_probability)));
  if (data.length < 1) {
    ctx.fillStyle = '#718099'; ctx.font = '12px sans-serif'; ctx.fillText('Trend appears after the first complete model window.', pad.left + 8, height / 2);
    return;
  }
  const maxTime = Math.max(4, Number(data[data.length - 1].elapsed_seconds) || data.length * 2);
  ctx.strokeStyle = '#f0a476'; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((item, index) => {
    const x = pad.left + (Number(item.elapsed_seconds) / maxTime) * plotW;
    const score = Math.max(0, Math.min(1, Number(item.stress_probability)));
    const y = pad.top + (1 - score) * plotH;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (data.length > 1) {
    const item = data[data.length - 1];
    const x = pad.left + (Number(item.elapsed_seconds) / maxTime) * plotW;
    const y = pad.top + (1 - Math.max(0, Math.min(1, Number(item.stress_probability)))) * plotH;
    ctx.fillStyle = '#ffd4b5'; ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
  }
}

ui.refresh.addEventListener('click', refreshSessions);
ui.start.addEventListener('click', startMonitor);
ui.stop.addEventListener('click', async () => {
  ui.stop.disabled = true;
  setStatus('Stopping Device 2 processing and preparing the session summary. Device 1 recording continues.', 'working');
  try { await window.eegApp.stopMonitor(); }
  catch (error) { setStatus(error.message, 'error'); }
});
ui.pdf.addEventListener('click', async () => {
  if (!lastReport) return;
  ui.pdf.disabled = true;
  try {
    const saved = await window.eegApp.savePdfReport(lastReport);
    setStatus(saved.canceled ? 'PDF save canceled.' : `PDF report saved to ${saved.filePath}`, saved.canceled ? '' : 'success');
  } catch (error) { setStatus(error.message, 'error'); }
  finally { ui.pdf.disabled = !lastReport; }
});

removeMonitorListener = window.eegApp.onMonitorEvent(handleMonitorEvent);
removeExitListener = window.eegApp.onMonitorExit(result => {
  if (result.exitCode !== 0 && monitorActive) {
    setMonitorState(false);
    setStatus(result.detail || `Analysis service stopped (exit ${result.exitCode}).`, 'error');
  }
});

window.addEventListener('resize', () => { drawEeg(); drawStress(); });
refreshSessions();
window.setInterval(refreshSessions, 3000);
