let EXPECTED_CHANNELS = [
  'Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'
];
let MODEL_RATE_HZ = 500;
const palette = ['#64e2c3','#8bb8ff','#d4a6ff','#ffbc79','#ff8298','#6ed7ef','#b6e573','#f18bc9',
  '#8bdb99','#bba7ff','#e0cf77','#68b4ba','#f19f72','#a9c2e8','#d1869d','#78d2ae','#b9a0e8','#d1a36d','#78a6d1'];
const ui = {
  stream: document.getElementById('streamSelect'),
  streamInfo: document.getElementById('streamInfo'),
  participantId: document.getElementById('participantIdInput'),
  participantName: document.getElementById('participantNameInput'),
  refreshStreams: document.getElementById('refreshStreamsButton'),
  refreshSessions: document.getElementById('refreshSessionsButton'),
  start: document.getElementById('startButton'), stop: document.getElementById('stopButton'),
  report: document.getElementById('reportButton'), status: document.getElementById('statusMessage'),
  sessionList: document.getElementById('sessionList'), eegCanvas: document.getElementById('eegCanvas'),
  stressCanvas: document.getElementById('stressCanvas')
};

let streams = [];
let active = false;
let listings = false;
let samples = [];
let labels = [];
let predictions = [];
let activeSession = null;
let activeModel = null;
let finalReport = null;

function setStatus(message, kind = '') {
  ui.status.textContent = message;
  ui.status.className = `status-message ${kind}`;
}
function setDot(id, state) { document.getElementById(id).className = state; }
function shortUid(value) { return value ? `${value.slice(0, 8)}…${value.slice(-6)}` : 'UID unavailable'; }
function clockText(seconds) {
  const n = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
  return h ? `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function selectedStream() { return streams.find(item => item.stream_id === ui.stream.value) || null; }
function updateStartState() {
  ui.start.disabled = active || !selectedStream() || !ui.participantId.value.trim() || !ui.participantName.value.trim();
}

function renderStreamInfo() {
  const stream = selectedStream();
  ui.streamInfo.className = 'stream-info';
  if (!stream) {
    ui.streamInfo.textContent = 'Start EEGStudio’s LSL Outlet, then discover and select its EEG stream.';
    updateStartState(); return;
  }
  const labelSet = new Set((stream.channel_labels || []).map(label => String(label).trim().toLowerCase()).filter(Boolean));
  const missing = EXPECTED_CHANNELS.filter(channel => !labelSet.has(channel.toLowerCase()));
  const labelsKnown = missing.length === 0;
  const rateCompatible = Number(stream.nominal_srate) > 0 && Math.abs(Number(stream.nominal_srate) - MODEL_RATE_HZ) / MODEL_RATE_HZ <= 0.02;
  const values = [
    `${stream.name || 'Unnamed stream'} · ${stream.stream_type || 'type unknown'}`,
    `Source ${stream.source_id || 'unknown'} · UID ${shortUid(stream.stream_id)}`,
    `${stream.channel_count} channels · nominal ${Number(stream.nominal_srate || 0).toFixed(1)} Hz · ${stream.channel_format || 'format unknown'}`
  ];
  const labelsSummary = (stream.channel_labels || []).some(Boolean)
    ? `Labels: ${(stream.channel_labels || []).map(label => String(label || 'unnamed')).join(', ')}`
    : 'Channel labels are not exposed by the resolver';
  if (labelsKnown && rateCompatible) {
    values.push(`${labelsSummary}. All 19 model channels are present; connected inlet metadata is checked again before inference.`);
    ui.streamInfo.classList.add('ready');
  } else {
    const warnings = [];
    if (!rateCompatible) warnings.push(`nominal rate is not compatible with the model’s verified ${MODEL_RATE_HZ} Hz`);
    if (missing.length) warnings.push((stream.channel_labels || []).some(Boolean)
      ? `missing model labels: ${missing.join(', ')}`
      : 'channel labels are not exposed by the resolver; connected inlet metadata will be checked');
    values.push(`${labelsSummary}. ${warnings.join(' · ')}. Predictions stop if rate or channel mapping is invalid.`);
    ui.streamInfo.classList.add('warning');
  }
  ui.streamInfo.textContent = values.join('  |  ');
  updateStartState();
}

async function discoverStreams() {
  ui.refreshStreams.disabled = true;
  setDot('lslDot', '');
  document.getElementById('lslStatus').textContent = 'Searching LSL…';
  try {
    const response = await window.eegApp.discoverStreams();
    streams = response.streams || [];
    if (response.model_input?.channels?.length) EXPECTED_CHANNELS = response.model_input.channels;
    if (Number(response.model_input?.sampling_frequency_hz) > 0) MODEL_RATE_HZ = Number(response.model_input.sampling_frequency_hz);
    const previous = ui.stream.value;
    ui.stream.replaceChildren(new Option('Select the EEGStudio outlet', ''));
    streams.forEach((stream, index) => {
      const text = `${index + 1}. ${stream.name || 'Unnamed'} · ${stream.stream_type || 'unknown'} · ${stream.channel_count} ch · ${Number(stream.nominal_srate || 0).toFixed(0)} Hz · ${shortUid(stream.stream_id)}`;
      ui.stream.add(new Option(text, stream.stream_id || ''));
    });
    if (streams.some(item => item.stream_id === previous)) ui.stream.value = previous;
    document.getElementById('lslStatus').textContent = streams.length ? `${streams.length} stream${streams.length === 1 ? '' : 's'} found` : 'No LSL streams found';
    setDot('lslDot', streams.length ? 'online' : 'offline');
    renderStreamInfo();
    if (!streams.length) setStatus('No LSL outlet found. Start EEGStudio’s LSL Outlet and refresh streams.', 'error');
  } catch (error) {
    document.getElementById('lslStatus').textContent = 'LSL discovery failed';
    setDot('lslDot', 'offline');
    setStatus(error.message, 'error');
  } finally { ui.refreshStreams.disabled = false; }
}

function resetDisplay() {
  samples = []; labels = []; predictions = []; finalReport = null;
  document.getElementById('sampleMetric').textContent = '0';
  document.getElementById('sampleSub').textContent = 'Waiting for session';
  document.getElementById('rateMetric').innerHTML = '-- <em>Hz</em>';
  document.getElementById('elapsedMetric').textContent = '00:00';
  document.getElementById('windowMetric').textContent = '0';
  document.getElementById('latestPrediction').textContent = 'Waiting for EEG window';
  document.getElementById('latestPrediction').className = '';
  document.getElementById('latestPredictionTime').textContent = 'A complete 4-second window is required.';
  document.getElementById('latestScore').textContent = '--%';
  document.getElementById('scoreBar').style.width = '0%';
  document.getElementById('historyCount').textContent = '0';
  document.getElementById('predictionList').innerHTML = '<div class="empty-state">Predictions will appear after the first complete window.</div>';
  document.getElementById('waveformStatus').textContent = 'NO SIGNAL';
  document.getElementById('waveDot').className = '';
  document.getElementById('waveStart').textContent = '--';
  document.getElementById('waveEnd').textContent = '--';
  document.getElementById('trendEnd').textContent = '00:00';
  drawEeg(); drawStress();
}

async function startLive() {
  const stream = selectedStream();
  if (!stream || active) return;
  if (!ui.participantId.value.trim() || !ui.participantName.value.trim()) {
    setStatus('Enter both the participant ID and full name before starting.', 'error'); return;
  }
  resetDisplay();
  active = true;
  activeSession = null;
  setDot('sideDot', 'side-live');
  document.getElementById('sideStatus').textContent = 'STARTING';
  ui.stop.disabled = false; ui.start.disabled = true;
  ui.stream.disabled = true; ui.participantId.disabled = true; ui.participantName.disabled = true;
  setStatus('Connecting to the selected LSL stream and validating channel metadata…', 'working');
  try {
    await window.eegApp.startLive({ streamUid: stream.stream_id,
      participantId: ui.participantId.value.trim(), participantName: ui.participantName.value.trim() });
  } catch (error) {
    active = false; restoreControls(); setStatus(error.message, 'error');
  }
}

async function stopLive() {
  if (!active) return;
  ui.stop.disabled = true;
  setStatus('Stopping acquisition safely and saving the session summary and predictions…', 'working');
  try { await window.eegApp.stopLive(); }
  catch (error) { ui.stop.disabled = false; setStatus(error.message, 'error'); }
}

function restoreControls() {
  active = false;
  ui.stream.disabled = false; ui.participantId.disabled = false; ui.participantName.disabled = false;
  ui.stop.disabled = true;
  document.getElementById('sideStatus').textContent = 'READY';
  setDot('sideDot', 'side-ready');
  document.getElementById('waveformStatus').textContent = samples.length ? 'SESSION COMPLETE' : 'NO SIGNAL';
  ui.report.disabled = !finalReport;
  updateStartState();
}

function handleLiveEvent(event) {
  if (!event) return;
  if (event.type === 'error') {
    setStatus(event.error || 'Live acquisition failed.', 'error');
    return;
  }
  if (event.type === 'log') { setStatus(event.message, 'working'); return; }
  if (event.type === 'session_started') {
    activeSession = event.session;
    labels = (event.session.channel_labels || []).map((label, index) => String(label || `Ch ${String(index + 1).padStart(2,'0')}`));
    const missing = EXPECTED_CHANNELS.filter(name => !new Set(labels.map(v => v.toLowerCase())).has(name.toLowerCase()));
    document.getElementById('waveformStatus').textContent = `${event.session.channel_count} CHANNELS · LIVE`;
    document.getElementById('waveDot').className = 'online';
    document.getElementById('sideStatus').textContent = 'LIVE';
    if (!event.inference_enabled) {
      setStatus(`Live signal is being recorded locally. Model predictions are disabled: ${event.inference_block_reason || 'model input validation failed.'}`, 'error');
    } else {
      setStatus(`Live session started for ${event.session.participant_name}. The trained model will predict after the first complete window.`, 'success');
    }
    return;
  }
  if (event.type === 'status' || event.type === 'notice') {
    if (!activeSession || event.type === 'notice') setStatus(event.message, event.type === 'notice' ? '' : 'working');
    return;
  }
  if (event.type === 'inference_blocked') {
    setStatus(`Signal acquisition continues, but model inference has stopped: ${event.reason}`, 'error');
    return;
  }
  if (event.type === 'signal_quality_warning') {
    setStatus(event.message, 'error');
    return;
  }
  if (event.type === 'sample_chunk') {
    if (activeSession && activeSession.first_timestamp == null && event.timestamps?.length) activeSession.first_timestamp = event.timestamps[0];
    samples.push(...event.samples);
    if (samples.length > 2500) samples.splice(0, samples.length - 2500);
    const total = Number(event.first_sample_index || 0) + Number(event.sample_count || 0);
    document.getElementById('sampleMetric').textContent = total.toLocaleString();
    document.getElementById('sampleSub').textContent = `${event.sample_count} new samples received`;
    const timestamps = event.timestamps || [];
    if (timestamps.length > 1) {
      const intervals = [];
      for (let i = 1; i < timestamps.length; i++) if (timestamps[i] > timestamps[i - 1]) intervals.push(timestamps[i] - timestamps[i - 1]);
      if (intervals.length) {
        intervals.sort((a,b) => a-b);
        const rate = 1 / intervals[Math.floor(intervals.length / 2)];
        document.getElementById('rateMetric').innerHTML = `${rate.toFixed(1)} <em>Hz</em>`;
      }
      document.getElementById('elapsedMetric').textContent = clockText(timestamps[timestamps.length - 1] - (activeSession?.first_timestamp || timestamps[0]));
      document.getElementById('waveEnd').textContent = clockText(total / Number(activeSession?.nominal_sampling_rate_hz || 500));
      document.getElementById('waveStart').textContent = clockText(Math.max(0, total / Number(activeSession?.nominal_sampling_rate_hz || 500) - 5));
      document.getElementById('trendEnd').textContent = clockText(total / Number(activeSession?.nominal_sampling_rate_hz || 500));
    }
    drawEeg();
    return;
  }
  if (event.type === 'prediction') {
    predictions.push(event.prediction);
    renderPrediction(event.prediction);
    drawStress();
    return;
  }
  if (event.type === 'session_stopped') {
    activeSession = event.session;
    finalReport = { session: event.session, summary: event.summary || {},
      predictions: event.predictions || predictions, model: event.model || null,
      model_algorithm: event.model?.algorithm || 'unknown' };
    restoreControls();
    document.getElementById('dbStatus').textContent = 'Local archive'; setDot('dbDot', 'online');
    setStatus(event.summary?.window_count
      ? `Session saved locally. ${event.summary.window_count} prediction windows; report is ready to download.`
      : `Session saved locally. No predictions were produced.${event.inference_block_reason ? ` ${event.inference_block_reason}` : ''}`,
      event.summary?.window_count ? 'success' : '');
    refreshSessions();
  }
}

function renderPrediction(prediction) {
  const condition = prediction.predicted_condition || 'unknown';
  const title = document.getElementById('latestPrediction');
  title.textContent = condition === 'stress' ? 'Predicted stress condition' : 'Predicted baseline condition';
  title.className = condition;
  document.getElementById('latestPredictionTime').textContent = `Window ${prediction.window_index} · ${clockText(prediction.elapsed_seconds)} into session`;
  const score = prediction.stress_probability;
  if (score == null || !Number.isFinite(Number(score))) {
    document.getElementById('latestScore').textContent = 'Not available';
    document.getElementById('scoreBar').style.width = '0%';
  } else {
    const percent = Math.max(0, Math.min(100, Number(score) * 100));
    document.getElementById('latestScore').textContent = `${percent.toFixed(1)}%`;
    document.getElementById('scoreBar').style.width = `${percent}%`;
  }
  document.getElementById('windowMetric').textContent = String(prediction.window_index);
  document.getElementById('historyCount').textContent = String(predictions.length);
  const items = predictions.slice(-10).reverse();
  ui.predictionList.replaceChildren(...items.map(item => {
    const row = document.createElement('div'); row.className = 'prediction-row';
    const index = document.createElement('span'); index.textContent = `#${String(item.window_index).padStart(3,'0')}`;
    const label = document.createElement('strong'); label.className = item.predicted_condition;
    label.textContent = item.predicted_condition === 'stress' ? 'STRESS CONDITION' : 'BASELINE CONDITION';
    const time = document.createElement('time'); time.textContent = clockText(item.elapsed_seconds);
    row.append(index, label, time); return row;
  }));
}

async function refreshSessions() {
  if (listings) return;
  listings = true; ui.refreshSessions.disabled = true;
  try {
    const result = await window.eegApp.listSessions(100);
    const sessions = result.sessions || [];
    document.getElementById('dbStatus').textContent = `Local archive · ${sessions.length} sessions`;
    setDot('dbDot', 'online');
    if (!sessions.length) {
      ui.sessionList.innerHTML = '<div class="empty-state">No saved sessions yet. Completed sessions will appear here.</div>';
      return;
    }
    ui.sessionList.replaceChildren(...sessions.map(session => {
      const item = document.createElement('article'); item.className = 'archive-item';
      const details = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = session.participant_name || session.participant_id || 'Participant not identified';
      const meta = document.createElement('small');
      meta.textContent = `${session.participant_id || 'No participant ID'} · ${Number(session.sample_count || 0).toLocaleString()} samples · ${session.prediction_count_already_stored || 0} windows · ${session.started_at_utc || 'time unknown'}`;
      details.append(name, meta);
      const actions = document.createElement('div'); actions.className = 'archive-actions';
      const view = document.createElement('button'); view.className = 'button secondary'; view.textContent = 'View results';
      view.disabled = false;
      view.addEventListener('click', () => loadSavedSession(session.session_id));
      const download = document.createElement('button'); download.className = 'button secondary'; download.textContent = 'PDF';
      download.disabled = false;
      download.addEventListener('click', () => downloadSavedReport(session.session_id));
      actions.append(view, download);
      item.append(details, actions); return item;
    }));
  } catch (error) {
    document.getElementById('dbStatus').textContent = 'Local archive error'; setDot('dbDot','offline');
    const message = document.createElement('div'); message.className = 'empty-state'; message.textContent = error.message;
    ui.sessionList.replaceChildren(message);
  } finally { listings = false; ui.refreshSessions.disabled = false; }
}

function reportFromSaved(record) {
  const session = record.session || {};
  const summary = {
    sample_count: session.sample_count || 0,
    processed_duration_seconds: session.sample_count && session.nominal_sampling_rate_hz
      ? session.sample_count / session.nominal_sampling_rate_hz : 0,
    stress_window_count: session.stress_window_count || 0,
    baseline_window_count: session.baseline_window_count || 0,
    average_stress_score: session.average_stress_score ?? null,
    majority_window_condition: session.majority_window_condition || 'not available'
  };
  return { session, summary, predictions: record.predictions || [],
    model_algorithm: session.model_algorithm || 'saved model',
    model: { validation_accuracy: session.model_validation_accuracy ?? null,
      validation_macro_f1: session.model_validation_macro_f1 ?? null } };
}

async function saveReport(report) {
  const result = await window.eegApp.savePdfReport(report);
  if (!result.canceled) setStatus(`Report saved to ${result.filePath}`, 'success');
}
async function downloadSavedReport(sessionId) {
  try {
    setStatus('Reading saved participant results from this computer…', 'working');
    const record = await window.eegApp.getSessionReport(sessionId);
    await saveReport(reportFromSaved(record));
  } catch (error) { setStatus(error.message, 'error'); }
}

async function loadSavedSession(sessionId) {
  if (active) { setStatus('Stop the live session before loading an archived result.', 'error'); return; }
  try {
    setStatus('Loading locally saved participant results…', 'working');
    const record = await window.eegApp.getSessionReport(sessionId);
    const report = reportFromSaved(record);
    const session = report.session;
    resetDisplay();
    predictions = report.predictions;
    finalReport = report;
    activeSession = session;
    labels = session.channel_labels_in_stream_order || session.channel_labels || [];
    ui.participantId.value = session.participant_id || '';
    ui.participantName.value = session.participant_name || '';
    document.getElementById('sampleMetric').textContent = Number(report.summary.sample_count || 0).toLocaleString();
    document.getElementById('sampleSub').textContent = 'Saved session samples; EEG remains local';
    document.getElementById('rateMetric').innerHTML = `${Number(session.observed_sampling_rate_hz || session.nominal_sampling_rate_hz || 0).toFixed(1)} <em>Hz</em>`;
    document.getElementById('elapsedMetric').textContent = clockText(report.summary.processed_duration_seconds);
    document.getElementById('windowMetric').textContent = String(predictions.length);
    document.getElementById('waveformStatus').textContent = 'ARCHIVED · EEG LOCAL';
    document.getElementById('waveDot').className = '';
    document.getElementById('latestPrediction').textContent = 'Saved session results';
    document.getElementById('latestPrediction').className = '';
    document.getElementById('latestPredictionTime').textContent = `${session.participant_name || session.participant_id || 'Participant'} · ${session.started_at_utc || 'time unknown'}`;
    document.getElementById('historyCount').textContent = String(predictions.length);
    const last = predictions[predictions.length - 1];
    if (last) renderPrediction(last);
    else document.getElementById('predictionList').innerHTML = '<div class="empty-state">No predictions were saved for this session.</div>';
    drawEeg(); drawStress();
    ui.report.disabled = false;
    setStatus(`Loaded ${predictions.length} saved prediction windows for ${session.participant_name || session.participant_id}. Raw waveform data is only available in the local recording.`, 'success');
  } catch (error) { setStatus(error.message, 'error'); }
}

function canvasSize(canvas) {
  const ratio = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  canvas.width = Math.floor(rect.width * ratio); canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0);
  return { ctx, width: rect.width, height: rect.height };
}
function drawEeg() {
  const result = canvasSize(ui.eegCanvas); if (!result) return;
  const {ctx,width,height}=result; ctx.clearRect(0,0,width,height); ctx.fillStyle='#091320';ctx.fillRect(0,0,width,height);
  if (!samples.length) { ctx.fillStyle='#71839a';ctx.font='12px Segoe UI';ctx.fillText('Waiting for LSL samples',18,height/2);return; }
  const channelCount = Math.min(samples[samples.length-1].length, 32), left=62, right=width-10, plotH=height/channelCount;
  for(let channel=0;channel<channelCount;channel++) {
    const yTop=channel*plotH, mid=yTop+plotH/2, label=labels[channel] || `Ch ${String(channel+1).padStart(2,'0')}`;
    ctx.strokeStyle='#1d2c3f';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,mid);ctx.lineTo(right,mid);ctx.stroke();
    ctx.fillStyle=palette[channel%palette.length];ctx.font='9px Consolas';ctx.fillText(label.slice(0,9),5,mid+3);
    const values=samples.map(row=>Number(row[channel])).filter(Number.isFinite);if(values.length<2)continue;
    let low=Math.min(...values),high=Math.max(...values);if(high-low<1e-12){low-=1;high+=1;}
    const center=(high+low)/2, scale=(plotH*.36)/((high-low)/2);
    ctx.strokeStyle=palette[channel%palette.length];ctx.lineWidth=1;ctx.beginPath();
    values.forEach((value,index)=>{const x=left+index*(right-left)/Math.max(values.length-1,1),y=mid-(value-center)*scale;if(index===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
  }
}
function drawStress() {
  const result=canvasSize(ui.stressCanvas);if(!result)return;
  const {ctx,width,height}=result;ctx.clearRect(0,0,width,height);ctx.fillStyle='#091320';ctx.fillRect(0,0,width,height);
  const pad={left:36,right:10,top:12,bottom:20},plotW=width-pad.left-pad.right,plotH=height-pad.top-pad.bottom;
  ctx.font='8px Consolas';ctx.fillStyle='#74859c';ctx.strokeStyle='#26364a';
  [0,.5,1].forEach(value=>{const y=pad.top+(1-value)*plotH;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(width-pad.right,y);ctx.stroke();ctx.fillText(`${Math.round(value*100)}%`,4,y+3);});
  const valid=predictions.filter(item=>Number.isFinite(Number(item.stress_probability)));if(!valid.length){ctx.fillStyle='#77889f';ctx.font='10px Segoe UI';ctx.fillText('Trend appears after the first complete window',pad.left+8,height/2);return;}
  const maxTime=Math.max(4,Number(valid[valid.length-1].elapsed_seconds)||valid.length*2);ctx.strokeStyle='#f2ad7a';ctx.lineWidth=2;ctx.beginPath();
  valid.forEach((item,index)=>{const x=pad.left+(Number(item.elapsed_seconds)/maxTime)*plotW,y=pad.top+(1-Math.max(0,Math.min(1,Number(item.stress_probability))))*plotH;if(index===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
}

ui.refreshStreams.addEventListener('click', discoverStreams);
ui.refreshSessions.addEventListener('click', refreshSessions);
ui.stream.addEventListener('change', renderStreamInfo);
ui.participantId.addEventListener('input', updateStartState);
ui.participantName.addEventListener('input', updateStartState);
ui.start.addEventListener('click', startLive);
ui.stop.addEventListener('click', stopLive);
ui.report.addEventListener('click', async () => { if (finalReport) { try { await saveReport(finalReport); } catch (error) { setStatus(error.message,'error'); } } });
window.eegApp.onLiveEvent(handleLiveEvent);
window.eegApp.onLiveExit(result => {
  if (active && result.exitCode !== 0) {
    restoreControls();
    setStatus(result.detail || `Acquisition process exited with code ${result.exitCode}.`, 'error');
  }
});
window.addEventListener('resize', () => { drawEeg(); drawStress(); });
discoverStreams(); refreshSessions();
