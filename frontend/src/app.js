import './styles.css';
import { renderSidebar } from './components/sidebar.js';
import { renderWorkspace } from './components/workspace.js';
import { updateHistory } from './components/history.js';
import { executeAudioAnalysis, executeBatchAnalysis, fetchHistory } from './utils/api.js';
import { compilePdfReport } from './utils/report.js';

document.getElementById('app').innerHTML = `
  <div class="layout">
    ${renderSidebar()}
    ${renderWorkspace()}
  </div>
`;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const previewWrapper = document.getElementById('preview-wrapper');

// Audio player + the two spectral-view tabs
const audioPreview = document.getElementById('audio-preview');
const melImg = document.getElementById('mel-img');
const melPlaceholder = document.getElementById('mel-placeholder');
const cqtImg = document.getElementById('cqt-img');
const cqtPlaceholder = document.getElementById('cqt-placeholder');

// State views and dynamic data containers
const idleState = document.getElementById('idle-state');
const resultState = document.getElementById('result-state');
const batchState = document.getElementById('batch-state');
const batchSummary = document.getElementById('batch-summary');
const batchList = document.getElementById('batch-list');
const gaugeFill = document.getElementById('gauge-fill');
const historyList = document.getElementById('history-list');

let selectedFile = null;
let isBatchMode = false;
let selectedBatchFiles = [];
let currentReport = null;
let sessionHistory = [];
let loadingInterval = null;
let objectUrlCache = null;

let activeFilter = 'ALL';
let searchQuery = '';

// Hydrates the session history from the backend audit ledger on initial
// page load. Relative /api paths work in both dev (proxied by
// vite.config.js) and prod (same origin, served by FastAPI) — no separate
// base URL needed.
async function syncDatabaseHistory() {
  try {
    const data = await fetchHistory();
    sessionHistory = data.entries.reverse();
    applyHistoryFilters();
  } catch (err) {
    console.error("Database sync failed:", err);
  }
}
syncDatabaseHistory();

function handleThrottled() {
  document.getElementById('warn-sys-text').textContent =
    'Rate limit exceeded — please wait a moment before retrying.';
  document.getElementById('warn-sys-error').classList.add('visible');
  previewWrapper.classList.remove('scanning');
}

// Re-evaluates the history array based on the active verdict chip and
// search query.
function applyHistoryFilters() {
  let filtered = sessionHistory;

  if (activeFilter !== 'ALL') {
      filtered = filtered.filter(item => item.verdict === activeFilter);
  }

  if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(item => item.filename.toLowerCase().includes(q));
  }

  updateHistory(filtered, sessionHistory);
}

// Search and Filter Event Listeners
document.getElementById('history-search').addEventListener('input', (e) => {
    searchQuery = e.target.value;
    applyHistoryFilters();
});

document.querySelectorAll('.filter-chip').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-chip').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        activeFilter = e.target.dataset.filter;
        applyHistoryFilters();
    });
});

// File Ingestion Listeners
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));

// Drag and drop UX handling
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', e => { e.preventDefault(); dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});

// Routes to the existing single-file flow, or batch mode when more than
// one file is selected/dropped.
function handleFiles(fileList) {
  const files = Array.from(fileList || []);
  if (files.length === 0) return;

  if (files.length === 1) {
    isBatchMode = false;
    selectedBatchFiles = [];
    handleFile(files[0]);
    return;
  }

  isBatchMode = true;
  selectedBatchFiles = files;
  selectedFile = null;

  batchState.classList.remove('visible');
  resultState.classList.remove('visible');
  idleState.style.display = 'flex';
  document.querySelectorAll('.hist-item').forEach(el => el.classList.remove('active-log'));

  analyzeBtn.disabled = false;
  analyzeBtn.textContent = `ANALYZE BATCH (${files.length} FILES)`;
}

// Workspace Tab Navigation (Mel View / CQT View)
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-layer').forEach(l => l.classList.remove('active'));

    const targetId = e.target.dataset.target;
    e.target.classList.add('active');
    document.getElementById(targetId).classList.add('active');
  });
});

// Clicking an old log entry restores that exact analysis to the dashboard.
// Historical entries come from the audit DB (see fetchHistory), which only
// ever stores scalar fields — the source audio and the mel/CQT view images
// are never persisted, only generated per-request — so a restored entry
// shows no player and no spectrogram images unless it's still the same
// in-session object that came straight back from /api/v1/analyze.
historyList.addEventListener('click', (e) => {
  const item = e.target.closest('.hist-item');
  if (!item) return;

  const hash = item.dataset.hash;
  const entry = sessionHistory.find(x => x.file_sha256 === hash);

  if (entry) {
    document.querySelectorAll('.hist-item').forEach(el => el.classList.remove('active-log'));
    item.classList.add('active-log');

    // Free memory from the currently active file before loading historical data
    if (objectUrlCache) {
      URL.revokeObjectURL(objectUrlCache);
      objectUrlCache = null;
    }

    idleState.style.display = 'none';
    resultState.classList.add('visible');
    audioPreview.removeAttribute('src');
    audioPreview.load();

    renderResult(entry, entry.filename);
  }
});

// Sets up the local preview (object URL) and resets the result view before
// the file is sent for analysis.
function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = `ANALYZE: ${file.name.length > 20 ? file.name.slice(0,18)+'…' : file.name}`;

  if (objectUrlCache) URL.revokeObjectURL(objectUrlCache);
  objectUrlCache = URL.createObjectURL(file);
  audioPreview.src = objectUrlCache;

  melImg.style.display = 'none';
  cqtImg.style.display = 'none';
  melPlaceholder.style.display = 'block';
  cqtPlaceholder.style.display = 'block';

  idleState.style.display = 'flex';
  resultState.classList.remove('visible');
  batchState.classList.remove('visible');
  gaugeFill.style.strokeDashoffset = 326.7;

  document.querySelectorAll('.hist-item').forEach(el => el.classList.remove('active-log'));
  document.getElementById('stat-score-sub').textContent = 'Pending';
  document.getElementById('stat-face-sub').textContent = 'Pending';
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  if (!on) {
    clearInterval(loadingInterval);
    analyzeBtn.textContent = `ANALYZE: ${selectedFile.name.length > 20 ? selectedFile.name.slice(0,18)+'…' : selectedFile.name}`;
    return;
  }

  const steps = ['Loading Waveform...', 'Computing Mel View...', 'Computing CQT View...', 'Running ViT-S/16...'];
  let i = 0;
  analyzeBtn.textContent = steps[0];
  loadingInterval = setInterval(() => {
    i = (i + 1) % steps.length;
    analyzeBtn.textContent = steps[i];
  }, 400);
}

analyzeBtn.addEventListener('click', () => {
  if (isBatchMode) {
    runBatchAnalysis();
  } else {
    runSingleAnalysis();
  }
});

async function runSingleAnalysis() {
  idleState.style.display = 'none';
  resultState.classList.add('visible');
  batchState.classList.remove('visible');
  previewWrapper.classList.add('scanning');

  document.querySelector('[data-target="tab-mel"]').click();

  setLoading(true);
  gaugeFill.style.strokeDashoffset = 326.7;
  document.getElementById('low-conf-warning').style.display = 'none';
  document.getElementById('low-agreement-warning').style.display = 'none';
  document.getElementById('warn-sys-error').classList.remove('visible');

  const includeViews = document.getElementById('explain-toggle').checked;

  try {
    const data = await executeAudioAnalysis(selectedFile, includeViews, handleThrottled);
    if (!data) return; // request was rate-limited; handleThrottled already surfaced it

    previewWrapper.classList.remove('scanning');
    renderResult(data, selectedFile.name);

    sessionHistory.unshift({ timestamp: new Date().toISOString(), filename: selectedFile.name, ...data });
    applyHistoryFilters();

    // Automatically highlight the newest log entry
    setTimeout(() => {
        const firstLog = document.querySelector('.hist-item');
        if(firstLog) firstLog.classList.add('active-log');
    }, 100);

  } catch (err) {
    document.getElementById('warn-sys-text').textContent = err.message;
    document.getElementById('warn-sys-error').classList.add('visible');
    previewWrapper.classList.remove('scanning');
  } finally {
    setLoading(false);
  }
}

async function runBatchAnalysis() {
  idleState.style.display = 'none';
  resultState.classList.remove('visible');
  batchState.classList.add('visible');
  document.getElementById('warn-sys-error').classList.remove('visible');

  analyzeBtn.disabled = true;
  const fileCount = selectedBatchFiles.length;
  analyzeBtn.textContent = `Analyzing ${fileCount} files...`;

  try {
    const data = await executeBatchAnalysis(selectedBatchFiles, handleThrottled);
    if (!data) return; // rate-limited; handleThrottled already surfaced it

    renderBatchResult(data);

    const timestamp = new Date().toISOString();
    data.results.forEach(r => {
      if (!r.error) sessionHistory.unshift({ timestamp, ...r });
    });
    applyHistoryFilters();
  } catch (err) {
    document.getElementById('warn-sys-text').textContent = err.message;
    document.getElementById('warn-sys-error').classList.add('visible');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = `ANALYZE BATCH (${fileCount} FILES)`;
  }
}

function renderBatchResult(data) {
  const { summary, results } = data;

  batchSummary.innerHTML = `
    <span class="batch-stat">TOTAL<strong>${summary.total}</strong></span>
    <span class="batch-stat batch-real">BONAFIDE<strong>${summary.bonafide}</strong></span>
    <span class="batch-stat batch-fake">SPOOF<strong>${summary.spoof}</strong></span>
    <span class="batch-stat batch-error">ERRORS<strong>${summary.errors}</strong></span>
  `;

  batchList.innerHTML = results.map(r => {
    if (r.error) {
      return `<div class="batch-row batch-row-error">
        <span class="batch-row-name" title="${r.filename}">${r.filename}</span>
        <span class="batch-row-badge error">ERROR</span>
        <span class="batch-row-detail" title="${r.error}">${r.error}</span>
      </div>`;
    }
    const cls = r.verdict === 'SPOOF' ? 'fake' : 'real';
    return `<div class="batch-row">
      <span class="batch-row-name" title="${r.filename}">${r.filename}</span>
      <span class="batch-row-badge ${cls}">${r.verdict}</span>
      <span class="batch-row-detail">${r.confidence}% &middot; agreement ${r.view_agreement}</span>
    </div>`;
  }).join('');
}

// A cosine-similarity threshold below which the two spectral views are
// flagged as disagreeing enough to warrant manual review — a calibrated
// heuristic (not learned from data), the same spirit as FORENSICS'
// blur/brightness thresholds in its own quality gate.
const LOW_AGREEMENT_THRESHOLD = 0.5;

// Translates the backend analysis response into the visual DOM elements.
function renderResult(data, filename) {
  currentReport = { ...data, filename };
  const isSpoof = data.verdict === 'SPOOF';
  const cls = isSpoof ? 'fake' : 'real';

  document.getElementById('trust-title').textContent = data.verdict;
  document.getElementById('trust-title').className = `trust-title title-${cls}`;
  document.getElementById('gauge-conf').textContent = `${data.confidence}%`;

  gaugeFill.className.baseVal = `gauge-fill ${cls}`;
  setTimeout(() => { gaugeFill.style.strokeDashoffset = 326.7 - (326.7 * (data.confidence / 100)); }, 100);

  document.getElementById('stat-score').textContent = data.probability.toFixed(4);
  document.getElementById('stat-score').style.color = isSpoof ? 'var(--red)' : 'var(--green)';
  document.getElementById('stat-score-sub').textContent = `${(data.probability * 100).toFixed(1)}% spoof probability`;

  const agreementPct = (data.view_agreement * 100).toFixed(1);
  document.getElementById('stat-face').textContent = data.view_agreement.toFixed(4);
  document.getElementById('stat-face-sub').textContent =
    data.view_agreement >= 0.9 ? `${agreementPct}% — High agreement` :
    data.view_agreement >= LOW_AGREEMENT_THRESHOLD ? `${agreementPct}% — Moderate agreement` :
    `${agreementPct}% — Low agreement`;

  document.getElementById('kpi-format').textContent = (data.type || '').toUpperCase();
  document.getElementById('kpi-frames').textContent =
    typeof data.file_size_bytes === 'number' ? `${(data.file_size_bytes / 1024).toFixed(1)} KB` : 'N/A';
  document.getElementById('kpi-time').textContent = `${data.processing_time_sec}s`;

  if (data.is_low_confidence) document.getElementById('low-conf-warning').style.display = 'flex';
  if (typeof data.view_agreement === 'number' && data.view_agreement < LOW_AGREEMENT_THRESHOLD) {
    document.getElementById('low-agreement-warning').style.display = 'flex';
  }

  // Inject Base64 mel/CQT view images if they were requested and generated
  if (data.visuals && data.visuals.mel) {
      melImg.src = `data:image/jpeg;base64,${data.visuals.mel}`;
      melImg.style.display = 'block';
      melPlaceholder.style.display = 'none';
  } else {
      melImg.style.display = 'none';
      melPlaceholder.style.display = 'block';
  }

  if (data.visuals && data.visuals.cqt) {
      cqtImg.src = `data:image/jpeg;base64,${data.visuals.cqt}`;
      cqtImg.style.display = 'block';
      cqtPlaceholder.style.display = 'none';
  } else {
      cqtImg.style.display = 'none';
      cqtPlaceholder.style.display = 'block';
  }
}

document.getElementById('clear-history-btn').addEventListener('click', () => {
  if (sessionHistory.length === 0) return;
  if (confirm("Clear the current session history view? (Note: Database logs remain securely stored in the backend)")) {
    sessionHistory = [];
    applyHistoryFilters();
    idleState.style.display = 'flex';
    resultState.classList.remove('visible');
    batchState.classList.remove('visible');
    selectedFile = null;
    isBatchMode = false;
    selectedBatchFiles = [];
    analyzeBtn.textContent = 'AWAITING EVIDENCE';
    analyzeBtn.disabled = true;
    gaugeFill.style.strokeDashoffset = 326.7;
  }
});

// PDF Report Generation (jsPDF, via utils/report.js)
document.getElementById('export-btn').addEventListener('click', () => {
  if (!currentReport) return;
  compilePdfReport(currentReport);
});
