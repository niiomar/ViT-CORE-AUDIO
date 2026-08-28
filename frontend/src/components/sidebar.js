export function renderSidebar() {
  return `
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">ViT</div>
        <div>
          <h1>ViT-CORE-Audio</h1>
          <p>Anti-Spoofing Workspace</p>
        </div>
      </div>

      <div class="section-heading">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
        Telemetry Overview
      </div>
      <div class="session-stats">
        <div class="stat-box"><span>Scans</span><strong id="stat-total">0</strong></div>
        <div class="stat-box"><span>Bonafide</span><strong id="stat-real-count" class="stat-real">0</strong></div>
        <div class="stat-box"><span>Spoof</span><strong id="stat-fake-count" class="stat-fake">0</strong></div>
      </div>

      <div class="section-heading">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
        Evidence Input
      </div>
      <div class="evidence-locker" id="drop-zone">
        <svg class="locker-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke-width="1.5">
          <path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle>
        </svg>
        <h2>Load Audio Evidence</h2>
        <p>WAV, FLAC, MP3, M4A, OGG &mdash; select or drop multiple files to batch analyze</p>
        <input type="file" id="file-input" accept="audio/*,.wav,.flac,.mp3,.m4a,.ogg,.aac,.wma" multiple />
      </div>

      <div class="section-heading">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        Model Parameters
      </div>
      <div class="options-panel">
        <div class="toggle-row">
          <label class="toggle"><input type="checkbox" id="explain-toggle" checked><span class="toggle-slider"></span></label>
          <span>Mel / CQT Views</span>
        </div>
      </div>

      <button id="analyze-btn" class="action-btn" disabled>AWAITING EVIDENCE</button>

      <div class="history-header">
        <div style="display: flex; align-items: center; gap: 8px;">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          SESSION LOG
        </div>
        <button class="clear-history" id="clear-history-btn">CLEAR</button>
      </div>

      <div class="history-controls">
        <input type="text" id="history-search" class="history-search" placeholder="Search filenames..." autocomplete="off"/>
        <div class="history-filters">
            <button class="filter-chip active" data-filter="ALL">ALL</button>
            <button class="filter-chip" data-filter="BONAFIDE">BONAFIDE</button>
            <button class="filter-chip" data-filter="SPOOF">SPOOF</button>
        </div>
      </div>

      <div class="history-list" id="history-list"></div>
    </aside>
  `;
}
