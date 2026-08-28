export function renderWorkspace() {
  return `
    <main class="main-view" id="main-view">

      <div id="idle-state">
        <div class="idle-shield">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
        </div>
        <p>AWAITING SPECTRAL TELEMETRY</p>
      </div>

      <div id="result-state">
        <div class="warning-banner warn-red" id="warn-sys-error">
          <svg class="banner-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
          <span id="warn-sys-text"></span>
        </div>
        <div class="warning-banner warn-amber" id="low-conf-warning">
          <svg class="banner-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
          AMBIGUOUS TELEMETRY — Model confidence is low. Manual review recommended.
        </div>
        <div class="warning-banner warn-purple" id="low-agreement-warning">
          <svg class="banner-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
          LOW VIEW AGREEMENT — The mel and CQT spectral views disagree; treat this verdict with extra caution.
        </div>

        <div class="executive-panel">
          <div class="exec-left">
            <div class="trust-ring-box">
              <svg class="gauge-svg" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="52" class="gauge-bg"></circle>
                <circle cx="60" cy="60" r="52" id="gauge-fill" class="gauge-fill"></circle>
              </svg>
              <div class="gauge-text">
                <span class="gauge-val" id="gauge-conf">0%</span>
                <span class="gauge-label">Conf</span>
              </div>
            </div>
            <div class="trust-info">
              <h2 class="trust-title" id="trust-title">UNKNOWN</h2>
              <p class="trust-sub">Model Verdict</p>
            </div>
          </div>

          <div class="exec-divider"></div>

          <div class="exec-right">
            <div class="tm-item">
              <span class="tm-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> Spoof Probability</span>
              <strong class="tm-val" id="stat-score">0.0000</strong>
              <span class="tm-sub" id="stat-score-sub">Pending</span>
            </div>
            <div class="tm-item">
              <span class="tm-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg> Mel/CQT Agreement</span>
              <strong class="tm-val" id="stat-face">N/A</strong>
              <span class="tm-sub" id="stat-face-sub">Pending</span>
            </div>
            <div class="tm-item">
              <span class="tm-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Analysis Window</span>
              <strong class="tm-val" id="stat-quality">4.0s</strong>
              <span class="tm-sub" id="stat-qual-sub">Fixed preprocessing window @ 16kHz</span>
            </div>
          </div>
        </div>

        <div class="kpi-strip">
          <div class="kpi-item">
            <svg class="kpi-chart" viewBox="0 0 60 40" fill="none" stroke="var(--text-mid)">
               <rect x="5" y="10" width="12" height="12" stroke-width="1.5" />
               <rect x="23" y="10" width="12" height="12" stroke-width="1.5" />
               <rect x="5" y="28" width="12" height="12" stroke-width="1.5" />
               <rect x="23" y="28" width="12" height="12" stroke-width="1.5" />
               <path d="M43 16 h12 M43 22 h7 M43 28 h12" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            <span class="kpi-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg> Format</span>
            <span class="kpi-val" id="kpi-format">N/A</span>
          </div>

          <div class="kpi-item">
            <svg class="kpi-chart" viewBox="0 0 60 40" fill="none">
               <path d="M5 30 Q 20 10, 35 25 T 65 15" stroke="var(--blue)" stroke-width="2" fill="none" />
               <circle cx="20" cy="18" r="3" fill="var(--blue)" />
               <circle cx="50" cy="20.5" r="3" fill="var(--blue)" />
            </svg>
            <span class="kpi-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg> File Size</span>
            <span class="kpi-val" id="kpi-frames">0 KB</span>
          </div>

          <div class="kpi-item">
            <svg class="kpi-chart" viewBox="0 0 60 40" fill="var(--text-mid)">
               <rect x="10" y="20" width="4" height="15" rx="1" />
               <rect x="20" y="12" width="4" height="23" rx="1" />
               <rect x="30" y="25" width="4" height="10" rx="1" />
               <rect x="40" y="8"  width="4" height="27" rx="1" />
               <rect x="50" y="18" width="4" height="17" rx="1" />
            </svg>
            <span class="kpi-label"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Compute Time</span>
            <span class="kpi-val" id="kpi-time">0s</span>
          </div>
        </div>

        <div class="media-panel" id="preview-wrapper">
          <div class="audio-player-wrap">
            <audio id="audio-preview" controls></audio>
          </div>
          <div class="tabs-header">
            <button class="tab-btn active" data-target="tab-mel">Mel View</button>
            <button class="tab-btn" data-target="tab-cqt">CQT View</button>
          </div>
          <div class="media-content">
            <div class="scan-line"></div>

            <div id="tab-mel" class="tab-layer active">
              <img id="mel-img" style="display:none;"/>
              <p id="mel-placeholder" style="font-family:var(--mono); color:var(--text-dim); font-size:11px; padding:40px;">MEL VIEW NOT GENERATED</p>
            </div>

            <div id="tab-cqt" class="tab-layer">
              <img id="cqt-img" style="display:none;"/>
              <p id="cqt-placeholder" style="font-family:var(--mono); color:var(--text-dim); font-size:11px; padding:40px;">CQT VIEW NOT GENERATED</p>
            </div>

          </div>
        </div>

        <div class="export-panel">
          <button id="export-btn" class="secondary-btn">Export Forensic Report (.PDF)</button>
        </div>

      </div>

      <div id="batch-state">
        <div class="batch-header">
          <h2>Batch Screening Results</h2>
          <div class="batch-summary" id="batch-summary"></div>
        </div>
        <div class="batch-list" id="batch-list"></div>
      </div>
    </main>
  `;
}
