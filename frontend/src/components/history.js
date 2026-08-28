export function renderHistoryItem(item) {
  const isSpoof = item.verdict === 'SPOOF';
  const cls = isSpoof ? 'fake' : 'real';

  return `
    <div class="hist-item ${cls}" data-hash="${item.file_sha256 || ''}">
      <div class="hi-top">
        <span class="hist-badge ${cls}">${item.verdict}</span>
        <span class="hi-conf">${item.confidence}%</span>
      </div>
      <div class="hi-bot">
        <span class="hi-name" title="${item.filename}">${item.filename}</span>
        <span class="hi-time">${new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
      </div>
    </div>
  `;
}

// Accepts the subset to render, but calculates global stats off the full history array
export function updateHistory(filteredHistory, fullHistory = []) {
  const list = document.getElementById('history-list');
  if (!list) return;

  list.innerHTML = '';

  if (filteredHistory.length === 0) {
      list.innerHTML = `<p style="text-align:center;font-size:10px;color:var(--text-dim);font-family:var(--mono);margin-top:20px;">NO LOGS MATCH QUERY</p>`;
  } else {
      filteredHistory.forEach(item => {
        list.innerHTML += renderHistoryItem(item);
      });
  }

  let spoofs = 0; let bonafides = 0;
  const statSource = fullHistory.length > 0 ? fullHistory : filteredHistory;

  statSource.forEach(item => {
    item.verdict === 'SPOOF' ? spoofs++ : bonafides++;
  });

  const totEl = document.getElementById('stat-total');
  const realEl = document.getElementById('stat-real-count');
  const fakeEl = document.getElementById('stat-fake-count');

  if(totEl) totEl.textContent = statSource.length;
  if(realEl) realEl.textContent = bonafides;
  if(fakeEl) fakeEl.textContent = spoofs;
}
