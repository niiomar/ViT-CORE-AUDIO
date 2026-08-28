// API key is injected at BUILD TIME from the VITE_API_KEY environment
// variable (see frontend/.env.example). This is still visible in the
// shipped JS bundle — browser-side secrets are never truly secret — but
// it at least removes the hardcoded literal from source control and lets
// each deployment use its own key without code changes.
//
// For a real multi-user deployment, replace this with a login flow that
// exchanges user credentials for a short-lived session token instead.
const API_KEY = import.meta.env.VITE_API_KEY || '';

function authHeaders() {
  return API_KEY ? { 'X-API-KEY': API_KEY } : {};
}

export async function executeAudioAnalysis(file, includeViews, onThrottled) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`/api/v1/analyze?explain=${includeViews}`, {
    method: 'POST',
    body: formData,
    headers: authHeaders(),
  });

  if (response.status === 429) {
    onThrottled();
    return null;
  }

  if (response.status === 401) {
    throw new Error('Unauthorized — check your API key configuration.');
  }

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Critical pipeline analysis error.');
  }

  return await response.json();
}

export async function executeBatchAnalysis(files, onThrottled) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  const response = await fetch(`/api/v1/analyze/batch`, {
    method: 'POST',
    body: formData,
    headers: authHeaders(),
  });

  if (response.status === 429) {
    onThrottled();
    return null;
  }

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Batch analysis error.');
  }

  return await response.json();
}

export async function fetchHistory(limit = 50) {
  const response = await fetch(`/api/v1/history?limit=${limit}`, {
    headers: authHeaders(),
  });
  if (!response.ok) return { entries: [] };
  return await response.json();
}
