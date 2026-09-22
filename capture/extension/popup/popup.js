import { getAllRecords } from '../lib/store.js';

const $ = id => document.getElementById(id);

async function refresh() {
  const status = await chrome.runtime.sendMessage({ action: 'STATUS' });
  $('total').textContent = status.counts.total;
  $('queued').textContent = status.counts.queued;

  const lines = [];
  for (const [provider, info] of Object.entries(status.lastSync || {})) {
    if (info.ok) {
      lines.push(`<div class="row"><span>${provider}</span><span class="ok">${info.captured} captured · ${new Date(info.at).toLocaleTimeString()}</span></div>`);
    } else {
      lines.push(`<div class="row"><span>${provider}</span><span class="err" title="${info.error}">stale — ${info.error?.slice(0, 40)}</span></div>`);
    }
  }
  $('log').innerHTML = lines.join('') || '<div class="muted">No syncs yet — hit Sync now.</div>';
}

$('sync').addEventListener('click', async () => {
  $('sync').textContent = 'Syncing…';
  $('sync').disabled = true;
  await chrome.runtime.sendMessage({ action: 'SYNC_NOW' });
  $('sync').textContent = 'Sync now';
  $('sync').disabled = false;
  refresh();
});

$('export').addEventListener('click', async () => {
  const items = await getAllRecords();
  const ndjson = items.map(i => JSON.stringify(i.record)).join('\n');
  const blob = new Blob([ndjson], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `tailpipe-export-${new Date().toISOString().slice(0, 10)}.ndjson`;
  a.click();
  URL.revokeObjectURL(url);
});

$('options').addEventListener('click', () => chrome.runtime.openOptionsPage());

refresh();
