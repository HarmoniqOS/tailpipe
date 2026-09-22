const $ = id => document.getElementById(id);

async function load() {
  const stored = await chrome.storage.local.get('settings');
  const s = stored.settings || {};
  $('en-claude').checked = s.enabled?.['claude.ai'] ?? true;
  $('en-chatgpt').checked = s.enabled?.['chatgpt.com'] ?? true;
  $('url').value = s.ingestUrl || '';
  $('token').value = s.ingestToken || '';
  $('owners').value = JSON.stringify(s.projectOwnerMap || {}, null, 2);
}

$('save').addEventListener('click', async () => {
  let projectOwnerMap = {};
  try {
    projectOwnerMap = JSON.parse($('owners').value || '{}');
  } catch {
    $('saved').textContent = 'owner map is not valid JSON';
    return;
  }
  await chrome.storage.local.set({
    settings: {
      enabled: { 'claude.ai': $('en-claude').checked, 'chatgpt.com': $('en-chatgpt').checked },
      ingestUrl: $('url').value.trim(),
      ingestToken: $('token').value.trim(),
      projectOwnerMap,
    },
  });
  $('saved').textContent = 'saved';
  setTimeout(() => ($('saved').textContent = ''), 2000);
});

load();
