// Tailpipe Collector — background orchestrator.
// Alarm-driven incremental sync per provider; captured records queue in
// IndexedDB and ship to the configured ingest endpoint when reachable.

import * as claude from './providers/claude.js';
import * as chatgpt from './providers/chatgpt.js';
import { upsertConversation, getQueued, markShipped, getCursor, setCursor, logSync, counts, getSyncLog } from './lib/store.js';

const PROVIDERS = { 'claude.ai': claude, 'chatgpt.com': chatgpt };
const SYNC_ALARM = 'tailpipe-sync';
const SYNC_PERIOD_MIN = 30;

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(SYNC_ALARM, { periodInMinutes: SYNC_PERIOD_MIN, delayInMinutes: 1 });
});

chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === SYNC_ALARM) syncAll('alarm');
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.action === 'SYNC_NOW') {
    syncAll('manual').then(sendResponse);
    return true;
  }
  if (msg.action === 'STATUS') {
    status().then(sendResponse);
    return true;
  }
  if (msg.action === 'SHIP_NOW') {
    ship().then(sendResponse);
    return true;
  }
});

async function getSettings() {
  const defaults = {
    enabled: { 'claude.ai': true, 'chatgpt.com': true },
    ingestUrl: 'http://localhost:8080/ingest', // configurable via Settings page
    ingestToken: '',
    projectOwnerMap: {},  // project/gizmo native id -> owner name for deterministic attribution
  };
  const stored = await chrome.storage.local.get('settings');
  return { ...defaults, ...(stored.settings || {}) };
}

async function syncAll(trigger) {
  const settings = await getSettings();
  const results = {};

  for (const [name, provider] of Object.entries(PROVIDERS)) {
    if (!settings.enabled[name]) { results[name] = { skipped: true }; continue; }
    try {
      const cursor = await getCursor(name);
      const res = await provider.sync({
        cursor,
        projectOwnerMap: settings.projectOwnerMap,
        onRecord: upsertConversation,
      });
      if (res.cursor) await setCursor(name, res.cursor);
      await logSync(name, { ok: true, trigger, captured: res.captured });
      results[name] = res;
    } catch (e) {
      // Failure = staleness, never data loss. Surface in status, retry next alarm.
      await logSync(name, { ok: false, trigger, error: String(e.message || e) });
      results[name] = { error: String(e.message || e) };
    }
  }

  await ship(); // opportunistic delivery after every sync
  updateBadge();
  return results;
}

async function ship() {
  const settings = await getSettings();
  if (!settings.ingestUrl) return { shipped: 0, reason: 'no ingest endpoint configured' };

  const queued = await getQueued(20);
  if (!queued.length) return { shipped: 0 };

  const delivered = [];
  for (const item of queued) {
    try {
      const resp = await fetch(settings.ingestUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(settings.ingestToken ? { authorization: `Bearer ${settings.ingestToken}` } : {}),
        },
        body: JSON.stringify(item.record),
      });
      if (!resp.ok) break; // endpoint unhappy — stop, retry next cycle
      delivered.push(item.key);
    } catch {
      break; // server unreachable — records stay queued
    }
  }
  if (delivered.length) await markShipped(delivered);
  updateBadge();
  return { shipped: delivered.length, remaining: queued.length - delivered.length };
}

async function status() {
  return {
    counts: await counts(),
    lastSync: await getSyncLog(),
    settings: await getSettings(),
  };
}

async function updateBadge() {
  const c = await counts();
  chrome.action.setBadgeText({ text: c.queued ? String(c.queued) : '' });
  chrome.action.setBadgeBackgroundColor({ color: '#4a7c59' });
}
