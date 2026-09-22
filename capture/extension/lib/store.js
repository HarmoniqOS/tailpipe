// IndexedDB storage: captured conversations (delivery queue for the ingest
// server), sync cursors per provider, and sync log. Conversations are keyed
// by `${source}:${native_id}` — re-capture of an updated conversation upserts.

const DB_NAME = 'tailpipe-collector';
const DB_VERSION = 1;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('conversations')) {
        const s = db.createObjectStore('conversations', { keyPath: 'key' });
        s.createIndex('by_source', 'source');
        s.createIndex('by_shipped', 'shipped');
      }
      if (!db.objectStoreNames.contains('meta')) {
        db.createObjectStore('meta', { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const result = fn(t.objectStore(store));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
  });
}

export async function upsertConversation(record) {
  const db = await openDb();
  const conv = record.conversation;
  const key = `${conv.source}:${conv.native_id}`;
  await tx(db, 'conversations', 'readwrite', s => s.put({
    key,
    source: conv.source,
    native_id: conv.native_id,
    title: conv.title,
    updated_at: conv.updated_at,
    shipped: 0,       // 0 = queued for delivery, 1 = delivered
    record,
  }));
  db.close();
  return key;
}

export async function getQueued(limit = 50) {
  const db = await openDb();
  const items = await new Promise((resolve, reject) => {
    const out = [];
    const t = db.transaction('conversations', 'readonly');
    const idx = t.objectStore('conversations').index('by_shipped');
    const cur = idx.openCursor(IDBKeyRange.only(0));
    cur.onsuccess = () => {
      const c = cur.result;
      if (c && out.length < limit) { out.push(c.value); c.continue(); }
      else resolve(out);
    };
    cur.onerror = () => reject(cur.error);
  });
  db.close();
  return items;
}

export async function markShipped(keys) {
  const db = await openDb();
  await tx(db, 'conversations', 'readwrite', s => {
    for (const key of keys) {
      const get = s.get(key);
      get.onsuccess = () => {
        const v = get.result;
        if (v) { v.shipped = 1; s.put(v); }
      };
    }
  });
  db.close();
}

export async function counts() {
  const db = await openDb();
  const all = await tx(db, 'conversations', 'readonly', s => s.count());
  const queued = await new Promise((resolve, reject) => {
    const t = db.transaction('conversations', 'readonly');
    const req = t.objectStore('conversations').index('by_shipped').count(IDBKeyRange.only(0));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return { total: all, queued };
}

export async function getAllRecords() {
  const db = await openDb();
  const items = await tx(db, 'conversations', 'readonly', s => s.getAll());
  db.close();
  return items;
}

export async function getCursor(provider) {
  const db = await openDb();
  const v = await tx(db, 'meta', 'readonly', s => s.get(`cursor:${provider}`));
  db.close();
  return v?.value || null;
}

export async function setCursor(provider, value) {
  const db = await openDb();
  await tx(db, 'meta', 'readwrite', s => s.put({ key: `cursor:${provider}`, value }));
  db.close();
}

export async function logSync(provider, info) {
  const db = await openDb();
  await tx(db, 'meta', 'readwrite', s => s.put({
    key: `lastsync:${provider}`,
    value: { at: new Date().toISOString(), ...info },
  }));
  db.close();
}

export async function getSyncLog() {
  const db = await openDb();
  const all = await tx(db, 'meta', 'readonly', s => s.getAll());
  db.close();
  return Object.fromEntries(
    all.filter(e => e.key.startsWith('lastsync:')).map(e => [e.key.slice(9), e.value])
  );
}
