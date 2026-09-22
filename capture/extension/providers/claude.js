// Claude.ai provider — session-API capture.
// Endpoints ride the browser's own cookies via host_permissions; no stored
// credentials. Normalization mirrors normalizers/normalize_claude_export.py.

import { contentHash, flatText, markActivePath, makeAttribution, conversationRecord } from '../lib/schema.js';

const BASE = 'https://claude.ai';
const ROLE_MAP = { human: 'user', assistant: 'assistant' };

async function api(path) {
  const resp = await fetch(`${BASE}${path}`, {
    credentials: 'include',
    headers: { accept: 'application/json' },
  });
  if (resp.status === 401 || resp.status === 403) throw new Error(`auth (${resp.status}) — not logged in to claude.ai?`);
  if (!resp.ok) throw new Error(`claude.ai ${path} -> ${resp.status}`);
  return resp.json();
}

function normalizeBlock(block) {
  const t = block.type || 'unknown';
  const out = { type: t };
  if (t === 'text') {
    out.text = block.text || '';
    if (block.citations?.length) out.citations = block.citations;
  } else if (t === 'thinking') {
    out.text = block.thinking || block.text || '';
    const summaries = (block.summaries || [])
      .map(s => (typeof s === 'object' && s !== null ? s.summary ?? '' : s))
      .filter(Boolean);
    if (summaries.length) out.summaries = summaries;
    if (block.thinking_hidden) out.redacted = true;
  } else if (t === 'tool_use') {
    out.tool_name = block.name || '';
    out.input = block.input || {};
  } else if (t === 'tool_result') {
    out.tool_name = block.name || '';
    out.content = block.content || [];
    out.is_error = !!block.is_error;
  } else {
    out.raw = Object.fromEntries(Object.entries(block).filter(([k, v]) => k !== 'type' && v != null));
  }
  return out;
}

async function normalize(raw, projectOwnerMap) {
  const messages = [];
  for (const m of raw.chat_messages || []) {
    const content = (m.content || []).map(normalizeBlock);
    const role = ROLE_MAP[m.sender] || m.sender || 'unknown';
    messages.push({
      native_id: m.uuid,
      parent_native_id: m.parent_message_uuid || null,
      role,
      created_at: m.created_at || null,
      updated_at: m.updated_at || null,
      content,
      content_hash: await contentHash(role, flatText(content)),
      truncated: !!m.truncated,
      // Full objects — attachments can carry extracted_content (pasted file text)
      attachments: m.attachments || [],
      files: m.files || [],
    });
  }
  markActivePath(messages, raw.current_leaf_message_uuid);

  return conversationRecord({
    source: 'claude.ai',
    messages,
    attribution: makeAttribution(raw.project_uuid, projectOwnerMap),
    extra: {
      native_id: raw.uuid,
      title: raw.name,
      summary: raw.summary,
      model: raw.model,
      project_native_id: raw.project_uuid,
      created_at: raw.created_at,
      updated_at: raw.updated_at,
      current_leaf_native_id: raw.current_leaf_message_uuid,
    },
  });
}

// Incremental sync: list newest-first, capture everything newer than the
// cursor, return the new cursor (max updated_at seen).
export async function sync({ cursor, projectOwnerMap, onRecord, maxConversations = 25 }) {
  const orgs = await api('/api/organizations');
  const org = orgs[0]?.uuid;
  if (!org) throw new Error('no organization found');

  const list = await api(`/api/organizations/${org}/chat_conversations?limit=${maxConversations}`);
  const changed = list.filter(c => !cursor || c.updated_at > cursor);

  let captured = 0;
  let newCursor = cursor;
  for (const meta of changed) {
    const raw = await api(
      `/api/organizations/${org}/chat_conversations/${meta.uuid}?tree=True&rendering_mode=messages&render_all_tools=true`
    );
    await onRecord(await normalize(raw, projectOwnerMap));
    captured++;
    if (!newCursor || meta.updated_at > newCursor) newCursor = meta.updated_at;
    await new Promise(r => setTimeout(r, 800)); // gentle pacing — own data, no hammering
  }
  return { captured, listed: list.length, cursor: newCursor };
}
