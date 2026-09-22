// ChatGPT provider — session-API capture.
// Token handshake via /api/auth/session, then /backend-api with Bearer.
// Normalization mirrors normalizers/normalize_chatgpt_export.py.

import { contentHash, flatText, markActivePath, makeAttribution, conversationRecord } from '../lib/schema.js';

const BASE = 'https://chatgpt.com';

function iso(ts) {
  return ts == null ? null : new Date(ts * 1000).toISOString();
}

async function getToken() {
  const resp = await fetch(`${BASE}/api/auth/session`, { credentials: 'include' });
  if (!resp.ok) throw new Error(`chatgpt session -> ${resp.status}`);
  const sess = await resp.json();
  if (!sess.accessToken) throw new Error('no access token — not logged in to chatgpt.com?');
  return sess.accessToken;
}

async function api(path, token) {
  const resp = await fetch(`${BASE}${path}`, {
    credentials: 'include',
    headers: { Authorization: `Bearer ${token}`, accept: 'application/json' },
  });
  if (resp.status === 401 || resp.status === 403) throw new Error(`auth (${resp.status}) on ${path}`);
  if (!resp.ok) throw new Error(`chatgpt ${path} -> ${resp.status}`);
  return resp.json();
}

function normalizeContent(content) {
  const ctype = content?.content_type || 'unknown';

  if (ctype === 'text') {
    return [{ type: 'text', text: (content.parts || []).filter(p => typeof p === 'string').join('\n') }];
  }
  if (ctype === 'thoughts') {
    // Reasoning summaries — ChatGPT's analog of Claude's thinking summaries.
    const summaries = [], texts = [];
    for (const t of content.thoughts || []) {
      if (t && typeof t === 'object') {
        if (t.summary) summaries.push(t.summary);
        if (t.content) texts.push(t.content);
      }
    }
    const block = { type: 'thinking', text: texts.join('\n') };
    if (summaries.length) block.summaries = summaries;
    return [block];
  }
  if (ctype === 'reasoning_recap') {
    return [{ type: 'thinking', text: '', summaries: [content.content || ''] }];
  }
  if (ctype === 'code') {
    return [{
      type: 'tool_use',
      tool_name: content.response_format_name || `code:${content.language || ''}`,
      input: { language: content.language, code: content.text || '' },
    }];
  }
  if (ctype === 'execution_output') {
    return [{ type: 'tool_result', tool_name: 'code', content: [{ type: 'text', text: content.text || '' }], is_error: false }];
  }
  if (ctype === 'multimodal_text') {
    return (content.parts || []).map(p => {
      if (typeof p === 'string') return { type: 'text', text: p };
      // Voice transcripts index as text (modality preserved)
      if (p?.content_type === 'audio_transcription' && p.text)
        return { type: 'text', text: p.text, modality: 'voice' };
      return { type: p?.content_type || 'asset', raw: Object.fromEntries(Object.entries(p || {}).filter(([, v]) => v != null)) };
    });
  }
  return [{ type: ctype, raw: Object.fromEntries(Object.entries(content || {}).filter(([k, v]) => k !== 'content_type' && v != null)) }];
}

async function normalize(raw, projectOwnerMap) {
  const messages = [];
  for (const [nodeId, node] of Object.entries(raw.mapping || {})) {
    const msg = node.message;
    if (!msg) continue; // synthetic root
    const role = msg.author?.role || 'unknown';
    const content = normalizeContent(msg.content || {});
    messages.push({
      native_id: nodeId,
      parent_native_id: node.parent || null,
      role,
      created_at: iso(msg.create_time),
      updated_at: iso(msg.update_time),
      content,
      content_hash: await contentHash(role, flatText(content)),
      model: msg.metadata?.model_slug || null,
      hidden: !!msg.metadata?.is_visually_hidden_from_conversation,
    });
  }
  markActivePath(messages, raw.current_node);

  // ChatGPT Projects surface as conversation_template_id (g-p-*); gizmo_id
  // covers custom GPTs. Prefer the project for attribution.
  const projectId = raw.conversation_template_id || raw.gizmo_id;
  return conversationRecord({
    source: 'chatgpt.com',
    messages,
    attribution: makeAttribution(projectId, projectOwnerMap),
    extra: {
      native_id: raw.conversation_id,
      title: raw.title,
      model: raw.default_model_slug,
      project_native_id: projectId,
      created_at: iso(raw.create_time),
      updated_at: iso(raw.update_time),
      current_leaf_native_id: raw.current_node,
    },
  });
}

export async function sync({ cursor, projectOwnerMap, onRecord, maxConversations = 25 }) {
  const token = await getToken();
  const list = await api(`/backend-api/conversations?offset=0&limit=${maxConversations}`, token);
  const items = list.items || [];
  const changed = items.filter(c => !cursor || c.update_time > cursor);

  let captured = 0;
  let newCursor = cursor;
  for (const meta of changed) {
    const raw = await api(`/backend-api/conversation/${meta.id}`, token);
    raw.conversation_id = raw.conversation_id || meta.id;
    await onRecord(await normalize(raw, projectOwnerMap));
    captured++;
    if (!newCursor || meta.update_time > newCursor) newCursor = meta.update_time;
    await new Promise(r => setTimeout(r, 800));
  }
  return { captured, listed: items.length, cursor: newCursor };
}
