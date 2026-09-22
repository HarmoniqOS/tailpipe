// Shared schema helpers — mirrors normalizers/normalize_{claude,chatgpt}_export.py
// Schema v1: conversation + messages with native IDs, parent pointers,
// role map, block vocabulary, content hashes, active-path flags, attribution.

export const SCHEMA_VERSION = 1;

export async function contentHash(role, text) {
  const data = new TextEncoder().encode(`${role}\x00${text}`);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)]
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 16);
}

export function flatText(content) {
  const parts = [];
  for (const b of content) {
    if ((b.type === 'text' || b.type === 'thinking')) {
      if (b.text) parts.push(b.text);
      if (b.summaries) parts.push(...b.summaries);
    } else if (b.type === 'tool_use') {
      parts.push(`[tool_use:${b.tool_name || ''}]`);
    } else if (b.type === 'tool_result') {
      parts.push(`[tool_result:${b.tool_name || ''}]`);
    }
  }
  return parts.filter(Boolean).join('\n');
}

// Attribution is never guessed at capture time. The memory core's ingestion
// pipeline (or a manual review) resolves "unknown" downstream. Conversations
// inside a project mapped to a known owner resolve deterministically.
export function makeAttribution(projectNativeId, projectOwnerMap) {
  if (projectNativeId && projectOwnerMap && projectOwnerMap[projectNativeId]) {
    return { owner: projectOwnerMap[projectNativeId], method: 'project-rule', confidence: 1.0 };
  }
  return { owner: 'unknown', method: null, confidence: 0.0 };
}

export function conversationRecord({ source, messages, attribution, extra = {} }) {
  return {
    schema_version: SCHEMA_VERSION,
    conversation: {
      source,
      native_id: extra.native_id,
      title: extra.title || '',
      summary: extra.summary || '',
      model: extra.model || null,
      project_native_id: extra.project_native_id || null,
      created_at: extra.created_at || null,
      updated_at: extra.updated_at || null,
      current_leaf_native_id: extra.current_leaf_native_id || null,
      attribution,
      sync: { ingestor: 'tailpipe-collector/0.1.0', captured_via: 'session-api', captured_at: new Date().toISOString() },
    },
    messages,
    stats: {
      message_count: messages.length,
      abandoned_messages: messages.filter(m => !m.on_active_path).length,
    },
  };
}

// Walk parent pointers from the leaf to mark the active path.
export function markActivePath(messages, leafId) {
  const byId = new Map(messages.map(m => [m.native_id, m]));
  const active = new Set();
  let cursor = leafId;
  while (cursor && byId.has(cursor)) {
    active.add(cursor);
    cursor = byId.get(cursor).parent_native_id;
  }
  for (const m of messages) m.on_active_path = active.has(m.native_id);
}
