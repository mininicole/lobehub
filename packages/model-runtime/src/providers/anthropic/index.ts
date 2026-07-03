import { ModelProvider } from 'model-bank';

import {
  buildDefaultAnthropicPayload,
  createAnthropicCompatibleParams,
  createAnthropicCompatibleRuntime,
} from '../../core/anthropicCompatibleFactory';
import type { ChatStreamPayload } from '../../types';
import { normalizeClaudeThinkingHistoryMessages } from './claudeThinkingHistory';

/**
 * Upgrades every ephemeral cache_control block to the 1h TTL tier.
 * buildDefaultAnthropicPayload always emits `{ type: 'ephemeral' }` (Anthropic's
 * implicit 5m default) at the three injection sites it shares with other
 * Anthropic-wire providers (Moonshot, DeepSeek). Mutating those shared builders
 * would silently push extended-cache-ttl onto connectors that never asked for
 * it, so the upgrade happens here instead, scoped to just this provider.
 */
const applyExtendedCacheTTL = <T extends { cache_control?: { ttl?: string; type: string } }>(
  block: T,
): T => {
  if (block.cache_control?.type !== 'ephemeral' || block.cache_control.ttl) return block;

  return { ...block, cache_control: { ...block.cache_control, ttl: '1h' } };
};

const withExtendedCacheTTL = (
  payload: Awaited<ReturnType<typeof buildDefaultAnthropicPayload>>,
): typeof payload => {
  if (Array.isArray(payload.system)) {
    payload.system = payload.system.map((block) => applyExtendedCacheTTL(block));
  }

  const lastMessage = payload.messages?.at(-1);
  if (lastMessage && Array.isArray(lastMessage.content)) {
    lastMessage.content = lastMessage.content.map((block) =>
      'cache_control' in block ? applyExtendedCacheTTL(block as any) : block,
    ) as typeof lastMessage.content;
  }

  if (Array.isArray(payload.tools)) {
    payload.tools = payload.tools.map((tool) => applyExtendedCacheTTL(tool as any)) as typeof payload.tools;
  }

  return payload;
};

const buildAnthropicPayload = async (payload: ChatStreamPayload) => {
  const basePayload = await buildDefaultAnthropicPayload({
    ...payload,
    messages: normalizeClaudeThinkingHistoryMessages(payload.messages),
  });

  return withExtendedCacheTTL(basePayload);
};

export const params = createAnthropicCompatibleParams({
  chatCompletion: {
    handlePayload: buildAnthropicPayload,
  },
  constructorOptions: {
    defaultHeaders: {
      // 1h prompt caching is a beta feature; Anthropic requires this header
      // on every request that sets cache_control.ttl to '1h'. If
      // ANTHROPIC_BETA_HEADERS is ever set in env, it fully replaces this
      // default (see createDefaultAnthropicClient) — remember to include
      // extended-cache-ttl-2025-04-11 in that comma-separated value too.
      'anthropic-beta': 'extended-cache-ttl-2025-04-11',
    },
  },
  debug: {
    chatCompletion: () => process.env.DEBUG_ANTHROPIC_CHAT_COMPLETION === '1',
  },
  provider: ModelProvider.Anthropic,
});

export const LobeAnthropicAI = createAnthropicCompatibleRuntime(params);

export default LobeAnthropicAI;
