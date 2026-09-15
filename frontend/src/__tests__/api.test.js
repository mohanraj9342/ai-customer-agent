import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getApiBaseUrl,
  sendChatMessage,
  checkBackendHealth,
  checkBackendReadiness,
  ApiError,
} from '../services/api';

describe('API Service (frontend/src/services/api.js)', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('resolves default backend URL when VITE_API_URL is not provided', () => {
    const url = getApiBaseUrl();
    expect(url).toBe('http://127.0.0.1:8000');
  });

  it('rejects client-side when message is empty or whitespace', async () => {
    await expect(sendChatMessage({ message: '' })).rejects.toThrow(ApiError);
    await expect(sendChatMessage({ message: '   \n  ' })).rejects.toThrow(
      'Message cannot be empty or solely whitespace.'
    );
  });

  it('sends POST /api/chat with valid payload and returns response data', async () => {
    const mockSuccessResponse = {
      status: 'success',
      data: {
        response: {
          customer_message: 'iPhone battery dies fast',
          predicted_intent: 'battery_power',
          intent_confidence: 0.96,
          draft_response: 'Check Settings > Battery.',
          should_escalate: false,
          model_used: 'qwen/qwen3.8-27b',
          retrieved_evidence: [],
          cited_evidence_ids: [],
        },
        review: {
          decision: 'PASS',
          overall_score: 0.95,
          dimension_scores: { grounding_support: 0.9 },
        },
        execution_time_ms: 85.2,
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockSuccessResponse,
    });

    const result = await sendChatMessage({
      message: 'iPhone battery dies fast',
      top_k_evidence: 3,
      review_mode: 'deterministic',
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [callUrl, callInit] = global.fetch.mock.calls[0];
    expect(callUrl).toContain('/api/chat');
    expect(callInit.method).toBe('POST');
    expect(JSON.parse(callInit.body)).toEqual({
      message: 'iPhone battery dies fast',
      top_k_evidence: 3,
      include_review: true,
      review_mode: 'deterministic',
    });

    expect(result.response.predicted_intent).toBe('battery_power');
    expect(result.review.decision).toBe('PASS');
  });

  it('parses RFC-compliant error responses on HTTP 422 validation failure', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        status: 'error',
        code: 'VALIDATION_ERROR',
        message: 'Request validation failed.',
        details: [{ location: 'body -> message', issue: 'String should have at least 1 character' }],
      }),
    });

    await expect(sendChatMessage({ message: 'Test message' })).rejects.toMatchObject({
      name: 'ApiError',
      status: 422,
      code: 'VALIDATION_ERROR',
      message: 'Request validation failed.',
    });
  });

  it('maps HTTP 503 upstream backend errors correctly', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        status: 'error',
        code: 'SERVICE_UNAVAILABLE',
        message: 'AI inference service is temporarily unavailable.',
      }),
    });

    await expect(sendChatMessage({ message: 'Hello' })).rejects.toMatchObject({
      status: 503,
      code: 'SERVICE_UNAVAILABLE',
    });
  });

  it('handles network offline failures gracefully', async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(sendChatMessage({ message: 'Hello' })).rejects.toMatchObject({
      code: 'NETWORK_ERROR',
    });
  });

  it('checks backend health and readiness successfully', async () => {
    global.fetch = vi.fn().mockImplementation((url) => {
      if (url.includes('/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ status: 'ok', service: 'apple-support-ai-agent' }),
        });
      }
      if (url.includes('/ready')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ status: 'ready', retriever_ready: true }),
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    const health = await checkBackendHealth();
    expect(health.status).toBe('ok');

    const ready = await checkBackendReadiness();
    expect(ready.status).toBe('ready');
  });
});
