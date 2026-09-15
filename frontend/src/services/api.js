/**
 * frontend/src/services/api.js
 * =============================
 * Phase 16 — Client API Service for Render Backend Integration.
 *
 * Consumes Phase 15 FastAPI endpoints:
 * - GET  /health   -> Liveness probe
 * - GET  /ready    -> Pipeline readiness probe
 * - POST /api/chat -> Grounded customer support processing
 */

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

/**
 * Resolves the backend base URL without trailing slashes.
 * Reads strictly from VITE_API_URL environment variable.
 */
export function getApiBaseUrl() {
  const envUrl = import.meta.env?.VITE_API_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return DEFAULT_BACKEND_URL;
}

/**
 * Custom error class capturing HTTP status and backend error envelope data.
 */
export class ApiError extends Error {
  constructor(message, status = 500, code = 'API_ERROR', details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

/**
 * Performs GET /health to verify backend connectivity.
 */
export async function checkBackendHealth(timeoutMs = 5000) {
  const baseUrl = getApiBaseUrl();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) {
      throw new ApiError(`Health check failed (${res.status})`, res.status);
    }
    return await res.json();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new ApiError('Backend health check timed out.', 408, 'TIMEOUT');
    }
    throw new ApiError(
      err.message || 'Cannot reach backend service.',
      err.status || 0,
      'NETWORK_ERROR'
    );
  }
}

/**
 * Performs GET /ready to inspect pipeline asset loading.
 */
export async function checkBackendReadiness(timeoutMs = 8000) {
  const baseUrl = getApiBaseUrl();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/ready`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) {
      throw new ApiError(`Readiness probe failed (${res.status})`, res.status);
    }
    return await res.json();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new ApiError('Backend readiness probe timed out.', 408, 'TIMEOUT');
    }
    throw new ApiError(
      err.message || 'Cannot reach backend readiness probe.',
      err.status || 0,
      'NETWORK_ERROR'
    );
  }
}

/**
 * Sends customer message to POST /api/chat.
 *
 * @param {Object} options
 * @param {string} options.message - Customer query text (1-1000 chars)
 * @param {number} [options.top_k_evidence=3] - Max historical tweets to retrieve (1-10)
 * @param {boolean} [options.include_review=true] - Whether to execute reviewer audit
 * @param {string} [options.review_mode='deterministic'] - Reviewer execution mode
 * @param {number} [options.timeoutMs=35000] - Request abort timeout
 * @returns {Promise<{response: Object, review: Object, execution_time_ms: number}>}
 */
export async function sendChatMessage({
  message,
  top_k_evidence = 3,
  include_review = true,
  review_mode = 'deterministic',
  timeoutMs = 35000,
}) {
  const baseUrl = getApiBaseUrl();
  const trimmed = (message || '').trim();
  if (!trimmed) {
    throw new ApiError('Message cannot be empty or solely whitespace.', 422, 'VALIDATION_ERROR');
  }

  const payload = {
    message: trimmed,
    top_k_evidence: Number(top_k_evidence) || 3,
    include_review: Boolean(include_review),
    review_mode: review_mode || 'deterministic',
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timer);

    let data = null;
    try {
      data = await res.json();
    } catch {
      throw new ApiError(
        `Server returned non-JSON response (HTTP ${res.status}).`,
        res.status,
        'INVALID_RESPONSE'
      );
    }

    if (!res.ok) {
      // Parse RFC-compliant ErrorResponse
      const code = data?.code || `HTTP_${res.status}`;
      const msg = data?.message || `Request failed with HTTP status ${res.status}`;
      const details = data?.details || null;
      throw new ApiError(msg, res.status, code, details);
    }

    if (data.status !== 'success' || !data.data) {
      throw new ApiError(
        'Unexpected response format from chat endpoint.',
        res.status,
        'MALFORMED_RESPONSE'
      );
    }

    return data.data;
  } catch (err) {
    clearTimeout(timer);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err.name === 'AbortError') {
      throw new ApiError(
        'The request to the AI support backend timed out. Render may be waking up or processing heavy inference.',
        408,
        'TIMEOUT'
      );
    }
    throw new ApiError(
      `Network connection failed: unable to communicate with backend at ${baseUrl}. Ensure backend is running and CORS is configured.`,
      0,
      'NETWORK_ERROR'
    );
  }
}
