import React, { useState } from 'react';

export default function AgentResponseCard({
  agentResponse,
  executionTimeMs = 0,
}) {
  const [copied, setCopied] = useState(false);

  if (!agentResponse) {
    return null;
  }

  const {
    draft_response = '',
    predicted_intent = 'unknown',
    intent_confidence = 0,
    model_used = 'unknown',
    requires_clarification = false,
    grounding_summary = '',
  } = agentResponse;

  const confidencePct = (intent_confidence * 100).toFixed(1);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(draft_response);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers or restricted permissions
      setCopied(false);
    }
  };

  return (
    <article className="agent-card" aria-label="AI Agent Response" data-testid="agent-response-card">
      <header className="agent-card-header">
        <div className="agent-identity">
          <div className="agent-avatar" aria-hidden="true"></div>
          <div>
            <div className="agent-name">AppleSupport Agent</div>
            <div className="header-badges">
              <span className="intent-pill" data-testid="predicted-intent-badge">
                Intent: {predicted_intent}
              </span>
              <span className="confidence-pill" data-testid="intent-confidence-badge">
                {confidencePct}% conf
              </span>
            </div>
          </div>
        </div>

        <button
          type="button"
          className="btn-copy"
          onClick={handleCopy}
          title="Copy response text to clipboard"
          data-testid="copy-response-button"
        >
          {copied ? '✓ Copied' : 'Copy Text'}
        </button>
      </header>

      {requires_clarification && (
        <div className="clarification-callout" role="note" data-testid="clarification-notice">
          <span aria-hidden="true">ℹ️</span>
          <span>
            <strong>Clarification Flag:</strong> Customer inquiry is broad or underspecified; the draft response asks targeted diagnostic questions.
          </span>
        </div>
      )}

      <div className="draft-response-body" data-testid="draft-response-text">
        {draft_response}
      </div>

      <footer className="agent-card-footer">
        <div className="meta-info-list">
          <span title="Inference model used by Phase 12 orchestrator">
            Model: <strong>{model_used}</strong>
          </span>
          {executionTimeMs > 0 && (
            <span title="Total backend execution latency">
              Latency: <strong>{executionTimeMs.toFixed(1)} ms</strong>
            </span>
          )}
        </div>

        {grounding_summary && (
          <span className="grounding-note" title={grounding_summary}>
            {grounding_summary}
          </span>
        )}
      </footer>
    </article>
  );
}
