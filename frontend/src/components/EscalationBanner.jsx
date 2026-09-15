import React from 'react';

/**
 * High-visibility banner rendered when should_escalate is true.
 */
export default function EscalationBanner({ agentResponse }) {
  if (!agentResponse?.should_escalate) {
    return null;
  }

  const {
    escalation_severity = 'normal',
    escalation_rule_triggered = 'manual_review',
    escalation_reason = 'Escalation triggered by safety or policy rule.',
  } = agentResponse;

  return (
    <div
      className="escalation-banner"
      role="alert"
      data-testid="escalation-banner"
    >
      <div className="escalation-header">
        <div className="escalation-title">
          <span aria-hidden="true">🚨</span>
          <span>HUMAN ESCALATION TRIGGERED</span>
        </div>
        <span
          className={`severity-pill ${escalation_severity.toLowerCase()}`}
          data-testid="escalation-severity"
        >
          {escalation_severity} severity
        </span>
      </div>

      <p className="escalation-desc" data-testid="escalation-reason">
        <strong>Reason:</strong> {escalation_reason}
      </p>

      <div className="escalation-meta">
        Rule Triggered: <code>{escalation_rule_triggered}</code>
      </div>
    </div>
  );
}
