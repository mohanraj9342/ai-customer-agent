import React from 'react';

/**
 * Phase 13 Reviewer Audit Inspector Card.
 */
export default function ReviewerAuditCard({ review }) {
  if (!review) {
    return null;
  }

  const {
    decision = 'PASS',
    overall_score = 0,
    dimension_scores = {},
    detected_issues = [],
    reviewer_rationale = '',
    reviewer_mode = 'deterministic',
  } = review;

  const decisionClass = decision.toLowerCase();
  const scorePct = (overall_score * 100).toFixed(1);

  // Friendly names for evaluation dimensions
  const dimensionLabels = {
    grounding_support: 'Grounding Support',
    hallucination_risk: 'Hallucination Safety',
    escalation_correctness: 'Escalation Accuracy',
    intent_consistency: 'Intent Consistency',
    response_relevance: 'Relevance',
    response_completeness: 'Completeness',
    professional_quality: 'Professional Tone',
  };

  return (
    <section className="reviewer-card" aria-label="Reviewer Audit Verdict" data-testid="reviewer-card">
      <div className="reviewer-header">
        <h2 className="reviewer-title">
          <span aria-hidden="true">🛡️</span>
          <span>Independent Quality Audit</span>
        </h2>
        <span
          className={`decision-badge ${decisionClass}`}
          data-testid="reviewer-decision-badge"
        >
          {decision.replace(/_/g, ' ')}
        </span>
      </div>

      <div className="score-overview">
        <span className="score-label">Overall Quality Score ({reviewer_mode}):</span>
        <span className="score-number" data-testid="reviewer-score">
          {scorePct}%
        </span>
      </div>

      <div className="dimension-grid" data-testid="dimension-scores-grid">
        {Object.entries(dimension_scores).map(([dimKey, score]) => {
          const pct = Math.round(score * 100);
          return (
            <div key={dimKey} className="dimension-item">
              <span className="dimension-name">
                {dimensionLabels[dimKey] || dimKey.replace(/_/g, ' ')}
              </span>
              <div className="dimension-bar-wrapper">
                <div className="dimension-bar">
                  <div
                    className="dimension-bar-fill"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="dimension-score">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {detected_issues && detected_issues.length > 0 && (
        <div className="detected-issues-list" data-testid="detected-issues">
          <span className="score-label">Detected Quality Issues:</span>
          {detected_issues.map((issue, idx) => (
            <div key={idx} className="issue-item">
              <strong>[{issue.severity || 'warning'}] {issue.dimension}:</strong>{' '}
              {issue.description}
            </div>
          ))}
        </div>
      )}

      {reviewer_rationale && (
        <p className="reviewer-rationale" data-testid="reviewer-rationale">
          <strong>Rationale:</strong> {reviewer_rationale}
        </p>
      )}
    </section>
  );
}
