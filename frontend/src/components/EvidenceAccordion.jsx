import React from 'react';

/**
 * Collapsible inspection cards for grounded historical evidence cases (Phase 11).
 */
export default function EvidenceAccordion({
  retrievedEvidence = [],
  citedEvidenceIds = [],
}) {
  if (!retrievedEvidence || retrievedEvidence.length === 0) {
    return null;
  }

  const citedSet = new Set(citedEvidenceIds || []);

  return (
    <section className="evidence-card" aria-label="Retrieved Historical Evidence" data-testid="evidence-section">
      <div className="evidence-header">
        <h2 className="evidence-title">
          <span aria-hidden="true">🔍</span>
          <span>Grounded Historical Context</span>
        </h2>
        <span className="evidence-count-badge">
          {retrievedEvidence.length} cases retrieved
        </span>
      </div>

      <div className="evidence-items-container">
        {retrievedEvidence.map((item, index) => {
          const isCited =
            citedSet.has(item.customer_tweet_id) ||
            citedSet.has(Number(item.customer_tweet_id));
          const simPct = (item.similarity_score * 100).toFixed(1);

          return (
            <article
              key={item.customer_tweet_id || index}
              className={`evidence-item ${isCited ? 'cited' : ''}`}
              data-testid={`evidence-item-${index}`}
            >
              <div className="evidence-item-header">
                <span className="evidence-rank">
                  Rank #{item.rank || index + 1} &bull; Thread #{item.thread_id}
                </span>
                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  {isCited && (
                    <span className="cited-badge" title="Cited in draft response">
                      ✓ Cited
                    </span>
                  )}
                  <span className="evidence-sim-pill" title="Dense Cosine Similarity">
                    {simPct}% match
                  </span>
                </div>
              </div>

              <div className="tweet-box">
                <span className="tweet-author">
                  Customer Tweet #{item.customer_tweet_id}:
                </span>
                <p className="tweet-text">{item.customer_text}</p>
              </div>

              <div className="tweet-box">
                <span className="tweet-author">
                  AppleSupport Response #{item.brand_tweet_id}:
                </span>
                <p className="tweet-text brand">{item.brand_text}</p>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
