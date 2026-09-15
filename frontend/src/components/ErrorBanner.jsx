import React from 'react';

/**
 * Error display banner with actionable guidance and dismiss button.
 */
export default function ErrorBanner({ error, onDismiss, onRetry }) {
  if (!error) {
    return null;
  }

  const isValidationError = error.status === 422 || error.code === 'VALIDATION_ERROR';
  const isServiceUnavailable = error.status === 503 || error.code === 'HTTP_503';
  const isNetwork = error.code === 'NETWORK_ERROR' || error.status === 0;

  let title = 'Request Failed';
  if (isValidationError) {
    title = 'Input Validation Error';
  } else if (isServiceUnavailable) {
    title = 'AI Backend Initializing / Unavailable';
  } else if (isNetwork) {
    title = 'Backend Network Connection Failed';
  }

  return (
    <div className="error-banner" role="alert" data-testid="error-banner">
      <div className="error-banner-content">
        <div className="error-banner-title">
          <span aria-hidden="true">⚠️</span> {title} [{error.code || error.status || 'ERROR'}]
        </div>
        <p className="error-banner-desc" data-testid="error-message">
          {error.message}
        </p>

        {error.details && Array.isArray(error.details) && error.details.length > 0 && (
          <div className="error-banner-details" data-testid="error-details">
            {error.details.map((d, idx) => (
              <div key={idx}>
                &bull; {d.location ? `${d.location}: ` : ''}{d.issue || JSON.stringify(d)}
              </div>
            ))}
          </div>
        )}

        {isNetwork && (
          <p className="error-banner-desc" style={{ marginTop: '0.25rem', fontSize: '0.8rem' }}>
            Tip: Verify that the Render AI backend is running and that CORS allows requests from this origin.
          </p>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {onRetry && (
          <button
            type="button"
            className="btn-secondary"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', color: '#fff' }}
            onClick={onRetry}
            data-testid="retry-button"
          >
            Retry
          </button>
        )}
        <button
          type="button"
          className="btn-dismiss"
          onClick={onDismiss}
          title="Dismiss alert"
          aria-label="Dismiss error"
          data-testid="dismiss-error-button"
        >
          &times;
        </button>
      </div>
    </div>
  );
}
