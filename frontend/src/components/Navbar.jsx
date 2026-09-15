import React from 'react';

/**
 * Navigation header with live backend status indicator.
 */
export default function Navbar({ healthStatus = 'checking', apiBaseUrl = '' }) {
  const statusLabels = {
    ready: 'Backend Ready',
    degraded: 'Backend Degraded',
    offline: 'Backend Offline',
    checking: 'Checking Connection...',
  };

  return (
    <header className="navbar" role="banner">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <span className="brand-icon" aria-hidden="true"></span>
          <div>
            <h1 className="brand-title">AppleSupport AI Customer Agent</h1>
            <p className="brand-subtitle">Grounded Multi-Stage Pipeline (Phase 10–15)</p>
          </div>
        </div>

        <div className="navbar-status">
          <span className="api-target-badge" title="Target Render backend">
            {apiBaseUrl || 'Render API'}
          </span>
          <div className="status-pill" title={`Backend status: ${healthStatus}`}>
            <span
              className={`status-dot ${healthStatus}`}
              data-testid="backend-status-dot"
            />
            <span>{statusLabels[healthStatus] || healthStatus}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
