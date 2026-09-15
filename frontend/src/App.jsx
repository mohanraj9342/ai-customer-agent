import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import PromptChips from './components/PromptChips';
import ChatInput from './components/ChatInput';
import AgentResponseCard from './components/AgentResponseCard';
import EscalationBanner from './components/EscalationBanner';
import ReviewerAuditCard from './components/ReviewerAuditCard';
import EvidenceAccordion from './components/EvidenceAccordion';
import ErrorBanner from './components/ErrorBanner';
import {
  checkBackendReadiness,
  checkBackendHealth,
  sendChatMessage,
  getApiBaseUrl,
} from './services/api';

export default function App() {
  const [message, setMessage] = useState('');
  const [topKEvidence, setTopKEvidence] = useState(3);
  const [reviewMode, setReviewMode] = useState('deterministic');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatResult, setChatResult] = useState(null);
  const [healthStatus, setHealthStatus] = useState('checking');
  const [lastSubmittedMessage, setLastSubmittedMessage] = useState('');

  const apiBaseUrl = getApiBaseUrl();

  // Check backend connectivity on mount
  const checkStatus = useCallback(async () => {
    try {
      const ready = await checkBackendReadiness(6000);
      setHealthStatus(ready.status || 'ready');
    } catch {
      try {
        const health = await checkBackendHealth(4000);
        setHealthStatus(health.status === 'ok' ? 'degraded' : 'offline');
      } catch {
        setHealthStatus('offline');
      }
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  // Handle message submission
  const handleSubmit = async () => {
    const trimmed = message.trim();
    if (!trimmed) {
      setError({
        status: 422,
        code: 'VALIDATION_ERROR',
        message: 'Message cannot be empty or solely whitespace.',
      });
      return;
    }

    setIsLoading(true);
    setError(null);
    setLastSubmittedMessage(trimmed);

    try {
      const result = await sendChatMessage({
        message: trimmed,
        top_k_evidence: topKEvidence,
        include_review: true,
        review_mode: reviewMode,
      });
      setChatResult(result);
      setHealthStatus('ready');
    } catch (err) {
      setError(err);
      if (err.code === 'NETWORK_ERROR') {
        setHealthStatus('offline');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectPrompt = (promptText) => {
    setMessage(promptText);
    setError(null);
  };

  return (
    <div className="app-container">
      <Navbar healthStatus={healthStatus} apiBaseUrl={apiBaseUrl} />

      <main className="main-content">
        <PromptChips
          onSelectPrompt={handleSelectPrompt}
          disabled={isLoading}
        />

        <ChatInput
          onSubmit={handleSubmit}
          isLoading={isLoading}
          message={message}
          setMessage={setMessage}
          topKEvidence={topKEvidence}
          setTopKEvidence={setTopKEvidence}
          reviewMode={reviewMode}
          setReviewMode={setReviewMode}
        />

        <ErrorBanner
          error={error}
          onDismiss={() => setError(null)}
          onRetry={lastSubmittedMessage ? handleSubmit : null}
        />

        {chatResult && (
          <div className="results-grid" data-testid="results-grid">
            <div className="column">
              <EscalationBanner agentResponse={chatResult.response} />
              <AgentResponseCard
                agentResponse={chatResult.response}
                executionTimeMs={chatResult.execution_time_ms}
              />
              <EvidenceAccordion
                retrievedEvidence={chatResult.response?.retrieved_evidence}
                citedEvidenceIds={chatResult.response?.cited_evidence_ids}
              />
            </div>

            <div className="column">
              <ReviewerAuditCard review={chatResult.review} />
            </div>
          </div>
        )}
      </main>

      <footer className="footer" role="contentinfo">
        AppleSupport AI Customer Agent &bull; Phase 16 Vercel Frontend &bull; Powered by Render AI Backend
      </footer>
    </div>
  );
}
