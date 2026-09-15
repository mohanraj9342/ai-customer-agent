import React, { useState } from 'react';

export default function ChatInput({
  onSubmit,
  isLoading = false,
  message,
  setMessage,
  topKEvidence,
  setTopKEvidence,
  reviewMode,
  setReviewMode,
}) {
  const maxChars = 1000;
  const charCount = message.length;
  const isOverLimit = charCount > maxChars;
  const isEmpty = message.trim().length === 0;

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isEmpty && !isOverLimit && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <form
      className="input-card"
      onSubmit={(e) => {
        e.preventDefault();
        if (!isEmpty && !isOverLimit && !isLoading) {
          onSubmit();
        }
      }}
      aria-label="Customer inquiry form"
    >
      <div className="input-textarea-wrapper">
        <textarea
          id="customer-message-input"
          className="chat-textarea"
          placeholder="Describe your Apple device issue, question, or error message... (Press Enter to send, Shift+Enter for newline)"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={3}
          maxLength={1200}
          aria-label="Customer inquiry text"
          data-testid="customer-message-input"
        />
      </div>

      <div className="input-footer">
        <div className="input-controls-group">
          <label className="control-item">
            <span>Evidence K:</span>
            <input
              type="number"
              min={1}
              max={10}
              value={topKEvidence}
              onChange={(e) => setTopKEvidence(Number(e.target.value))}
              disabled={isLoading}
              title="Number of grounded historical interactions to retrieve"
              data-testid="top-k-input"
            />
          </label>

          <label className="control-item">
            <span>Reviewer:</span>
            <select
              value={reviewMode}
              onChange={(e) => setReviewMode(e.target.value)}
              disabled={isLoading}
              title="Quality audit mode"
              data-testid="review-mode-select"
            >
              <option value="deterministic">Deterministic (Fast Policy)</option>
              <option value="hybrid">Hybrid (Groq Reviewer)</option>
            </select>
          </label>

          <span
            className={`char-counter ${
              isOverLimit ? 'error' : charCount > 850 ? 'warning' : ''
            }`}
            aria-live="polite"
          >
            {charCount}/{maxChars}
          </span>
        </div>

        <div className="input-actions">
          {message && !isLoading && (
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setMessage('')}
              title="Clear input"
            >
              Clear
            </button>
          )}

          <button
            type="submit"
            className="btn-primary"
            disabled={isEmpty || isOverLimit || isLoading}
            data-testid="send-button"
            aria-busy={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>Send Inquiry</span>
                <span aria-hidden="true">→</span>
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
