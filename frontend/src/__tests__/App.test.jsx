import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import App from '../App';
import * as api from '../services/api';

describe('App Interface (frontend/src/App.jsx)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Mock backend readiness check default
    vi.spyOn(api, 'checkBackendReadiness').mockResolvedValue({
      status: 'ready',
      intent_classifier_ready: true,
      retriever_ready: true,
      groq_configured: true,
    });
    vi.spyOn(api, 'checkBackendHealth').mockResolvedValue({
      status: 'ok',
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders navbar, prompt chips, and chat input form', async () => {
    render(<App />);

    expect(screen.getByText('AppleSupport AI Customer Agent')).toBeInTheDocument();
    expect(screen.getByText(/Grounded Multi-Stage Pipeline/i)).toBeInTheDocument();
    expect(screen.getByTestId('customer-message-input')).toBeInTheDocument();
    expect(screen.getByTestId('send-button')).toBeInTheDocument();
    expect(screen.getByText(/Quick Test Scenarios:/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Backend Ready')).toBeInTheDocument();
    });
  });

  it('populates message input when clicking a prompt chip', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Backend Ready')).toBeInTheDocument();
    });

    const batteryChip = screen.getByRole('button', { name: /🔋 Battery Drain/i });
    fireEvent.click(batteryChip);

    const input = screen.getByTestId('customer-message-input');
    expect(input.value).toBe(
      'My iPhone 13 battery dies after only 2 hours of normal use.'
    );
  });

  it('disables submit button and shows loading state during message processing', async () => {
    let resolveApiCall;
    const promise = new Promise((resolve) => {
      resolveApiCall = resolve;
    });

    vi.spyOn(api, 'sendChatMessage').mockReturnValue(promise);

    render(<App />);

    const input = screen.getByTestId('customer-message-input');
    fireEvent.change(input, { target: { value: 'How to check battery health?' } });

    const sendBtn = screen.getByTestId('send-button');
    fireEvent.click(sendBtn);

    // Should enter loading state
    expect(screen.getByText(/Processing.../i)).toBeInTheDocument();
    expect(sendBtn).toBeDisabled();
    expect(input).toBeDisabled();

    // Resolve API call
    resolveApiCall({
      response: {
        customer_message: 'How to check battery health?',
        predicted_intent: 'battery_power',
        intent_confidence: 0.98,
        draft_response: 'Navigate to Settings > Battery > Battery Health.',
        should_escalate: false,
        model_used: 'qwen/qwen3.8-27b',
        retrieved_evidence: [],
        cited_evidence_ids: [],
      },
      review: {
        decision: 'PASS',
        overall_score: 0.96,
        dimension_scores: { grounding_support: 0.95 },
        reviewer_rationale: 'Accurate navigation steps.',
      },
      execution_time_ms: 92.4,
    });

    await waitFor(() => {
      expect(screen.getByTestId('draft-response-text')).toBeInTheDocument();
      expect(screen.getByText('Send Inquiry')).toBeInTheDocument();
    });
  });

  it('renders agent response card, intent badge, and reviewer card upon success', async () => {
    vi.spyOn(api, 'sendChatMessage').mockResolvedValue({
      response: {
        customer_message: 'iPhone screen is flickering',
        predicted_intent: 'display_screen',
        intent_confidence: 0.9412,
        draft_response: 'Try restarting your device or checking True Tone in Settings > Display.',
        should_escalate: false,
        model_used: 'qwen/qwen3.8-27b',
        grounding_summary: 'Guided by display troubleshooting evidence.',
        retrieved_evidence: [
          {
            rank: 1,
            similarity_score: 0.875,
            thread_id: '88412',
            customer_tweet_id: 100201,
            customer_text: 'Screen keeps flickering after iOS update',
            brand_tweet_id: 100202,
            brand_text: 'Have you tried a forced restart?',
            inferred_intent: 'display_screen',
          },
        ],
        cited_evidence_ids: [100201],
      },
      review: {
        decision: 'PASS',
        overall_score: 0.935,
        dimension_scores: {
          grounding_support: 0.92,
          hallucination_risk: 1.0,
          response_relevance: 0.95,
        },
        detected_issues: [],
        reviewer_rationale: 'Safe grounded response.',
        reviewer_mode: 'deterministic',
      },
      execution_time_ms: 105.8,
    });

    render(<App />);

    const input = screen.getByTestId('customer-message-input');
    fireEvent.change(input, { target: { value: 'iPhone screen is flickering' } });
    fireEvent.click(screen.getByTestId('send-button'));

    await waitFor(() => {
      expect(screen.getByTestId('draft-response-text')).toHaveTextContent(
        'Try restarting your device or checking True Tone in Settings > Display.'
      );
      expect(screen.getByTestId('predicted-intent-badge')).toHaveTextContent(
        'Intent: display_screen'
      );
      expect(screen.getByTestId('intent-confidence-badge')).toHaveTextContent('94.1% conf');
      expect(screen.getByTestId('reviewer-decision-badge')).toHaveTextContent('PASS');
      expect(screen.getByTestId('reviewer-score')).toHaveTextContent('93.5%');
      expect(screen.getByText(/Screen keeps flickering after iOS update/i)).toBeInTheDocument();
      expect(screen.getByText('✓ Cited')).toBeInTheDocument();
    });
  });

  it('renders critical safety escalation banner when safety hazard alert triggers', async () => {
    vi.spyOn(api, 'sendChatMessage').mockResolvedValue({
      response: {
        customer_message: 'My battery exploded in smoke!',
        predicted_intent: 'battery_power',
        intent_confidence: 0.99,
        draft_response: 'SAFETY NOTICE: Discontinue use immediately and visit an Apple Store.',
        should_escalate: true,
        escalation_severity: 'critical',
        escalation_rule_triggered: 'safety_hazard_alert',
        escalation_reason: 'Physical safety hazard or battery expansion detected',
        model_used: 'qwen/qwen3.8-27b',
        retrieved_evidence: [],
        cited_evidence_ids: [],
      },
      review: {
        decision: 'PASS',
        overall_score: 1.0,
        dimension_scores: { escalation_correctness: 1.0 },
        reviewer_rationale: 'Correctly escalated critical safety hazard.',
        reviewer_mode: 'deterministic',
      },
      execution_time_ms: 45.1,
    });

    render(<App />);

    const hazardChip = screen.getByRole('button', { name: /⚠️ Swollen Battery Alert/i });
    fireEvent.click(hazardChip);
    fireEvent.click(screen.getByTestId('send-button'));

    await waitFor(() => {
      expect(screen.getByTestId('escalation-banner')).toBeInTheDocument();
      expect(screen.getByTestId('escalation-severity')).toHaveTextContent('critical severity');
      expect(screen.getByTestId('escalation-reason')).toHaveTextContent(
        'Physical safety hazard or battery expansion detected'
      );
      expect(screen.getByText('safety_hazard_alert')).toBeInTheDocument();
    });
  });

  it('renders error banner on backend failure and allows retry', async () => {
    const errorObj = new api.ApiError(
      'AI inference service is temporarily unavailable.',
      503,
      'SERVICE_UNAVAILABLE'
    );
    vi.spyOn(api, 'sendChatMessage').mockRejectedValue(errorObj);

    render(<App />);

    const input = screen.getByTestId('customer-message-input');
    fireEvent.change(input, { target: { value: 'Help with my iPad' } });
    fireEvent.click(screen.getByTestId('send-button'));

    await waitFor(() => {
      expect(screen.getByTestId('error-banner')).toBeInTheDocument();
      expect(screen.getByTestId('error-message')).toHaveTextContent(
        'AI inference service is temporarily unavailable.'
      );
    });

    // Dismiss error
    const dismissBtn = screen.getByTestId('dismiss-error-button');
    fireEvent.click(dismissBtn);

    expect(screen.queryByTestId('error-banner')).not.toBeInTheDocument();
  });
});
