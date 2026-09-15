import React from 'react';

const SAMPLE_PROMPTS = [
  {
    label: '🔋 Battery Drain',
    text: 'My iPhone 13 battery dies after only 2 hours of normal use.',
    hazard: false,
  },
  {
    label: '🔄 Update Loop',
    text: 'iOS update keeps failing and stuck on Apple logo reboot loop.',
    hazard: false,
  },
  {
    label: '⚠️ Swollen Battery Alert',
    text: 'My iPhone battery is swollen and smoking with a strong chemical smell!',
    hazard: true,
  },
  {
    label: '😡 Repeated Failure Frustration',
    text: 'I have talked to 4 representatives, none of them helped. Let me speak to a human manager now!',
    hazard: false,
  },
  {
    label: '💳 Cancel Subscription',
    text: 'How do I cancel my Apple Music trial subscription before it charges?',
    hazard: false,
  },
];

export default function PromptChips({ onSelectPrompt, disabled = false }) {
  return (
    <section className="chips-section" aria-label="Quick test inquiries">
      <span className="chips-label">Quick Test Scenarios:</span>
      <div className="chips-list">
        {SAMPLE_PROMPTS.map((item, idx) => (
          <button
            key={idx}
            type="button"
            className={`chip-btn ${item.hazard ? 'hazard' : ''}`}
            onClick={() => onSelectPrompt(item.text)}
            disabled={disabled}
            title={item.text}
          >
            {item.label}
          </button>
        ))}
      </div>
    </section>
  );
}
