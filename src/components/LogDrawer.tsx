import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { LogEntry, Trade } from '../types';
import { LearnedWeights, DEFAULT_WEIGHTS } from '../ai/learner';

interface Props {
  activityLog:    LogEntry[];
  learnedWeights: LearnedWeights;
  closedTrades:   Trade[];
  onClose:        () => void;
}

export function LogDrawer({ activityLog, learnedWeights, closedTrades, onClose }: Props) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return createPortal(
    <div className="log-drawer-overlay" onClick={onClose}>
      <div className="log-drawer" onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="log-drawer-header">
          <div className="log-drawer-title">
            <span className="log-drawer-icon">⟳</span>
            APEX LOG
          </div>
          <button className="log-drawer-close" onClick={onClose}>✕</button>
        </div>

        <div className="log-drawer-body">

          {/* ── Activity Feed ───────────────────────────── */}
          <div className="log-drawer-section-title">ACTIVITY</div>
          {activityLog.length === 0 ? (
            <div className="log-drawer-empty">No activity yet — waiting for first trade…</div>
          ) : (
            <div className="log-drawer-feed">
              {activityLog.map(entry => (
                <div key={entry.id} className={`log-entry log-entry--${entry.level}`}>
                  <span className="log-time">
                    {new Date(entry.timestamp).toLocaleTimeString('en-US', {
                      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
                    })}
                  </span>
                  <div className="log-body">
                    <div className="log-message">{entry.message}</div>
                    {entry.detail && <div className="log-detail">{entry.detail}</div>}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ── Model Weights ────────────────────────────── */}
          <div className="log-drawer-section-title" style={{ marginTop: 20 }}>
            MODEL WEIGHTS
            <span className="log-drawer-section-meta">
              {learnedWeights.version > 0
                ? `v${learnedWeights.version} · ${learnedWeights.trainedOn} trades`
                : `v0 default · needs ${Math.max(0, 15 - closedTrades.length)} more trades`}
            </span>
          </div>
          <div className="log-drawer-weights">
            <div className="log-drawer-weights-header">
              <span>Signal</span>
              <span>Default</span>
              <span>Current</span>
              <span>Accuracy</span>
            </div>
            {Object.keys(DEFAULT_WEIGHTS).map(name => {
              const def  = DEFAULT_WEIGHTS[name];
              const cur  = learnedWeights.weights[name] ?? def;
              const acc  = learnedWeights.accuracy[name];
              const diff = cur - def;
              return (
                <div key={name} className="log-drawer-weights-row">
                  <span className="log-drawer-sig-name">{name}</span>
                  <span className="log-drawer-sig-def">{(def * 100).toFixed(0)}%</span>
                  <span className={`log-drawer-sig-cur ${diff > 0.005 ? 'up' : diff < -0.005 ? 'down' : ''}`}>
                    {(cur * 100).toFixed(1)}%
                    {Math.abs(diff) > 0.005 && (
                      <span className="learn-weight-delta">
                        {diff > 0 ? ' ▲' : ' ▼'}{Math.abs(diff * 100).toFixed(1)}
                      </span>
                    )}
                  </span>
                  <span className="log-drawer-sig-acc">
                    {acc != null ? `${(acc * 100).toFixed(0)}%` : '—'}
                  </span>
                </div>
              );
            })}
          </div>

          {/* ── Learning History ─────────────────────────── */}
          {(learnedWeights.history ?? []).length > 0 && (
            <>
              <div className="log-drawer-section-title" style={{ marginTop: 20 }}>LEARNING HISTORY</div>
              <div className="log-drawer-history">
                {(learnedWeights.history ?? []).map((event, i) => (
                  <div key={i} className="learn-event">
                    <div className="learn-event-header">
                      <span className="learn-event-title">Model v{event.version}</span>
                      <span className="learn-event-meta">
                        {new Date(event.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        {' · '}{event.trainedOn} trades
                      </span>
                    </div>
                    <div className="learn-event-summary">{event.summary}</div>
                    {event.changes.filter(c => Math.abs(c.delta) > 0.003).map(c => (
                      <div key={c.signal} className={`learn-change learn-change--${c.delta >= 0 ? 'up' : 'down'}`}>
                        <span>{c.delta >= 0 ? '▲' : '▼'} {c.signal}</span>
                        <span>{(c.oldWeight * 100).toFixed(1)}% → {(c.newWeight * 100).toFixed(1)}%</span>
                        <span className="learn-change-acc">{(c.accuracy * 100).toFixed(0)}% acc</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}

          <div style={{ height: 32 }} />
        </div>
      </div>
    </div>,
    document.body,
  );
}
