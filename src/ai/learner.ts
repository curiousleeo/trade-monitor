/**
 * APEX Self-Learning Weight Optimizer
 *
 * How it works:
 *   Every LEARN_EVERY closed trades, it looks at the signal snapshot saved with
 *   each trade and measures how often each signal was "correct" (its direction
 *   agreed with the trade direction AND the trade won, OR it disagreed AND the
 *   trade lost). Signals with >50% accuracy get a higher weight next cycle;
 *   signals below 50% get reduced. Weights are renormalised to sum to 1 after
 *   each update.
 *
 * Design constraints:
 *   • MIN_WEIGHT / MAX_WEIGHT prevent any signal from vanishing or dominating.
 *   • LEARN_RATE smooths changes so a bad 10-trade streak can't flip the model.
 *   • MIN_TRADES ensures we have enough data before touching anything.
 *   • Only signals with |value| > OPINION_THRESHOLD counted — neutral readings
 *     shouldn't affect the learning signal.
 *
 * Storage: localStorage key 'apex-signal-weights'.
 */

import { Signal, Trade } from '../types';

// ─── Config ──────────────────────────────────────────────────────────────────

const STORAGE_KEY       = 'apex-signal-weights';
const LEARN_EVERY       = 10;   // retrain every N new closed trades
const MIN_TRADES        = 15;   // minimum closed trades before first adjustment
const LEARN_RATE        = 0.25; // 0 = no change, 1 = full jump to target
const MIN_WEIGHT        = 0.03; // floor: no signal below 3%
const MAX_WEIGHT        = 0.40; // ceiling: no signal above 40%
const OPINION_THRESHOLD = 20;   // ignore signals with |value| ≤ 20 (neutral readings)

// ─── Default weights (hardcoded starting point) ──────────────────────────────

export const DEFAULT_WEIGHTS: Record<string, number> = {
  'Multi-TF Alignment': 0.20,
  'Liquidity Sweep':    0.20,
  'MACD Momentum':      0.15,
  'Stoch RSI':          0.15,
  'EMA Trend':          0.15,
  'Candle Pattern':     0.10,
  'VWAP':               0.05,
};

// ─── Types ───────────────────────────────────────────────────────────────────

export interface LearnedWeights {
  weights:    Record<string, number>;  // current learned weights (sum to 1)
  version:    number;                  // increments each learning cycle
  trainedOn:  number;                  // total trades used in last training
  lastUpdate: number;                  // timestamp of last update
  accuracy:   Record<string, number>;  // per-signal accuracy 0–1
}

// ─── Persistence ─────────────────────────────────────────────────────────────

export function loadWeights(): LearnedWeights {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as LearnedWeights;
  } catch { /* ignore */ }
  return {
    weights:    { ...DEFAULT_WEIGHTS },
    version:    0,
    trainedOn:  0,
    lastUpdate: 0,
    accuracy:   {},
  };
}

export function saveWeights(lw: LearnedWeights): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(lw)); } catch { /* ignore */ }
}

export function resetWeights(): void {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}

// ─── Core learning function ───────────────────────────────────────────────────

/**
 * Analyzes all closed trades that have a signal snapshot and updates weights.
 * Returns the new LearnedWeights if training ran, or null if not enough data yet.
 */
export function learnFromTrades(
  trades: Trade[],
  current: LearnedWeights,
): LearnedWeights | null {
  // Only use closed trades that have a signal snapshot
  const usable = trades.filter(
    t =>
      t.signals &&
      t.signals.length > 0 &&
      (t.status === 'CLOSED_TP' || t.status === 'CLOSED_SL'),
  );

  if (usable.length < MIN_TRADES) return null;

  // Tally per-signal: how often was the signal "correct"?
  const correct: Record<string, number> = {};
  const total:   Record<string, number> = {};

  usable.forEach(trade => {
    const won    = trade.status === 'CLOSED_TP';
    const isLong = trade.direction === 'LONG';

    trade.signals!.forEach(sig => {
      if (!total[sig.name]) { total[sig.name] = 0; correct[sig.name] = 0; }

      const inFavor = isLong ? sig.value >  OPINION_THRESHOLD
                              : sig.value < -OPINION_THRESHOLD;
      const against = isLong ? sig.value < -OPINION_THRESHOLD
                              : sig.value >  OPINION_THRESHOLD;

      // Skip neutral readings — they carry no directional opinion
      if (!inFavor && !against) return;

      total[sig.name]++;
      // Correct = signal backed winner OR signal opposed loser
      if ((inFavor && won) || (against && !won)) correct[sig.name]++;
    });
  });

  const signalNames = Object.keys(DEFAULT_WEIGHTS);
  const accuracy:    Record<string, number> = {};
  const newWeights = { ...current.weights };

  signalNames.forEach(name => {
    const n = total[name] ?? 0;

    if (n < 5) {
      // Too few samples for this signal — preserve current weight
      accuracy[name] = current.accuracy[name] ?? 0.50;
      return;
    }

    const acc = correct[name] / n;
    accuracy[name] = acc;

    // Target weight scales linearly: 50% accuracy = no change, 100% = 2× current
    const targetWeight = current.weights[name] * (acc / 0.50);
    const adjusted     = current.weights[name] + LEARN_RATE * (targetWeight - current.weights[name]);
    newWeights[name]   = Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT, adjusted));
  });

  // Renormalise so weights always sum exactly to 1
  const sum = signalNames.reduce((s, n) => s + (newWeights[n] ?? DEFAULT_WEIGHTS[n] ?? 0.10), 0);
  signalNames.forEach(n => {
    newWeights[n] = (newWeights[n] ?? DEFAULT_WEIGHTS[n]) / sum;
  });

  const updated: LearnedWeights = {
    weights:    newWeights,
    version:    current.version + 1,
    trainedOn:  usable.length,
    lastUpdate: Date.now(),
    accuracy,
  };

  saveWeights(updated);
  return updated;
}

/** Returns true when enough new trades have closed to warrant a retrain. */
export function shouldLearn(closedCount: number, lastTrainedOn: number): boolean {
  return closedCount >= MIN_TRADES && (closedCount - lastTrainedOn) >= LEARN_EVERY;
}

/** Merge default weights with any learned weights (fills gaps if new signals added). */
export function resolveWeights(lw: LearnedWeights): Record<string, number> {
  const merged = { ...DEFAULT_WEIGHTS, ...lw.weights };
  // Renormalise in case new default signals were added
  const sum = Object.values(merged).reduce((a, b) => a + b, 0);
  const result: Record<string, number> = {};
  Object.keys(merged).forEach(k => { result[k] = merged[k] / sum; });
  return result;
}

/** Human-readable summary for debugging / UI. */
export function describeWeights(lw: LearnedWeights): string {
  if (lw.version === 0) return 'Model v0 — using default weights (not enough trades yet)';
  const lines = Object.keys(DEFAULT_WEIGHTS).map(name => {
    const w   = (lw.weights[name] ?? DEFAULT_WEIGHTS[name]) * 100;
    const acc = lw.accuracy[name] != null ? ` (${(lw.accuracy[name] * 100).toFixed(0)}% acc)` : '';
    return `  ${name}: ${w.toFixed(1)}%${acc}`;
  });
  return `Model v${lw.version} — trained on ${lw.trainedOn} trades\n${lines.join('\n')}`;
}
