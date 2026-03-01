import { Sentiment } from '../types';

const BULLISH = [
  'surge', 'surges', 'rally', 'rallies', 'bull', 'bullish', 'breakout', 'breaks out',
  'all-time high', 'record high', 'ath', 'gain', 'gains', 'rise', 'rises', 'soar',
  'pump', 'adoption', 'approval', 'approved', 'launch', 'launches', 'partnership',
  'upgrade', 'buy', 'accumulate', 'inflow', 'bottom', 'recovery', 'rebound',
  'outperform', 'positive', 'growth', 'milestone', 'integration',
];

const BEARISH = [
  'crash', 'crashes', 'dump', 'dumps', 'bear', 'bearish', 'hack', 'hacked', 'exploit',
  'ban', 'banned', 'regulation', 'crackdown', 'lawsuit', 'sued', 'fall', 'falls',
  'drop', 'drops', 'plunge', 'plunges', 'decline', 'declines', 'sell-off', 'selloff',
  'liquidation', 'fear', 'concern', 'warning', 'fraud', 'scam', 'ponzi', 'outflow',
  'down', 'loss', 'losses', 'bankrupt', 'collapse', 'collapses', 'risk', 'negative',
];

// Negation words — if found within 20 chars before the matched term, flip the signal
const NEGATIONS = [
  'not ', 'no ', 'never ', "doesn't ", "don't ", "won't ", "isn't ",
  "aren't ", 'without ', 'unlikely ', 'denies ', 'deny ',
];

/**
 * Match a term against text using word boundaries for single words,
 * or substring match for multi-word / hyphenated terms.
 * Prevents "bulletin" matching "bull", "rainfall" matching "fall", etc.
 */
function matchesTerm(text: string, term: string): boolean {
  if (/[\s-]/.test(term)) {
    // Multi-word or hyphenated: exact substring match
    return text.includes(term);
  }
  // Single word: require word boundaries
  return new RegExp(`\\b${term}\\b`).test(text);
}

/** Check if the matched term is preceded by a negation within 25 chars */
function isNegated(text: string, term: string): boolean {
  const idx = text.indexOf(term);
  if (idx < 0) return false;
  const before = text.slice(Math.max(0, idx - 25), idx);
  return NEGATIONS.some(neg => before.includes(neg));
}

export function scoreSentiment(title: string): Sentiment {
  const lower = title.toLowerCase();
  let score = 0;
  BULLISH.forEach(w => {
    if (matchesTerm(lower, w) && !isNegated(lower, w)) score++;
  });
  BEARISH.forEach(w => {
    if (matchesTerm(lower, w) && !isNegated(lower, w)) score--;
  });
  if (score > 0) return 'bullish';
  if (score < 0) return 'bearish';
  return 'neutral';
}
