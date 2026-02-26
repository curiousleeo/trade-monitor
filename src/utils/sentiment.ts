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

export function scoreSentiment(title: string): Sentiment {
  const lower = title.toLowerCase();
  let score = 0;
  BULLISH.forEach(w => { if (lower.includes(w)) score++; });
  BEARISH.forEach(w => { if (lower.includes(w)) score--; });
  if (score > 0) return 'bullish';
  if (score < 0) return 'bearish';
  return 'neutral';
}
