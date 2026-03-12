/**
 * APEX Prediction Engine — TA v2
 *
 * Pure TA model targeting >60% win rate via:
 *   1. Multi-TF Alignment   (20%) — 4 resampled TFs must agree
 *   2. Liquidity Sweep      (20%) — SSL/BSL stop hunts = highest-prob entries
 *   3. MACD Momentum        (15%) — histogram + crossover
 *   4. Stoch RSI            (15%) — oversold/overbought K/D crossovers
 *   5. EMA Trend            (15%) — price vs EMA20/50/200 stack
 *   6. Candle Pattern       (10%) — pin bar, engulfing, marubozu
 *   7. VWAP Position         (5%) — rolling VWAP reclaim/rejection
 *
 * Hard filter: if Multi-TF strongly opposes the composite signal, force NEUTRAL.
 * No sentiment or news signals — pure price action.
 */

import { Candle, Coin, FearGreed, FundingRate, Prediction, PredictionDirection, Signal, TFBias, Timeframe } from '../types';
import { calcEMA, calcRSI, calcVWAP, calcStochRSI, calcMACD } from '../utils/indicators';
import { DEFAULT_WEIGHTS, loadWeights, resolveWeights } from './learner';

// ─── ATR ─────────────────────────────────────────────────────────────────────

export function calcATR(candles: Candle[], period = 14): number {
  if (candles.length < period + 1) return candles.length > 1 ? candles[candles.length - 1].high - candles[candles.length - 1].low : 100;
  const trs: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const hl = candles[i].high - candles[i].low;
    const hc = Math.abs(candles[i].high - candles[i - 1].close);
    const lc = Math.abs(candles[i].low  - candles[i - 1].close);
    trs.push(Math.max(hl, hc, lc));
  }
  let atr = trs.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < trs.length; i++) {
    atr = (atr * (period - 1) + trs[i]) / period;
  }
  return atr;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function resampleCandles(candles: Candle[], factor: number): Candle[] {
  if (factor <= 1) return candles;
  const result: Candle[] = [];
  for (let i = 0; i + factor <= candles.length; i += factor) {
    const slice = candles.slice(i, i + factor);
    result.push({
      time:   slice[0].time,
      open:   slice[0].open,
      high:   Math.max(...slice.map(c => c.high)),
      low:    Math.min(...slice.map(c => c.low)),
      close:  slice[slice.length - 1].close,
      volume: slice.reduce((s, c) => s + c.volume, 0),
    });
  }
  return result;
}

function quickTrend(candles: Candle[]): { direction: PredictionDirection; score: number } {
  if (candles.length < 20) return { direction: 'NEUTRAL', score: 50 };
  const closes = candles.map(c => c.close);
  const price = closes[closes.length - 1];
  const ema20arr = calcEMA(closes, 20);
  const ema50arr = candles.length >= 50 ? calcEMA(closes, 50) : null;
  const ema20 = ema20arr[ema20arr.length - 1];
  const ema50 = ema50arr ? ema50arr[ema50arr.length - 1] : null;

  let score = 50;
  if (ema20 !== null) score += price > ema20 ? 20 : -20;
  if (ema20 !== null && ema50 !== null) {
    score += price > ema50 ? 15 : -15;
    score += ema20 > ema50 ? 15 : -15;
  }
  score = Math.max(0, Math.min(100, score));
  const direction: PredictionDirection = score > 60 ? 'LONG' : score < 40 ? 'SHORT' : 'NEUTRAL';
  return { direction, score };
}

/** Find recent swing highs and lows (excludes last 2 candles so current candle can sweep them). */
function findRecentSwings(candles: Candle[], lookback = 4, window = 50): { highs: number[]; lows: number[] } {
  const slice = candles.slice(-window - lookback, -2); // exclude last 2
  const highs: number[] = [];
  const lows: number[] = [];

  for (let i = lookback; i < slice.length - lookback; i++) {
    const c = slice[i];
    let isHigh = true;
    let isLow  = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (slice[j].high >= c.high) isHigh = false;
      if (slice[j].low  <= c.low)  isLow  = false;
    }
    if (isHigh) highs.push(c.high);
    if (isLow)  lows.push(c.low);
  }

  return { highs, lows };
}

// ─── Individual Scorers (each returns -100 to +100) ──────────────────────────

/**
 * Multi-TF Alignment — resample to 4 timeframes and count direction agreement.
 * Penalises trades where timeframes disagree. Best trades: 3-4/4 TFs aligned.
 */
function scoreMultiTFAlignment(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 80) return { value: 0, description: 'Not enough candles for multi-TF analysis' };

  const factors = [2, 4, 8, 16];
  const results = factors.map(f => quickTrend(resampleCandles(candles, f)));

  const longCount  = results.filter(r => r.direction === 'LONG').length;
  const shortCount = results.filter(r => r.direction === 'SHORT').length;

  let score: number;
  let desc: string;

  if (longCount === 4) {
    score = 100; desc = '4/4 TFs bullish — perfect uptrend alignment';
  } else if (longCount === 3) {
    score = 70;  desc = `3/4 TFs bullish — strong uptrend alignment`;
  } else if (shortCount === 4) {
    score = -100; desc = '4/4 TFs bearish — perfect downtrend alignment';
  } else if (shortCount === 3) {
    score = -70; desc = `3/4 TFs bearish — strong downtrend alignment`;
  } else if (longCount === 2 && shortCount === 0) {
    score = 30;  desc = '2/4 TFs bullish, rest neutral — mild bullish bias';
  } else if (shortCount === 2 && longCount === 0) {
    score = -30; desc = '2/4 TFs bearish, rest neutral — mild bearish bias';
  } else if (longCount === 2 && shortCount === 1) {
    score = 15;  desc = '2/4 TFs bullish but 1 opposing — weak signal';
  } else if (shortCount === 2 && longCount === 1) {
    score = -15; desc = '2/4 TFs bearish but 1 opposing — weak signal';
  } else {
    score = (longCount - shortCount) * 10;
    desc = `Mixed TFs (${longCount}↑ ${shortCount}↓) — no clear alignment`;
  }

  return { value: Math.max(-100, Math.min(100, score)), description: desc };
}

/**
 * Liquidity Sweep — the highest-probability TA entry.
 *
 * SSL sweep (bullish): current candle wicked BELOW a swing low but CLOSED above it.
 *   → Smart money grabbed sell-stop orders below, now driving price up. LONG signal.
 *
 * BSL sweep (bearish): current candle wicked ABOVE a swing high but CLOSED below it.
 *   → Smart money grabbed buy-stop orders above, now driving price down. SHORT signal.
 *
 * Equal highs/lows: clusters of swing points at the same level = liquidity pool.
 *   → Price is likely drawn to sweep them before reversing.
 */
function scoreLiquiditySweep(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 30) return { value: 0, description: 'Not enough data for liquidity analysis' };

  const cur = candles[candles.length - 1];
  const { highs, lows } = findRecentSwings(candles, 4, 60);

  if (highs.length === 0 && lows.length === 0) {
    return { value: 0, description: 'No swing points found — price in discovery mode' };
  }

  // SSL sweep: wick below swing low + close above it (stop hunt then reclaim)
  const sweptLows = lows.filter(l => cur.low < l && cur.close > l);
  if (sweptLows.length > 0) {
    const level = Math.max(...sweptLows); // closest swept low
    const rng = cur.high - cur.low + 0.0001;
    const reclaimPct = ((cur.close - cur.low) / rng) * 100;
    // More swept levels = stronger signal, higher wick reclaim = more conviction
    const strength = Math.min(100, 60 + sweptLows.length * 10 + (reclaimPct > 70 ? 20 : 0));
    return {
      value: strength,
      description: `SSL sweep at $${level.toFixed(2)} — stop hunt below swing low, closed above (reclaim ${reclaimPct.toFixed(0)}% of wick) — high-prob LONG`,
    };
  }

  // BSL sweep: wick above swing high + close below it (stop hunt then rejection)
  const sweptHighs = highs.filter(h => cur.high > h && cur.close < h);
  if (sweptHighs.length > 0) {
    const level = Math.min(...sweptHighs); // closest swept high
    const rng = cur.high - cur.low + 0.0001;
    const reclaimPct = ((cur.high - cur.close) / rng) * 100;
    const strength = Math.min(100, 60 + sweptHighs.length * 10 + (reclaimPct > 70 ? 20 : 0));
    return {
      value: -strength,
      description: `BSL sweep at $${level.toFixed(2)} — stop hunt above swing high, closed below (rejection ${reclaimPct.toFixed(0)}% of wick) — high-prob SHORT`,
    };
  }

  // Equal highs / lows detection (liquidity magnet — not yet swept)
  const tolerance = 0.0015; // 0.15% price tolerance
  let eqHighClusters = 0;
  let eqLowClusters = 0;
  for (let i = 0; i < highs.length; i++) {
    for (let j = i + 1; j < highs.length; j++) {
      if (Math.abs(highs[i] - highs[j]) / highs[i] < tolerance) eqHighClusters++;
    }
  }
  for (let i = 0; i < lows.length; i++) {
    for (let j = i + 1; j < lows.length; j++) {
      if (Math.abs(lows[i] - lows[j]) / lows[i] < tolerance) eqLowClusters++;
    }
  }

  if (eqHighClusters >= 2) {
    return { value: -20, description: `Equal highs detected (${eqHighClusters} clusters) — BSL pool above, potential SHORT after sweep` };
  }
  if (eqLowClusters >= 2) {
    return { value: 20, description: `Equal lows detected (${eqLowClusters} clusters) — SSL pool below, potential LONG after sweep` };
  }

  return { value: 0, description: 'No liquidity sweep — price in discovery, no stop-hunt signal' };
}

/**
 * MACD Momentum — histogram direction, crossovers, and zero-line position.
 * Crossovers near the zero line have the highest win rates.
 */
function scoreMACD(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 40) return { value: 0, description: 'Not enough data for MACD' };

  const closes = candles.map(c => c.close);
  const macdArr = calcMACD(closes);

  const last  = macdArr[macdArr.length - 1];
  const prev  = macdArr[macdArr.length - 2];

  if (last.macd === null || last.signal === null || last.histogram === null) {
    return { value: 0, description: 'MACD warming up' };
  }

  const parts: string[] = [];
  let score = 0;

  // Histogram direction and expansion
  if (last.histogram > 0) {
    score += 30;
    parts.push('histogram positive');
    if (prev.histogram !== null && last.histogram > prev.histogram) {
      score += 20; parts.push('expanding bullish');
    } else {
      score -= 5; parts.push('shrinking');
    }
  } else {
    score -= 30;
    parts.push('histogram negative');
    if (prev.histogram !== null && last.histogram < prev.histogram) {
      score -= 20; parts.push('expanding bearish');
    } else {
      score += 5; parts.push('shrinking bearish');
    }
  }

  // Crossover (strongest signal — fresh crossovers are most reliable)
  if (prev.macd !== null && prev.signal !== null) {
    if (prev.macd < prev.signal && last.macd > last.signal) {
      score += 45; parts.push('bullish crossover ✓');
    } else if (prev.macd > prev.signal && last.macd < last.signal) {
      score -= 45; parts.push('bearish crossover ✓');
    }
  }

  // Zero line — above zero = established bull momentum
  if (last.macd > 0) { score += 10; parts.push('above zero'); }
  else               { score -= 10; parts.push('below zero'); }

  return {
    value: Math.max(-100, Math.min(100, score)),
    description: parts.join(', '),
  };
}

/**
 * Stochastic RSI — K/D crossovers in oversold (<20) or overbought (>80) zones
 * are the highest-probability reversal signals in this model.
 */
function scoreStochRSI(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 40) return { value: 0, description: 'Not enough data for StochRSI' };

  const closes = candles.map(c => c.close);
  const arr = calcStochRSI(closes);

  const last = arr[arr.length - 1];
  const prev = arr[arr.length - 2];

  if (last.k === null || last.d === null) return { value: 0, description: 'StochRSI warming up' };

  const { k, d } = last;
  const prevK = prev.k;
  const prevD = prev.d;

  let score = 0;
  let desc = '';

  if (k < 20 && d < 20) {
    // Oversold zone
    if (prevK !== null && prevD !== null && prevK < prevD && k > d) {
      score = 90;
      desc = `StochRSI bullish cross in oversold zone (K:${k.toFixed(0)}>D:${d.toFixed(0)}) — high-prob reversal`;
    } else {
      score = 50;
      desc = `StochRSI oversold (K:${k.toFixed(0)}, D:${d.toFixed(0)}) — accumulation zone, watch for K/D cross`;
    }
  } else if (k > 80 && d > 80) {
    // Overbought zone
    if (prevK !== null && prevD !== null && prevK > prevD && k < d) {
      score = -90;
      desc = `StochRSI bearish cross in overbought zone (K:${k.toFixed(0)}<D:${d.toFixed(0)}) — high-prob reversal`;
    } else {
      score = -50;
      desc = `StochRSI overbought (K:${k.toFixed(0)}, D:${d.toFixed(0)}) — distribution zone, watch for K/D cross`;
    }
  } else {
    // Mid-range — use direction for mild signal
    const kTrend = prevK !== null ? k - prevK : 0;
    if (k > 50) {
      score = kTrend > 0 ? 35 : 15;
      desc = `StochRSI bullish bias (K:${k.toFixed(0)}, ${kTrend > 0 ? 'rising' : 'pulling back'})`;
    } else {
      score = kTrend < 0 ? -35 : -15;
      desc = `StochRSI bearish bias (K:${k.toFixed(0)}, ${kTrend < 0 ? 'falling' : 'bouncing'})`;
    }
  }

  return { value: Math.max(-100, Math.min(100, score)), description: desc };
}

/**
 * EMA Trend — price vs EMA20/50/200 stack.
 * Full bull stack (price > EMA20 > EMA50 > EMA200) = high-confidence trend context.
 */
function scoreEMATrend(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 50) return { value: 0, description: 'Not enough data for EMA trend' };

  const closes = candles.map(c => c.close);
  const price  = closes[closes.length - 1];
  const ema20  = calcEMA(closes, 20);
  const ema50  = calcEMA(closes, 50);
  const ema200 = candles.length >= 200 ? calcEMA(closes, 200) : null;

  const e20 = ema20[ema20.length - 1];
  const e50 = ema50[ema50.length - 1];
  const e200 = ema200 ? ema200[ema200.length - 1] : null;

  if (e20 === null || e50 === null) return { value: 0, description: 'EMA warm-up period' };

  let score = 0;
  const parts: string[] = [];

  if (price > e20) { score += 25; parts.push('price > EMA20'); }
  else             { score -= 25; parts.push('price < EMA20'); }

  if (price > e50) { score += 25; parts.push('price > EMA50'); }
  else             { score -= 25; parts.push('price < EMA50'); }

  if (e20 > e50) { score += 30; parts.push('EMA20 > EMA50 (bullish cross)'); }
  else           { score -= 30; parts.push('EMA20 < EMA50 (bearish cross)'); }

  if (e200 !== null) {
    if (price > e200) { score += 20; parts.push('above EMA200 (macro bull)'); }
    else              { score -= 20; parts.push('below EMA200 (macro bear)'); }
  }

  return { value: Math.max(-100, Math.min(100, score)), description: parts.join(', ') };
}

/**
 * Candle Pattern — price action signals on the last 1–3 candles.
 * Pin bars and engulfing patterns at swing points have historically high accuracy.
 */
function scoreCandlePattern(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 3) return { value: 0, description: 'Not enough candles for pattern' };

  const c  = candles[candles.length - 1];
  const p  = candles[candles.length - 2];

  const body       = Math.abs(c.close - c.open);
  const range      = c.high - c.low;
  const upperWick  = c.high - Math.max(c.open, c.close);
  const lowerWick  = Math.min(c.open, c.close) - c.low;
  const isBull     = c.close >= c.open;
  const bodyRatio  = range > 0.000001 ? body / range : 0;

  // Doji — no clear signal
  if (bodyRatio < 0.08) {
    return { value: 0, description: `Doji — indecision at $${c.close.toFixed(2)}` };
  }

  // Bullish hammer / pin bar — strong lower wick rejection
  if (lowerWick > body * 2.5 && upperWick < body && isBull) {
    return {
      value: 75,
      description: `Bullish hammer — ${(lowerWick / range * 100).toFixed(0)}% lower wick rejection, strong demand`,
    };
  }

  // Bearish shooting star — strong upper wick rejection
  if (upperWick > body * 2.5 && lowerWick < body && !isBull) {
    return {
      value: -75,
      description: `Bearish shooting star — ${(upperWick / range * 100).toFixed(0)}% upper wick rejection, strong supply`,
    };
  }

  // Bullish engulfing — current bull candle fully engulfs prior bear candle
  const pBull = p.close >= p.open;
  if (isBull && !pBull && c.open <= p.close && c.close >= p.open) {
    return { value: 70, description: 'Bullish engulfing — bull candle absorbs prior bearish pressure' };
  }

  // Bearish engulfing — current bear candle fully engulfs prior bull candle
  if (!isBull && pBull && c.open >= p.close && c.close <= p.open) {
    return { value: -70, description: 'Bearish engulfing — bear candle absorbs prior bullish pressure' };
  }

  // Marubozu — strong momentum, minimal wicks
  if (bodyRatio > 0.85) {
    return { value: isBull ? 50 : -50, description: `${isBull ? 'Bullish' : 'Bearish'} marubozu — strong conviction (${(bodyRatio * 100).toFixed(0)}% body)` };
  }

  // Mild directional bias for normal candles
  return {
    value: isBull ? 15 : -15,
    description: `${isBull ? 'Bullish' : 'Bearish'} candle (body ${(bodyRatio * 100).toFixed(0)}% of range)`,
  };
}

/**
 * VWAP Position — rolling 100-candle VWAP as an institutional reference level.
 * Reclaim of VWAP from below = bullish, rejection from above = bearish.
 */
function scoreVWAP(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 20) return { value: 0, description: 'Not enough data for VWAP' };

  const window = candles.slice(-100); // rolling 100-candle VWAP
  const vwap = calcVWAP(window);
  if (!vwap) return { value: 0, description: 'VWAP calculation failed' };

  const price     = candles[candles.length - 1].close;
  const prevPrice = candles[candles.length - 2].close;
  const pct       = (price - vwap) / vwap * 100;
  const prevPct   = (prevPrice - vwap) / vwap * 100;

  let score: number;
  let desc: string;

  if (pct > 0.3 && prevPct < 0) {
    score = 70; desc = `VWAP reclaim — crossed above $${vwap.toFixed(2)} — institutional buying signal`;
  } else if (pct < -0.3 && prevPct > 0) {
    score = -70; desc = `VWAP break — crossed below $${vwap.toFixed(2)} — institutional selling signal`;
  } else if (pct > 1.5) {
    score = 45; desc = `${pct.toFixed(1)}% above VWAP ($${vwap.toFixed(2)}) — bullish`;
  } else if (pct > 0) {
    score = 20; desc = `${pct.toFixed(1)}% above VWAP ($${vwap.toFixed(2)})`;
  } else if (pct < -1.5) {
    score = -45; desc = `${Math.abs(pct).toFixed(1)}% below VWAP ($${vwap.toFixed(2)}) — bearish`;
  } else {
    score = -20; desc = `${pct.toFixed(1)}% below VWAP ($${vwap.toFixed(2)})`;
  }

  return { value: score, description: desc };
}

// ─── Main Prediction Generator ────────────────────────────────────────────────

export function generatePrediction(
  candles: Candle[],
  coin: Coin,
  timeframe: Timeframe,
  _fearGreed: FearGreed | null,
  _fundingRate: FundingRate | null,
  _news: unknown[],
  _prevDay: { high: number; low: number } | null,
): Prediction {
  const price = candles[candles.length - 1].close;
  const atr   = calcATR(candles);

  const multiTF  = scoreMultiTFAlignment(candles);
  const liquidity = scoreLiquiditySweep(candles);
  const macd      = scoreMACD(candles);
  const stochRsi  = scoreStochRSI(candles);
  const emaTrend  = scoreEMATrend(candles);
  const candle    = scoreCandlePattern(candles);
  const vwap      = scoreVWAP(candles);

  // Use learned weights if available, otherwise fall back to defaults
  const learnedWeights = resolveWeights(loadWeights());

  const signals: Signal[] = [
    { name: 'Multi-TF Alignment', value: multiTF.value,   weight: learnedWeights['Multi-TF Alignment'] ?? DEFAULT_WEIGHTS['Multi-TF Alignment'], description: multiTF.description },
    { name: 'Liquidity Sweep',    value: liquidity.value,  weight: learnedWeights['Liquidity Sweep']    ?? DEFAULT_WEIGHTS['Liquidity Sweep'],    description: liquidity.description },
    { name: 'MACD Momentum',      value: macd.value,       weight: learnedWeights['MACD Momentum']      ?? DEFAULT_WEIGHTS['MACD Momentum'],      description: macd.description },
    { name: 'Stoch RSI',          value: stochRsi.value,   weight: learnedWeights['Stoch RSI']          ?? DEFAULT_WEIGHTS['Stoch RSI'],          description: stochRsi.description },
    { name: 'EMA Trend',          value: emaTrend.value,   weight: learnedWeights['EMA Trend']          ?? DEFAULT_WEIGHTS['EMA Trend'],          description: emaTrend.description },
    { name: 'Candle Pattern',     value: candle.value,     weight: learnedWeights['Candle Pattern']     ?? DEFAULT_WEIGHTS['Candle Pattern'],     description: candle.description },
    { name: 'VWAP',               value: vwap.value,       weight: learnedWeights['VWAP']               ?? DEFAULT_WEIGHTS['VWAP'],               description: vwap.description },
  ];

  // Composite: weighted average, scaled 0–100 (50 = neutral)
  const raw = signals.reduce((sum, s) => sum + s.value * s.weight, 0);
  const composite50 = (raw + 100) / 2;

  let direction: PredictionDirection;
  let confidence: number;

  if (composite50 > 63) {
    direction  = 'LONG';
    confidence = 65 + (composite50 - 63) * (30 / 37);
  } else if (composite50 < 37) {
    direction  = 'SHORT';
    confidence = 65 + (37 - composite50) * (30 / 37);
  } else {
    direction  = 'NEUTRAL';
    confidence = 50;
  }
  confidence = Math.min(95, Math.max(50, confidence));

  // ── Hard filter: if multi-TF strongly opposes the signal, force NEUTRAL ──────
  // This is the key >60% win rate guard — never trade into a strong opposing trend.
  if (direction === 'LONG'  && multiTF.value < -40) {
    direction  = 'NEUTRAL';
    confidence = 50;
  }
  if (direction === 'SHORT' && multiTF.value > 40) {
    direction  = 'NEUTRAL';
    confidence = 50;
  }

  // Entry zone: ±0.3% from current price
  const entryZone: [number, number] = [price * 0.997, price * 1.003];

  // SL at 1.5×ATR, TP at 3×ATR (1:2 R:R minimum)
  const stopDist   = atr * 1.5;
  const targetDist = atr * 3.0;
  const stopPrice   = direction === 'LONG'  ? price - stopDist  : price + stopDist;
  const targetPrice = direction === 'LONG'  ? price + targetDist : price - targetDist;

  const topSignal = [...signals].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
  const bullBear  = direction === 'LONG' ? 'bullish' : direction === 'SHORT' ? 'bearish' : 'neutral';
  const reasoning = direction === 'NEUTRAL'
    ? `Mixed TA signals (composite ${composite50.toFixed(0)}/100). Key driver: ${topSignal.name} — ${topSignal.description}. No high-conviction setup — staying flat.`
    : `${direction} setup with ${confidence.toFixed(0)}% confidence (composite ${composite50.toFixed(0)}/100). ` +
      `Primary driver: ${topSignal.name} — ${topSignal.description}. ` +
      `ATR stop at $${stopPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}, ` +
      `target $${targetPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })} (1:2 R:R + trailing stop). ` +
      `${signals.filter(s => (direction === 'LONG' ? s.value > 0 : s.value < 0)).length}/7 signals ${bullBear}.`;

  return {
    id: `pred-${coin}-${timeframe}-${Date.now()}`,
    coin,
    timeframe,
    timestamp: Date.now(),
    direction,
    confidence,
    entryZone,
    targetPrice,
    stopPrice,
    atr,
    signals,
    reasoning,
    resolved: false,
  };
}

// ─── TF Bias Matrix ──────────────────────────────────────────────────────────

const TF_RESAMPLE: Record<Timeframe, number> = {
  '1m':  1,
  '5m':  5,
  '15m': 15,
  '1h':  60,
  '4h':  240,
  '1d':  1440,
};

function resampleByMinutes(candles: Candle[], targetMinutes: number, sourceMinutes: number): Candle[] {
  const factor = Math.round(targetMinutes / sourceMinutes);
  return resampleCandles(candles, factor);
}

export function getTFMatrix(candles: Candle[], currentTF: Timeframe): TFBias[] {
  const sourceMinutes = TF_RESAMPLE[currentTF];
  const ALL_TFS: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

  return ALL_TFS.map(tf => {
    const targetMinutes = TF_RESAMPLE[tf];
    if (targetMinutes < sourceMinutes) {
      // Can't go lower than current TF — proxy via current data
      const { direction, score } = quickTrend(candles);
      return { timeframe: tf, direction, score };
    }
    const resampled = resampleByMinutes(candles, targetMinutes, sourceMinutes);
    const { direction, score } = quickTrend(resampled);
    return { timeframe: tf, direction, score };
  });
}
