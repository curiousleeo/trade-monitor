/**
 * APEX Prediction Engine
 * Pure functions — no React, no side effects.
 * Generates market predictions and TF bias matrix.
 */

import { Candle, Coin, FearGreed, FundingRate, NewsItem, Prediction, PredictionDirection, Signal, TFBias, Timeframe } from '../types';
import { calcEMA, calcRSI } from '../utils/indicators';
import { scoreSentiment } from '../utils/sentiment';

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
  // Wilder's smoothing
  let atr = trs.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < trs.length; i++) {
    atr = (atr * (period - 1) + trs[i]) / period;
  }
  return atr;
}

// ─── Individual Scorers (each returns -100 to +100) ──────────────────────────

function scoreTrend(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 50) return { value: 0, description: 'Not enough data for trend analysis' };

  const closes = candles.map(c => c.close);
  const price = closes[closes.length - 1];
  const ema20arr = calcEMA(closes, 20);
  const ema50arr = calcEMA(closes, 50);
  const ema200arr = candles.length >= 200 ? calcEMA(closes, 200) : null;

  const ema20 = ema20arr[ema20arr.length - 1];
  const ema50 = ema50arr[ema50arr.length - 1];
  const ema200 = ema200arr ? ema200arr[ema200arr.length - 1] : null;

  if (ema20 === null || ema50 === null) return { value: 0, description: 'EMA warm-up period' };

  let score = 0;
  const parts: string[] = [];

  // Price vs EMAs
  if (price > ema20) { score += 25; parts.push('price > EMA20'); }
  else { score -= 25; parts.push('price < EMA20'); }

  if (price > ema50) { score += 25; parts.push('price > EMA50'); }
  else { score -= 25; parts.push('price < EMA50'); }

  // EMA alignment
  if (ema20 > ema50) { score += 30; parts.push('EMA20 > EMA50 (bullish cross)'); }
  else { score -= 30; parts.push('EMA20 < EMA50 (bearish cross)'); }

  if (ema200 !== null) {
    if (price > ema200) { score += 20; parts.push('price above EMA200 (macro bull)'); }
    else { score -= 20; parts.push('price below EMA200 (macro bear)'); }
  }

  return {
    value: Math.max(-100, Math.min(100, score)),
    description: parts.join(', '),
  };
}

function scoreRSIMomentum(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 16) return { value: 0, description: 'Not enough data for RSI' };
  const closes = candles.map(c => c.close);
  const rsiArr = calcRSI(closes, 14);
  const rsi = rsiArr[rsiArr.length - 1];
  const prevRsi = rsiArr[rsiArr.length - 2];

  if (rsi === null || prevRsi === null) return { value: 0, description: 'RSI warm-up period' };

  const momentum = rsi - prevRsi; // positive = gaining strength
  let score: number;
  let desc: string;

  if (rsi > 70) {
    score = 30 + (rsi - 70) * 0.5; // still bullish but overbought, cap enthusiasm
    desc = `RSI ${rsi.toFixed(1)} (overbought, watch for reversal)`;
  } else if (rsi > 55) {
    score = 50 + (rsi - 55) * 3.3;
    desc = `RSI ${rsi.toFixed(1)} (bullish momentum zone)`;
  } else if (rsi > 45) {
    score = momentum > 0 ? 20 : -20;
    desc = `RSI ${rsi.toFixed(1)} (neutral, ${momentum > 0 ? 'gaining' : 'losing'} momentum)`;
  } else if (rsi > 30) {
    score = -50 - (55 - rsi) * 3.3;
    desc = `RSI ${rsi.toFixed(1)} (bearish momentum zone)`;
  } else {
    score = -30 + (30 - rsi) * 0.5; // oversold bounce potential
    desc = `RSI ${rsi.toFixed(1)} (oversold, potential reversal)`;
  }

  return { value: Math.max(-100, Math.min(100, score)), description: desc };
}

function scoreVolume(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 21) return { value: 0, description: 'Not enough data for volume analysis' };

  const recent = candles.slice(-21);
  const last = recent[recent.length - 1];
  const avgVolume = recent.slice(0, -1).reduce((sum, c) => sum + c.volume, 0) / 20;

  if (avgVolume === 0) return { value: 0, description: 'Zero average volume' };

  const ratio = last.volume / avgVolume;
  const isBullishCandle = last.close >= last.open;

  let score: number;
  let desc: string;

  if (ratio >= 2.0) {
    score = isBullishCandle ? 80 : -80;
    desc = `Volume spike ${ratio.toFixed(1)}×avg — ${isBullishCandle ? 'bullish' : 'bearish'} conviction`;
  } else if (ratio >= 1.3) {
    score = isBullishCandle ? 40 : -40;
    desc = `Above-avg volume ${ratio.toFixed(1)}×avg — ${isBullishCandle ? 'bullish' : 'bearish'}`;
  } else if (ratio < 0.5) {
    score = 0;
    desc = `Low volume ${ratio.toFixed(1)}×avg — weak conviction`;
  } else {
    score = 0;
    desc = `Normal volume ${ratio.toFixed(1)}×avg`;
  }

  return { value: Math.max(-100, Math.min(100, score)), description: desc };
}

function scoreSentimentComposite(
  news: NewsItem[],
  fearGreed: FearGreed | null,
): { value: number; description: string } {
  const now = Date.now() / 1000;
  // 4h window — captures macro events (reg/hack/exchange) which may break hours before price reacts
  const recentNews = news.filter(n => now - n.publishedAt < 14400);

  // High-impact categories get 3× weight (Hack/Regulation can move prices 10%+)
  const HIGH_IMPACT = ['Hack', 'Regulation', 'Exchange', 'Stablecoin', 'MacroEcon', 'GeoPolitic'];

  let weightedScore = 0;
  let totalWeight = 0;
  let bullCount = 0;
  let bearCount = 0;

  recentNews.forEach(n => {
    const age = now - n.publishedAt;
    // Time decay: < 30 min = 3×, 30 min–2h = 1.5×, 2h–4h = 1×
    const timeWeight = age < 1800 ? 3 : age < 7200 ? 1.5 : 1;
    const categoryWeight = HIGH_IMPACT.some(cat => n.categories.includes(cat)) ? 3 : 1;
    const w = timeWeight * categoryWeight;

    const s = scoreSentiment(n.title);
    if (s === 'bullish') { weightedScore += w; bullCount++; }
    if (s === 'bearish') { weightedScore -= w; bearCount++; }
    totalWeight += w;
  });

  const newsNorm = totalWeight > 0
    ? Math.max(-100, Math.min(100, (weightedScore / totalWeight) * 100))
    : 0;

  const fg = fearGreed?.value ?? 50;
  // Map F&G 0-100 to -100/+100 with 50 as neutral
  const fgScore = (fg - 50) * 2;

  const combined = newsNorm * 0.6 + fgScore * 0.4;
  const parts: string[] = [];
  if (recentNews.length > 0) parts.push(`${bullCount} bullish / ${bearCount} bearish news (4h)`);
  if (fearGreed) parts.push(`Fear & Greed ${fg} (${fearGreed.label})`);

  return {
    value: Math.max(-100, Math.min(100, combined)),
    description: parts.length ? parts.join(', ') : 'No recent news data',
  };
}

function scoreStructure(
  price: number,
  prevDay: { high: number; low: number } | null,
): { value: number; description: string } {
  if (!prevDay) return { value: 0, description: 'No previous day data' };

  const { high, low } = prevDay;
  const range = high - low;

  if (price > high) {
    const breakoutStrength = Math.min(100, ((price - high) / range) * 200);
    return { value: 60 + breakoutStrength * 0.4, description: `Breaking above prev day high $${high.toLocaleString()}` };
  } else if (price < low) {
    const breakdownStrength = Math.min(100, ((low - price) / range) * 200);
    return { value: -(60 + breakdownStrength * 0.4), description: `Breaking below prev day low $${low.toLocaleString()}` };
  } else {
    // Inside range — position within range
    const position = (price - low) / range; // 0 = at low, 1 = at high
    const score = (position - 0.5) * 60; // -30 to +30
    return { value: score, description: `Inside prev day range (${(position * 100).toFixed(0)}% from low)` };
  }
}

function scoreFundingRate(fundingRate: FundingRate | null): { value: number; description: string } {
  if (!fundingRate) return { value: 0, description: 'No funding rate data' };
  const { rate } = fundingRate;
  // Positive funding = longs pay shorts = market bullish (trend following)
  // Extreme positive funding → caution (over-leveraged longs)
  const pct = rate * 100;
  let score: number;
  let desc: string;

  if (pct > 0.1) {
    score = 40; desc = `Funding ${pct.toFixed(4)}% (high — longs dominant, caution)`;
  } else if (pct > 0) {
    score = 60; desc = `Funding +${pct.toFixed(4)}% (longs pay, bullish sentiment)`;
  } else if (pct < -0.1) {
    score = -40; desc = `Funding ${pct.toFixed(4)}% (high negative — shorts dominant, caution)`;
  } else {
    score = -60; desc = `Funding ${pct.toFixed(4)}% (shorts pay, bearish sentiment)`;
  }

  return { value: score, description: desc };
}

// ─── Higher TF Trend (4× resample) ───────────────────────────────────────────
// Penalises entries that go against the higher timeframe structure.

function scoreHigherTF(candles: Candle[]): { value: number; description: string } {
  if (candles.length < 80) return { value: 0, description: 'Not enough data for higher TF' };
  const factor = 4;
  const resampled: Candle[] = [];
  for (let i = 0; i + factor <= candles.length; i += factor) {
    const slice = candles.slice(i, i + factor);
    resampled.push({
      time:   slice[0].time,
      open:   slice[0].open,
      high:   Math.max(...slice.map(c => c.high)),
      low:    Math.min(...slice.map(c => c.low)),
      close:  slice[slice.length - 1].close,
      volume: slice.reduce((s, c) => s + c.volume, 0),
    });
  }
  const result = scoreTrend(resampled);
  return { value: result.value, description: `4×TF — ${result.description}` };
}

// ─── Main Prediction Generator ────────────────────────────────────────────────

export function generatePrediction(
  candles: Candle[],
  coin: Coin,
  timeframe: Timeframe,
  fearGreed: FearGreed | null,
  fundingRate: FundingRate | null,
  news: NewsItem[],
  prevDay: { high: number; low: number } | null,
): Prediction {
  const price = candles[candles.length - 1].close;
  const atr = calcATR(candles);

  const trend     = scoreTrend(candles);
  const higherTF  = scoreHigherTF(candles);
  const momentum  = scoreRSIMomentum(candles);
  const volume    = scoreVolume(candles);
  const sentiment = scoreSentimentComposite(news, fearGreed);
  const structure = scoreStructure(price, prevDay);
  const funding   = scoreFundingRate(fundingRate);

  // Weights: current EMA 20%, higher TF 15%, RSI 22%, Sentiment 18%, Volume 10%, Structure 10%, Funding 5%
  const signals: Signal[] = [
    { name: 'EMA Trend',        value: trend.value,     weight: 0.20, description: trend.description },
    { name: 'Higher TF',        value: higherTF.value,  weight: 0.15, description: higherTF.description },
    { name: 'RSI Momentum',     value: momentum.value,  weight: 0.22, description: momentum.description },
    { name: 'Sentiment',        value: sentiment.value, weight: 0.18, description: sentiment.description },
    { name: 'Volume',           value: volume.value,    weight: 0.10, description: volume.description },
    { name: 'Market Structure', value: structure.value, weight: 0.10, description: structure.description },
    { name: 'Funding Bias',     value: funding.value,   weight: 0.05, description: funding.description },
  ];

  // Composite: weighted average of signal values, scaled to 0–100 (50 = neutral)
  const raw = signals.reduce((sum, s) => sum + s.value * s.weight, 0);
  const composite50 = (raw + 100) / 2; // map -100..+100 → 0..100

  let direction: PredictionDirection;
  let confidence: number;
  // Raised threshold to 68/32 — only enter on stronger, more aligned setups
  if (composite50 > 68) {
    direction = 'LONG';
    // Map composite50 [68, 100] → confidence [65, 95]
    confidence = 65 + (composite50 - 68) * (30 / 32);
  } else if (composite50 < 32) {
    direction = 'SHORT';
    confidence = 65 + (32 - composite50) * (30 / 32);
  } else {
    direction = 'NEUTRAL';
    confidence = 50;
  }
  confidence = Math.min(95, Math.max(50, confidence));

  // Entry zone: ±0.3% from current price
  const entryZone: [number, number] = [price * 0.997, price * 1.003];

  // SL at 1.5×ATR, TP at 3×ATR (1:2 R:R) — trailing stop captures moves beyond TP
  const stopDist   = atr * 1.5;
  const targetDist = atr * 3.0;

  const stopPrice   = direction === 'LONG'  ? price - stopDist  : price + stopDist;
  const targetPrice = direction === 'LONG'  ? price + targetDist : price - targetDist;

  // Reasoning paragraph
  const bullBear = direction === 'LONG' ? 'bullish' : direction === 'SHORT' ? 'bearish' : 'neutral';
  const topSignal = [...signals].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
  const reasoning = direction === 'NEUTRAL'
    ? `Mixed signals (composite ${composite50.toFixed(0)}/100). Primary driver: ${topSignal.name} (${topSignal.description}). No high-confidence trade setup — staying flat.`
    : `${direction} setup with ${confidence.toFixed(0)}% confidence (composite ${composite50.toFixed(0)}/100). ` +
      `Primary driver: ${topSignal.name} — ${topSignal.description}. ` +
      `ATR-based stop at $${stopPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })} ` +
      `with target $${targetPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })} (R:R 1:2, trailing stop active). ` +
      `Overall ${bullBear} bias confirmed by ${signals.filter(s => (direction === 'LONG' ? s.value > 0 : s.value < 0)).length}/6 signals.`;

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
// Simulate higher timeframes by resampling the existing candle set (no extra API calls)

const TF_RESAMPLE: Record<Timeframe, number> = {
  '1m':  1,
  '5m':  5,
  '15m': 15,
  '1h':  60,
  '4h':  240,
  '1d':  1440,
};

function resampleCandles(candles: Candle[], targetMinutes: number, sourceMinutes: number): Candle[] {
  const factor = Math.round(targetMinutes / sourceMinutes);
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

export function getTFMatrix(candles: Candle[], currentTF: Timeframe): TFBias[] {
  const sourceMinutes = TF_RESAMPLE[currentTF];
  const ALL_TFS: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

  return ALL_TFS.map(tf => {
    const targetMinutes = TF_RESAMPLE[tf];
    if (targetMinutes < sourceMinutes) {
      // Can't go to lower TF without more granular data — use current data trend
      const { direction, score } = quickTrend(candles);
      return { timeframe: tf, direction, score };
    }
    const resampled = resampleCandles(candles, targetMinutes, sourceMinutes);
    const { direction, score } = quickTrend(resampled);
    return { timeframe: tf, direction, score };
  });
}
