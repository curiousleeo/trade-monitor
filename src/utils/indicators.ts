import type { Candle } from '../types';

/** Calculate EMA. Returns array same length as `closes`, nulls for warm-up period. */
export function calcEMA(closes: number[], period: number): (number | null)[] {
  if (closes.length < period) return closes.map(() => null);

  const k = 2 / (period + 1);
  const result: (number | null)[] = new Array(period - 1).fill(null);

  // Seed with SMA of first `period` values
  const seed = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(seed);

  for (let i = period; i < closes.length; i++) {
    result.push(closes[i] * k + result[result.length - 1]! * (1 - k));
  }

  return result;
}

/** Calculate Bollinger Bands (SMA ± n×σ). Returns array same length as `closes`. */
export function calcBollingerBands(
  closes: number[],
  period = 20,
  mult = 2,
): { upper: number | null; middle: number | null; lower: number | null }[] {
  return closes.map((_, i) => {
    if (i < period - 1) return { upper: null, middle: null, lower: null };
    const slice = closes.slice(i - period + 1, i + 1);
    const sma = slice.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(slice.reduce((s, v) => s + (v - sma) ** 2, 0) / period);
    return { upper: sma + mult * std, middle: sma, lower: sma - mult * std };
  });
}

/** Calculate RSI (Wilder's smoothing). Returns array same length as `closes`. */
export function calcRSI(closes: number[], period = 14): (number | null)[] {
  if (closes.length <= period) return closes.map(() => null);

  const result: (number | null)[] = new Array(period).fill(null);

  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) avgGain += diff;
    else avgLoss += Math.abs(diff);
  }
  avgGain /= period;
  avgLoss /= period;

  const rsi = (ag: number, al: number) => 100 - 100 / (1 + ag / (al || 0.0001));
  result.push(rsi(avgGain, avgLoss));

  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? Math.abs(diff) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    result.push(rsi(avgGain, avgLoss));
  }

  return result;
}

/**
 * VWAP anchored to the provided candle window (rolling VWAP).
 * For 24/7 crypto, a rolling 100-candle VWAP acts as an institutional mean-reversion level.
 */
export function calcVWAP(candles: Candle[]): number | null {
  if (candles.length === 0) return null;
  let tpv = 0;
  let vol = 0;
  for (const c of candles) {
    const tp = (c.high + c.low + c.close) / 3;
    tpv += tp * c.volume;
    vol += c.volume;
  }
  return vol > 0 ? tpv / vol : null;
}

/**
 * Stochastic RSI — RSI within its own range, smoothed.
 * Returns K and D lines (0–100). High probability reversal when K/D cross in overbought/oversold.
 */
export function calcStochRSI(
  closes: number[],
  rsiPeriod = 14,
  stochPeriod = 14,
): { k: number | null; d: number | null }[] {
  const rsiArr = calcRSI(closes, rsiPeriod);
  const result: { k: number | null; d: number | null }[] = closes.map(() => ({ k: null, d: null }));

  // Collect valid RSI values with original indices
  const validRsi: { val: number; origIdx: number }[] = [];
  rsiArr.forEach((v, i) => {
    if (v !== null) validRsi.push({ val: v as number, origIdx: i });
  });

  // Raw StochRSI K
  const rawK: number[] = validRsi.map((_, j) => {
    if (j < stochPeriod - 1) return NaN;
    const window = validRsi.slice(j - stochPeriod + 1, j + 1).map(x => x.val);
    const lo = Math.min(...window);
    const hi = Math.max(...window);
    return hi === lo ? 50 : ((validRsi[j].val - lo) / (hi - lo)) * 100;
  });

  // Smooth K (3-period SMA)
  const smoothK: number[] = rawK.map((_, j) => {
    if (j < stochPeriod + 1) return NaN;
    const w = [rawK[j - 2], rawK[j - 1], rawK[j]];
    if (w.some(isNaN)) return NaN;
    return (w[0] + w[1] + w[2]) / 3;
  });

  // D = 3-period SMA of smoothed K, write back to original indices
  for (let j = 0; j < validRsi.length; j++) {
    if (isNaN(smoothK[j])) continue;
    const k = smoothK[j];
    let d: number | null = null;
    if (j >= 2 && !isNaN(smoothK[j - 1]) && !isNaN(smoothK[j - 2])) {
      d = (smoothK[j - 2] + smoothK[j - 1] + smoothK[j]) / 3;
    }
    result[validRsi[j].origIdx] = { k, d };
  }

  return result;
}

/**
 * MACD (12, 26, 9).
 * Returns macd line, signal line, and histogram for each candle.
 * Nulls during warm-up period.
 */
export function calcMACD(
  closes: number[],
  fast = 12,
  slow = 26,
  signalPeriod = 9,
): { macd: number | null; signal: number | null; histogram: number | null }[] {
  const ema12 = calcEMA(closes, fast);
  const ema26 = calcEMA(closes, slow);

  // MACD line = EMA12 − EMA26
  const macdLine: (number | null)[] = closes.map((_, i) => {
    const a = ema12[i];
    const b = ema26[i];
    return a !== null && b !== null ? a - b : null;
  });

  // EMA(9) of MACD line — computed on valid values only, mapped back
  const validMacd: number[] = [];
  const validIdx: number[] = [];
  macdLine.forEach((v, i) => {
    if (v !== null) { validMacd.push(v); validIdx.push(i); }
  });

  const signalLine = calcEMA(validMacd, signalPeriod);

  const result: { macd: number | null; signal: number | null; histogram: number | null }[] =
    closes.map(() => ({ macd: null, signal: null, histogram: null }));

  validIdx.forEach((origIdx, j) => {
    const m = validMacd[j];
    const s = signalLine[j];
    result[origIdx] = { macd: m, signal: s, histogram: s !== null ? m - s : null };
  });

  return result;
}
