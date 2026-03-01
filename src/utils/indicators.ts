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
