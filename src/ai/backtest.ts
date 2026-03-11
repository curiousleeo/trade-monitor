/**
 * APEX Backtest Engine — Realistic Model Evaluation
 *
 * Realistic assumptions:
 *   - Entry at NEXT candle's open after signal (no lookahead)
 *   - Binance taker fee: 0.1% per side
 *   - Slippage: 0.05% on entry (conservative market order)
 *   - TP/SL checked against candle high/low (not close)
 *   - If both SL and TP are hit in the same candle → SL wins (conservative)
 *   - Post-SL cooldown: 3 candles before re-entering
 *   - Max 4 concurrent positions
 *   - Predict every 2 candles (matches live system behavior)
 *   - Walk-forward split: first 60% = in-sample, last 40% = out-of-sample
 *
 * No sentiment, news, or funding data in backtest → pure TA model honesty.
 */

import { Candle, Coin, Signal, Timeframe } from '../types';
import { generatePrediction } from './engine';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BacktestTrade {
  entryCandle:   number;
  exitCandle:    number;
  direction:     'LONG' | 'SHORT';
  entryPrice:    number;
  exitPrice:     number;
  stopLoss:      number;
  takeProfit:    number;
  confidence:    number;
  pnlPct:        number;   // net of round-trip fees + slippage
  result:        'WIN' | 'LOSS';
  holdingCandles: number;
  signals:       Signal[];
  exitReason:    'TP' | 'SL' | 'END';
}

export interface SignalAttribution {
  name:                 string;
  weight:               number;
  avgAlignedValue:      number;  // avg signal value when it agrees with direction (wins)
  avgMisalignedValue:   number;  // avg signal value when it disagrees (losses)
  winRateWhenAligned:   number;  // win rate when signal agrees with trade direction
  winRateWhenOpposed:   number;  // win rate when signal opposes trade direction
  alignedCount:         number;
}

export interface ConfidenceBucket {
  label:     string;
  winRate:   number;
  count:     number;
  avgReturn: number;
}

export interface WalkForwardResult {
  inSampleTrades:   number;
  inSampleWinRate:  number;
  oosTrades:        number;
  oosWinRate:       number;
  oosReturnPct:     number;
  degradation:      number;  // inSample - oos winRate (measures overfitting)
}

export interface BacktestResult {
  // Core performance
  trades:         BacktestTrade[];
  winRate:        number;
  profitFactor:   number;
  maxDrawdownPct: number;
  totalReturnPct: number;
  sharpeRatio:    number;  // annualised (based on per-trade returns)
  calmarRatio:    number;  // totalReturn / maxDrawdown

  // Trade quality
  avgWinPct:          number;
  avgLossPct:         number;
  avgRR:              number;   // actual average R:R achieved
  avgHoldingCandles:  number;
  tradesPerHundredCandles: number;

  // Signal attribution (which signals predict wins)
  signalAttribution: SignalAttribution[];

  // Confidence calibration
  confidenceBuckets: ConfidenceBucket[];

  // Walk-forward honesty check
  walkForward: WalkForwardResult;

  // Equity curve (balance after each closed trade)
  equityCurve: number[];
  startBalance: number;
  finalBalance: number;
}

// ─── Engine ───────────────────────────────────────────────────────────────────

interface OpenPosition {
  direction:   'LONG' | 'SHORT';
  entryPrice:  number;
  stopLoss:    number;
  takeProfit:  number;
  confidence:  number;
  entryCandle: number;
  signals:     Signal[];
  atr:         number;
}

export function runBacktest(
  candles:      Candle[],
  coin:         Coin,
  timeframe:    Timeframe,
  config: {
    feeRate?:        number;   // default 0.001 (0.1% Binance taker)
    slippagePct?:    number;   // default 0.0005 (0.05%)
    minConfidence?:  number;   // default 65
    warmupCandles?:  number;   // default 100 (enough for EMA200 + indicators)
    maxConcurrent?:  number;   // default 4
    postSLCooldown?: number;   // default 3 candles
    predInterval?:   number;   // default 2 (predict every N candles, matches live)
  } = {},
): BacktestResult {
  const {
    feeRate       = 0.001,
    slippagePct   = 0.0005,
    minConfidence = 65,
    warmupCandles = 100,
    maxConcurrent = 4,
    postSLCooldown = 3,
    predInterval  = 2,
  } = config;

  // Total round-trip cost: entry fee + exit fee + slippage (approximate)
  const ROUND_TRIP = feeRate * 2 + slippagePct;

  const START_BALANCE = 1000;
  let balance         = START_BALANCE;
  const equityCurve   = [balance];

  const closedTrades: BacktestTrade[] = [];
  const openPositions: OpenPosition[] = [];
  let cooldownUntil = 0; // candle index when cooldown expires
  let lastPredCandle = 0; // rate-limit predictions

  for (let i = warmupCandles; i < candles.length; i++) {
    const cur = candles[i];

    // ── 1. Update open positions: check TP/SL on current candle ──────────────
    const toClose: number[] = [];

    for (let p = 0; p < openPositions.length; p++) {
      const pos = openPositions[p];
      let exitPrice: number | null = null;
      let exitReason: 'TP' | 'SL' | null = null;

      if (pos.direction === 'LONG') {
        // Conservative: if both SL and TP hit in same candle, assume SL hit first
        if (cur.low <= pos.stopLoss) {
          exitPrice  = pos.stopLoss;
          exitReason = 'SL';
        } else if (cur.high >= pos.takeProfit) {
          exitPrice  = pos.takeProfit;
          exitReason = 'TP';
        }
      } else {
        if (cur.high >= pos.stopLoss) {
          exitPrice  = pos.stopLoss;
          exitReason = 'SL';
        } else if (cur.low <= pos.takeProfit) {
          exitPrice  = pos.takeProfit;
          exitReason = 'TP';
        }
      }

      if (exitPrice !== null && exitReason !== null) {
        const rawPnl = pos.direction === 'LONG'
          ? (exitPrice - pos.entryPrice) / pos.entryPrice
          : (pos.entryPrice - exitPrice) / pos.entryPrice;
        const pnlPct = rawPnl - ROUND_TRIP;

        closedTrades.push({
          entryCandle:    pos.entryCandle,
          exitCandle:     i,
          direction:      pos.direction,
          entryPrice:     pos.entryPrice,
          exitPrice,
          stopLoss:       pos.stopLoss,
          takeProfit:     pos.takeProfit,
          confidence:     pos.confidence,
          pnlPct,
          result:         pnlPct > 0 ? 'WIN' : 'LOSS',
          holdingCandles: i - pos.entryCandle,
          signals:        pos.signals,
          exitReason,
        });

        balance *= (1 + pnlPct);
        equityCurve.push(balance);

        if (exitReason === 'SL') cooldownUntil = i + postSLCooldown;
        toClose.push(p);
      }
    }

    // Remove closed positions (reverse order to preserve indices)
    for (let k = toClose.length - 1; k >= 0; k--) {
      openPositions.splice(toClose[k], 1);
    }

    // ── 2. Can we enter a new position? ──────────────────────────────────────
    const canEnter =
      i < candles.length - 1 &&          // need next candle for entry
      openPositions.length < maxConcurrent &&
      i >= cooldownUntil &&
      (i - lastPredCandle) >= predInterval; // respect prediction rate limit

    if (canEnter) {
      lastPredCandle = i;

      // Generate prediction on data available up to (and including) candle i
      const pred = generatePrediction(
        candles.slice(0, i + 1),
        coin,
        timeframe,
        null,   // no fearGreed in backtest
        null,   // no fundingRate
        [],     // no news
        null,   // no prevDay
      );

      if (pred.direction !== 'NEUTRAL' && pred.confidence >= minConfidence) {
        const nextOpen = candles[i + 1].open;
        // Apply slippage: entry is slightly worse than signal
        const slip = pred.direction === 'LONG' ? 1 + slippagePct : 1 - slippagePct;
        const entryPrice = nextOpen * slip;

        openPositions.push({
          direction:   pred.direction,
          entryPrice,
          stopLoss:    pred.stopPrice,
          takeProfit:  pred.targetPrice,
          confidence:  pred.confidence,
          entryCandle: i + 1,
          signals:     pred.signals,
          atr:         pred.atr,
        });
      }
    }
  }

  // ── 3. Close any remaining positions at last candle's close ─────────────────
  const lastCandle = candles[candles.length - 1];
  for (const pos of openPositions) {
    const exitPrice = lastCandle.close;
    const rawPnl = pos.direction === 'LONG'
      ? (exitPrice - pos.entryPrice) / pos.entryPrice
      : (pos.entryPrice - exitPrice) / pos.entryPrice;
    const pnlPct = rawPnl - ROUND_TRIP;

    closedTrades.push({
      entryCandle:    pos.entryCandle,
      exitCandle:     candles.length - 1,
      direction:      pos.direction,
      entryPrice:     pos.entryPrice,
      exitPrice,
      stopLoss:       pos.stopLoss,
      takeProfit:     pos.takeProfit,
      confidence:     pos.confidence,
      pnlPct,
      result:         pnlPct > 0 ? 'WIN' : 'LOSS',
      holdingCandles: candles.length - 1 - pos.entryCandle,
      signals:        pos.signals,
      exitReason:     'END',
    });
    balance *= (1 + pnlPct);
    equityCurve.push(balance);
  }

  // ─── Statistics ────────────────────────────────────────────────────────────

  const wins   = closedTrades.filter(t => t.result === 'WIN');
  const losses = closedTrades.filter(t => t.result === 'LOSS');
  const n      = closedTrades.length;

  const winRate = n > 0 ? wins.length / n : 0;

  const grossProfit = wins.reduce((s, t) => s + t.pnlPct, 0);
  const grossLoss   = Math.abs(losses.reduce((s, t) => s + t.pnlPct, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 99 : 0);

  // Max drawdown from equity curve
  let peak = equityCurve[0];
  let maxDD = 0;
  for (const v of equityCurve) {
    if (v > peak) peak = v;
    const dd = (peak - v) / peak;
    if (dd > maxDD) maxDD = dd;
  }

  const totalReturnPct = (balance - START_BALANCE) / START_BALANCE;

  // Sharpe: per-trade return mean/std, scaled to ~annual (252 trading days / avg hold)
  const returns  = closedTrades.map(t => t.pnlPct);
  const meanR    = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
  const stdR     = returns.length > 1
    ? Math.sqrt(returns.reduce((s, r) => s + (r - meanR) ** 2, 0) / returns.length)
    : 0;
  const avgHold  = n > 0 ? closedTrades.reduce((s, t) => s + t.holdingCandles, 0) / n : 1;
  // Approximate trades per year based on holding time (rough annualisation)
  const tradesPerYear = avgHold > 0 ? (252 * 24) / avgHold : 100;
  const sharpeRatio   = stdR > 0 ? (meanR / stdR) * Math.sqrt(tradesPerYear) : 0;

  const calmarRatio = maxDD > 0 ? totalReturnPct / maxDD : (totalReturnPct > 0 ? 99 : 0);

  const avgWinPct  = wins.length > 0   ? wins.reduce((s, t) => s + t.pnlPct, 0) / wins.length : 0;
  const avgLossPct = losses.length > 0 ? losses.reduce((s, t) => s + t.pnlPct, 0) / losses.length : 0;
  const avgRR      = avgLossPct !== 0  ? Math.abs(avgWinPct / avgLossPct) : 0;
  const tradesPerHundredCandles = (n / candles.length) * 100;

  // ─── Signal Attribution ────────────────────────────────────────────────────
  // For each signal, check: when signal agreed with direction → what was win rate?

  const signalNames = closedTrades[0]?.signals.map(s => s.name) ?? [];

  const signalAttribution: SignalAttribution[] = signalNames.map(name => {
    const weight = closedTrades[0]?.signals.find(s => s.name === name)?.weight ?? 0;

    const tradesWithSig = closedTrades.filter(t => t.signals.some(s => s.name === name));

    const aligned  = tradesWithSig.filter(t => {
      const sig = t.signals.find(s => s.name === name)!;
      return (t.direction === 'LONG' && sig.value > 10) || (t.direction === 'SHORT' && sig.value < -10);
    });
    const opposed  = tradesWithSig.filter(t => {
      const sig = t.signals.find(s => s.name === name)!;
      return (t.direction === 'LONG' && sig.value < -10) || (t.direction === 'SHORT' && sig.value > 10);
    });

    const alignedWins   = aligned.filter(t => t.result === 'WIN');
    const opposedWins   = opposed.filter(t => t.result === 'WIN');

    // Average signal value in winning trades (direction-normalised)
    const inWins  = wins.map(t => {
      const sig = t.signals.find(s => s.name === name);
      if (!sig) return null;
      return t.direction === 'LONG' ? sig.value : -sig.value;
    }).filter((v): v is number => v !== null);

    const inLosses = losses.map(t => {
      const sig = t.signals.find(s => s.name === name);
      if (!sig) return null;
      return t.direction === 'LONG' ? sig.value : -sig.value;
    }).filter((v): v is number => v !== null);

    return {
      name,
      weight,
      avgAlignedValue:    inWins.length  > 0 ? inWins.reduce((a, b)  => a + b, 0) / inWins.length  : 0,
      avgMisalignedValue: inLosses.length > 0 ? inLosses.reduce((a, b) => a + b, 0) / inLosses.length : 0,
      winRateWhenAligned:  aligned.length > 0 ? alignedWins.length / aligned.length : 0,
      winRateWhenOpposed:  opposed.length > 0 ? opposedWins.length / opposed.length : 0,
      alignedCount: aligned.length,
    };
  });

  // ─── Confidence Calibration ────────────────────────────────────────────────
  // Does higher confidence → higher win rate? (should be yes for a well-calibrated model)

  const bucketDefs = [
    { min: 65, max: 70, label: '65–70%' },
    { min: 70, max: 75, label: '70–75%' },
    { min: 75, max: 80, label: '75–80%' },
    { min: 80, max: 85, label: '80–85%' },
    { min: 85, max: 96, label: '85–95%' },
  ];

  const confidenceBuckets: ConfidenceBucket[] = bucketDefs.map(b => {
    const inBucket = closedTrades.filter(t => t.confidence >= b.min && t.confidence < b.max);
    const bWins    = inBucket.filter(t => t.result === 'WIN');
    return {
      label:     b.label,
      winRate:   inBucket.length > 0 ? bWins.length / inBucket.length : 0,
      count:     inBucket.length,
      avgReturn: inBucket.length > 0
        ? inBucket.reduce((s, t) => s + t.pnlPct, 0) / inBucket.length
        : 0,
    };
  });

  // ─── Walk-Forward Honesty Check ────────────────────────────────────────────
  // The model was designed on the same indicator logic tested here.
  // Out-of-sample results = last 40% of trades (by entry candle time).

  const sorted    = [...closedTrades].sort((a, b) => a.entryCandle - b.entryCandle);
  const splitAt   = Math.floor(sorted.length * 0.6);
  const inSample  = sorted.slice(0, splitAt);
  const oosSample = sorted.slice(splitAt);

  const inSampleWins  = inSample.filter(t => t.result === 'WIN');
  const oosWins       = oosSample.filter(t => t.result === 'WIN');

  const inSampleWinRate = inSample.length  > 0 ? inSampleWins.length  / inSample.length  : 0;
  const oosWinRate      = oosSample.length > 0 ? oosWins.length       / oosSample.length : 0;

  // OOS return: compound the OOS trades from $1000
  let oosBalance = 1000;
  for (const t of oosSample) oosBalance *= (1 + t.pnlPct);

  const walkForward: WalkForwardResult = {
    inSampleTrades:  inSample.length,
    inSampleWinRate,
    oosTrades:       oosSample.length,
    oosWinRate,
    oosReturnPct:    (oosBalance - 1000) / 1000,
    degradation:     inSampleWinRate - oosWinRate,
  };

  return {
    trades:          closedTrades,
    winRate,
    profitFactor,
    maxDrawdownPct:  maxDD,
    totalReturnPct,
    sharpeRatio,
    calmarRatio,
    avgWinPct,
    avgLossPct,
    avgRR,
    avgHoldingCandles: avgHold,
    tradesPerHundredCandles,
    signalAttribution,
    confidenceBuckets,
    walkForward,
    equityCurve,
    startBalance:    START_BALANCE,
    finalBalance:    balance,
  };
}
