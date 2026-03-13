/**
 * APEX AI Trader Hook — v2
 *
 * Rate-limiting rules (prevents spam trades + storage abuse):
 *   • Predictions: once every PRED_CANDLE_GAP candles per coin (not every close)
 *   • Trade entry: min ENTRY_COOLDOWN_CANDLES gap between any new trade
 *   • Post-SL cooldown: POSTSL_COOLDOWN_CANDLES before re-entering same coin
 *   • Max drawdown: trading halts if balance < startBalance * (1 - MAX_DD)
 *   • Concurrent cap: MAX_OPEN positions across all coins
 *
 * Storage: writes only on trade open / trade close (never on ticks or predictions).
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  Candle, Coin, FearGreed, FundingRate,
  LogEntry, Prediction, TFBias, TickerData, Trade, Timeframe,
} from '../types';
import { generatePrediction, getTFMatrix } from '../ai/engine';
import {
  StoredPortfolio,
  loadPortfolio, loadHistory,
  onTradeOpened, onTradeClosed, resetStorage,
} from '../lib/db';
import {
  loadWeights, learnFromTrades, shouldLearn, resetWeights,
  LearnedWeights,
} from '../ai/learner';

// ─── Config ──────────────────────────────────────────────────────────────────

const COINS: Coin[] = [
  'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
  'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
  'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ',
  'PAXG',
];
const START_BALANCE        = 1000;
const MIN_CONFIDENCE       = 65;     // min score to enter
const MAX_OPEN             = 4;      // max concurrent positions (4 with 21 coins scanning)
const PRED_CANDLE_GAP      = 2;      // predict every 2 candles (was 5 — too slow on 1h/4h)
const POSTSL_COOLDOWN      = 3;      // candles to wait after SL before re-entering same coin
const ENTRY_GAP_CANDLES    = 1;      // min candles between any two new entries
const MAX_DD               = 0.20;   // halt if down 20% from start
const MIN_ATR_PCT          = 0.002;  // skip if ATR < 0.2% (was 0.3% — blocked PAXG/low-vol coins)
const MAX_ATR_PCT          = 0.06;   // skip if ATR > 6% (flash crash / manipulation)
// High-correlation pairs — never hold both in same direction simultaneously
const CORRELATED_PAIRS: [Coin, Coin][] = [
  ['BTC', 'ETH'],
  ['ARB', 'OP'],   // L2s move together
  ['APT', 'SUI'],  // Move competitors
];

// ─── Types ───────────────────────────────────────────────────────────────────

export interface TradeMarkerData {
  time: number;
  coin: Coin;
  type: 'ENTRY_LONG' | 'ENTRY_SHORT' | 'EXIT_TP' | 'EXIT_SL';
}

export interface AITraderResult {
  portfolio:      StoredPortfolio;
  closedTrades:   Trade[];
  predictions:    Record<Coin, Prediction | null>;
  tfMatrix:       TFBias[];
  tradeMarkers:   TradeMarkerData[];
  learnedWeights: LearnedWeights;
  activityLog:    LogEntry[];
  resetPortfolio: () => void;
  forceEntry:     (coin: Coin) => void;
}

interface Props {
  allCandles:      Record<Coin, Candle[]>;
  activeCandles:   Candle[];
  activeCoin:      Coin;
  activeTimeframe: Timeframe;
  tickers:         Record<Coin, TickerData | null>;
  fearGreed:       FearGreed | null;
  fundingRates:    Record<Coin, FundingRate | null>;
  prevDay:         { high: number; low: number } | null;
}

function uid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

// ─── Activity log persistence ─────────────────────────────────────────────────

const LOG_KEY = 'apex-activity-log';
const MAX_LOG = 150;

function loadActivityLog(): LogEntry[] {
  try {
    const raw = localStorage.getItem(LOG_KEY);
    return raw ? (JSON.parse(raw) as LogEntry[]) : [];
  } catch { return []; }
}

function saveActivityLog(log: LogEntry[]): void {
  try { localStorage.setItem(LOG_KEY, JSON.stringify(log.slice(0, MAX_LOG))); } catch {}
}

function fmtPrice(v: number): string {
  return v >= 1000
    ? `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
    : `$${v.toFixed(2)}`;
}

function fmtR(pnl: number, riskDollars: number): string {
  if (riskDollars <= 0) return '';
  const r = pnl / riskDollars;
  return `${r >= 0 ? '+' : ''}${r.toFixed(1)}R`;
}

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useAITrader({
  allCandles,
  activeCandles,
  activeCoin,
  activeTimeframe,
  tickers,
  fearGreed,
  fundingRates,
  prevDay,
}: Props): AITraderResult {

  // Portfolio state — only trades + balance (predictions are ephemeral)
  const [portfolio,      setPortfolio]      = useState<StoredPortfolio>(loadPortfolio);
  const [closedTrades,   setClosedTrades]   = useState<Trade[]>(loadHistory);
  const [predictions,    setPredictions]    = useState<Record<Coin, Prediction | null>>(
    () => Object.fromEntries(COINS.map(c => [c, null])) as Record<Coin, Prediction | null>
  );
  const [tfMatrix,       setTfMatrix]       = useState<TFBias[]>([]);
  const [learnedWeights, setLearnedWeights] = useState<LearnedWeights>(loadWeights);
  const [activityLog,    setActivityLog]    = useState<LogEntry[]>(loadActivityLog);

  const addLog = useCallback((entry: Omit<LogEntry, 'id' | 'timestamp'>) => {
    const full: LogEntry = { id: uid(), timestamp: Date.now(), ...entry };
    setActivityLog(prev => {
      const next = [full, ...prev].slice(0, MAX_LOG);
      saveActivityLog(next);
      return next;
    });
  }, []);

  // Refs for rate-limiting (mutated without triggering re-renders)
  const portfolioRef           = useRef<StoredPortfolio>(portfolio);
  const closedRef              = useRef<Trade[]>(closedTrades);
  const lastPredCandle         = useRef<Partial<Record<Coin, number>>>({});
  const slCooldownUntilCandle  = useRef<Partial<Record<Coin, number>>>({});  // candle # not to trade before
  const lastEntryCandle        = useRef(0);  // last candle index any trade was entered
  const activeCandleCount      = useRef(0);

  // Keep refs in sync
  useEffect(() => { portfolioRef.current = portfolio; }, [portfolio]);
  useEffect(() => { closedRef.current    = closedTrades; }, [closedTrades]);

  // ─── Prediction rate limiter ────────────────────────────────────────────

  const shouldPredict = useCallback((coin: Coin, currentCount: number): boolean => {
    const lastPred = lastPredCandle.current[coin] ?? 0;
    if (currentCount - lastPred < PRED_CANDLE_GAP) return false;

    const slUntil = slCooldownUntilCandle.current[coin] ?? 0;
    if (currentCount < slUntil) return false;       // still in post-SL cooldown

    return true;
  }, []);

  // ─── Run predictions (rate-limited) ────────────────────────────────────

  const runPredictions = useCallback((candleCount: number) => {
    const next = Object.fromEntries(COINS.map(c => [c, null])) as Record<Coin, Prediction | null>;

    COINS.forEach(coin => {
      if (!shouldPredict(coin, candleCount)) {
        next[coin] = predictions[coin]; // keep previous
        return;
      }

      const candles = coin === activeCoin ? activeCandles : allCandles[coin];
      if (candles.length < 20) { next[coin] = null; return; }

      const pred = generatePrediction(
        candles, coin, activeTimeframe,
        fearGreed, fundingRates[coin] ?? null, [],
        coin === activeCoin ? prevDay : null,
      );
      next[coin] = pred;
      lastPredCandle.current[coin] = candleCount;
    });

    setPredictions(next);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCoin, activeCandles, allCandles, activeTimeframe, fearGreed, fundingRates, prevDay, shouldPredict]);

  // Fire when active candles close (but only every PRED_CANDLE_GAP candles)
  const prevCandleCount = useRef(0);
  useEffect(() => {
    if (activeCandles.length === prevCandleCount.current) return;
    const newCount = activeCandles.length;
    prevCandleCount.current = newCount;
    activeCandleCount.current = newCount;
    if (newCount < 20) return;
    runPredictions(newCount);
  }, [activeCandles.length, runPredictions]);

  // Fire once when REST candles for other coins arrive
  const allLoaded = useRef(false);
  useEffect(() => {
    if (allLoaded.current) return;
    const hasAll = COINS.every(c => allCandles[c].length >= 20);
    if (!hasAll) return;
    allLoaded.current = true;
    runPredictions(activeCandleCount.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allCandles.BTC?.length,  allCandles.ETH?.length,  allCandles.BNB?.length,
      allCandles.XRP?.length,  allCandles.LTC?.length,  allCandles.TRX?.length,
      allCandles.SOL?.length,  allCandles.AVAX?.length, allCandles.DOT?.length,
      allCandles.LINK?.length, allCandles.ATOM?.length, allCandles.NEAR?.length,
      allCandles.UNI?.length,  allCandles.ADA?.length,
      allCandles.DOGE?.length, allCandles.SUI?.length,  allCandles.APT?.length,
      allCandles.ARB?.length,  allCandles.OP?.length,   allCandles.INJ?.length,
      allCandles.PAXG?.length]);

  // TF matrix update (follows same cadence as active candles, cached)
  const lastTFCandle = useRef(0);
  useEffect(() => {
    const count = activeCandles.length;
    if (count < 20 || count - lastTFCandle.current < PRED_CANDLE_GAP) return;
    lastTFCandle.current = count;
    setTfMatrix(getTFMatrix(activeCandles, activeTimeframe));
  }, [activeCandles, activeTimeframe]);

  // ─── Enter trade (only on fresh prediction, respecting all gates) ───────

  useEffect(() => {
    const port       = portfolioRef.current;
    const candleIdx  = activeCandleCount.current;
    const drawdown   = (port.startBalance - port.balance) / port.startBalance;

    // Global gates
    if (drawdown >= MAX_DD) return;  // drawdown halt
    if (port.openTrades.length >= MAX_OPEN) return;
    if (candleIdx - lastEntryCandle.current < ENTRY_GAP_CANDLES) return;

    // Accumulate all qualifying trades first, then write once.
    // React batches setPortfolio calls — only the last survives if called in a loop,
    // so we must collect all trades and call setPortfolio exactly once.
    const tradesToEnter: Trade[] = [];

    COINS.forEach(coin => {
      // Respect MAX_OPEN including trades already queued this cycle
      if (port.openTrades.length + tradesToEnter.length >= MAX_OPEN) return;

      const pred = predictions[coin];
      if (!pred || pred.direction === 'NEUTRAL') return;
      if (pred.confidence < MIN_CONFIDENCE) return;

      // Per-coin gates
      const alreadyOpen = port.openTrades.some(t => t.coin === coin)
        || tradesToEnter.some(t => t.coin === coin);
      if (alreadyOpen) return;
      const slUntil = slCooldownUntilCandle.current[coin] ?? 0;
      if (candleIdx < slUntil) return; // SL cooldown active

      const ticker = tickers[coin];
      if (!ticker) return;

      const price = ticker.price;

      // Volatility gate: skip choppy or panic markets
      const atrPct = pred.atr / price;
      if (atrPct < MIN_ATR_PCT || atrPct > MAX_ATR_PCT) return;

      // Correlation gate: check against both existing and pending trades
      const allOpen = [...port.openTrades, ...tradesToEnter];
      const hasCorrelated = CORRELATED_PAIRS.some(([a, b]) => {
        if (coin !== a && coin !== b) return false;
        const other = coin === a ? b : a;
        return allOpen.some(t => t.coin === other && t.direction === pred.direction);
      });
      if (hasCorrelated) return;

      // Dynamic risk: scale with confidence — 1% weak, 2% normal, 3% strong
      const riskPct     = pred.confidence >= 80 ? 0.03 : pred.confidence >= 72 ? 0.02 : 0.01;
      const riskDollars = port.balance * riskPct;
      const stopDist    = Math.abs(price - pred.stopPrice);
      if (stopDist === 0) return;

      const coinUnits = riskDollars / stopDist;
      const rawSize   = coinUnits * price;
      // Cap at 95% of balance — never reject, just size down to what we have
      const size      = Math.min(rawSize, port.balance * 0.95);
      if (size < 1) return;

      tradesToEnter.push({
        id:           uid(),
        predictionId: pred.id,
        coin,
        direction:    pred.direction as 'LONG' | 'SHORT',
        entryPrice:   price,
        entryTime:    Date.now(),
        stopLoss:     pred.stopPrice,
        takeProfit:   pred.targetPrice,
        size,
        status:       'OPEN',
        signals:      pred.signals,  // snapshot for self-learning
      });
    });

    if (tradesToEnter.length > 0) {
      const next: StoredPortfolio = {
        ...port,
        openTrades: [...port.openTrades, ...tradesToEnter],
      };
      lastEntryCandle.current = candleIdx;
      setPortfolio(next);
      tradesToEnter.forEach(t => {
        onTradeOpened(next, t);
        const pred = predictions[t.coin];
        const conf = pred ? `${pred.confidence.toFixed(0)}% conf` : '';
        const riskPct = t.size > 0 ? ((Math.abs(t.entryPrice - t.stopLoss) * (t.size / t.entryPrice)) / port.balance * 100).toFixed(1) : '?';
        addLog({
          type: 'trade_open',
          level: 'info',
          message: `◆ ${t.coin} ${t.direction} entered at ${fmtPrice(t.entryPrice)}`,
          detail: `${conf} · risk ${riskPct}% · TP ${fmtPrice(t.takeProfit)} · SL ${fmtPrice(t.stopLoss)}`,
        });
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions, tickers]);

  // ─── Monitor positions on price ticks (NO storage writes here) ─────────

  const prevPrices = useRef<Record<Coin, number | undefined>>(
    Object.fromEntries(COINS.map(c => [c, undefined])) as Record<Coin, number | undefined>
  );

  useEffect(() => {
    const port = portfolioRef.current;
    if (port.openTrades.length === 0) return;

    // Debounce: only run if a relevant coin's price actually changed
    const changed = COINS.some(c => tickers[c]?.price !== prevPrices.current[c]);
    if (!changed) return;
    COINS.forEach(c => { prevPrices.current[c] = tickers[c]?.price; });

    let balanceDelta = 0;
    let tradeChanged = false;
    const remaining: Trade[] = [];
    const newlyClosed: Trade[] = [];

    port.openTrades.forEach(trade => {
      const ticker = tickers[trade.coin];
      if (!ticker) { remaining.push(trade); return; }

      const price  = ticker.price;
      const isLong = trade.direction === 'LONG';
      const hitTP  = isLong ? price >= trade.takeProfit : price <= trade.takeProfit;
      const hitSL  = isLong ? price <= trade.stopLoss   : price >= trade.stopLoss;

      if (!hitTP && !hitSL) {
        // Trailing stop: ratchet stop loss up (LONG) or down (SHORT) as price moves
        // Derived ATR from original TP distance (TP = entry ± ATR×3)
        const riskDist  = Math.abs(trade.entryPrice - trade.stopLoss);
        if (riskDist > 0) {
          const unrealizedR = isLong
            ? (price - trade.entryPrice) / riskDist
            : (trade.entryPrice - price) / riskDist;

          if (unrealizedR >= 1.0) {
            // Trail at 1.5×ATR behind price — but never worse than entry (always risk-free)
            const tradeATR = Math.abs(trade.entryPrice - trade.takeProfit) / 3.0;
            const trailDist = tradeATR * 1.5;
            const newSL = isLong
              ? Math.max(trade.entryPrice, price - trailDist)   // floor = breakeven
              : Math.min(trade.entryPrice, price + trailDist);   // ceiling = breakeven

            const improved = isLong ? newSL > trade.stopLoss : newSL < trade.stopLoss;
            if (improved) {
              remaining.push({ ...trade, stopLoss: newSL });
              tradeChanged = true;
              return;
            }
          }
        }

        remaining.push(trade);
        return;
      }

      const exitPrice = hitTP ? trade.takeProfit : trade.stopLoss;
      const units     = trade.size / trade.entryPrice;
      const pnl       = isLong
        ? units * (exitPrice - trade.entryPrice)
        : units * (trade.entryPrice - exitPrice);

      const closed: Trade = {
        ...trade,
        status:    hitTP ? 'CLOSED_TP' : 'CLOSED_SL',
        exitPrice,
        exitTime:  Date.now(),
        pnl,
        pnlPct:    (pnl / trade.size) * 100,
      };

      balanceDelta += pnl;
      newlyClosed.push(closed);
      tradeChanged = true;

      // SL cooldown — don't re-enter this coin for N candles
      if (hitSL) {
        slCooldownUntilCandle.current[trade.coin] =
          (activeCandleCount.current) + POSTSL_COOLDOWN;
      }
    });

    if (!tradeChanged) return;

    const nextPortfolio: StoredPortfolio = {
      ...port,
      balance:    Math.max(0, port.balance + balanceDelta),
      openTrades: remaining,
    };

    setPortfolio(nextPortfolio);

    // Only update closed history + storage on actual closes (not breakeven updates)
    if (newlyClosed.length > 0) {
      const nextHistory = [...newlyClosed, ...closedRef.current].slice(0, 200);
      setClosedTrades(nextHistory);
      newlyClosed.forEach(t => {
        onTradeClosed(nextPortfolio, t);
        const isWin   = t.status === 'CLOSED_TP';
        const pnl     = t.pnl ?? 0;
        const riskAmt = Math.abs(t.entryPrice - t.stopLoss) * (t.size / t.entryPrice);
        const rStr    = fmtR(pnl, riskAmt);
        const pnlStr  = `${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}`;
        addLog({
          type:  'trade_close',
          level: isWin ? 'success' : 'error',
          message: isWin
            ? `✓ ${t.coin} ${t.direction} hit TP — ${pnlStr} (${rStr})`
            : `✗ ${t.coin} ${t.direction} hit SL — ${pnlStr} (${rStr})`,
          detail: `entry ${fmtPrice(t.entryPrice)} → exit ${fmtPrice(t.exitPrice ?? 0)}`,
        });
      });

      // Self-learning: retrain weights every LEARN_EVERY new closed trades
      const current = loadWeights();
      if (shouldLearn(nextHistory.length, current.trainedOn)) {
        const updated = learnFromTrades(nextHistory, current);
        if (updated) {
          setLearnedWeights(updated);
          const boosted = updated.history[0]?.changes.filter(c => c.delta >  0.005).map(c => c.signal) ?? [];
          const reduced = updated.history[0]?.changes.filter(c => c.delta < -0.005).map(c => c.signal) ?? [];
          addLog({
            type:  'learn',
            level: 'warning',
            message: `⟳ Model v${updated.version} updated — trained on ${updated.trainedOn} trades`,
            detail: updated.history[0]?.summary ?? '',
          });
          boosted.forEach(sig => {
            const ch = updated.history[0]?.changes.find(c => c.signal === sig);
            if (ch) addLog({
              type: 'learn', level: 'success',
              message: `▲ ${sig} weight raised ${(ch.oldWeight * 100).toFixed(1)}% → ${(ch.newWeight * 100).toFixed(1)}%`,
              detail: `accuracy ${(ch.accuracy * 100).toFixed(0)}% — performing well`,
            });
          });
          reduced.forEach(sig => {
            const ch = updated.history[0]?.changes.find(c => c.signal === sig);
            if (ch) addLog({
              type: 'learn', level: 'error',
              message: `▼ ${sig} weight cut ${(ch.oldWeight * 100).toFixed(1)}% → ${(ch.newWeight * 100).toFixed(1)}%`,
              detail: `accuracy ${(ch.accuracy * 100).toFixed(0)}% — underperforming`,
            });
          });
        }
      }
    }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickers.BTC?.price,  tickers.ETH?.price,  tickers.BNB?.price,
      tickers.XRP?.price,  tickers.LTC?.price,  tickers.TRX?.price,
      tickers.SOL?.price,  tickers.AVAX?.price, tickers.DOT?.price,
      tickers.LINK?.price, tickers.ATOM?.price, tickers.NEAR?.price,
      tickers.UNI?.price,  tickers.ADA?.price,
      tickers.DOGE?.price, tickers.SUI?.price,  tickers.APT?.price,
      tickers.ARB?.price,  tickers.OP?.price,   tickers.INJ?.price,
      tickers.PAXG?.price]);

  // ─── Trade markers for chart ────────────────────────────────────────────

  const tradeMarkers: TradeMarkerData[] = [
    ...portfolio.openTrades.map(t => ({
      time: Math.floor(t.entryTime / 1000),
      coin: t.coin,
      type: (t.direction === 'LONG' ? 'ENTRY_LONG' : 'ENTRY_SHORT') as TradeMarkerData['type'],
    })),
    ...closedTrades.slice(0, 50).flatMap(t => {
      const m: TradeMarkerData[] = [
        { time: Math.floor(t.entryTime / 1000), coin: t.coin, type: t.direction === 'LONG' ? 'ENTRY_LONG' : 'ENTRY_SHORT' },
      ];
      if (t.exitTime) m.push({
        time: Math.floor(t.exitTime / 1000), coin: t.coin,
        type: t.status === 'CLOSED_TP' ? 'EXIT_TP' : 'EXIT_SL',
      });
      return m;
    }),
  ];

  // ─── Force Entry (bypasses cooldowns — manual trigger) ─────────────────

  const forceEntry = useCallback((coin: Coin) => {
    const port = portfolioRef.current;
    const pred = predictions[coin];
    if (!pred || pred.direction === 'NEUTRAL') return;
    if (port.openTrades.some(t => t.coin === coin)) return; // already open

    const ticker = tickers[coin];
    if (!ticker) return;
    const price = ticker.price;

    const riskPct     = pred.confidence >= 80 ? 0.03 : pred.confidence >= 72 ? 0.02 : 0.01;
    const riskDollars = port.balance * riskPct;
    const stopDist    = Math.abs(price - pred.stopPrice);
    if (stopDist === 0) return;

    const coinUnits = riskDollars / stopDist;
    const rawSize   = coinUnits * price;
    const size      = Math.min(rawSize, port.balance * 0.95);
    if (size < 1) return;

    const trade: Trade = {
      id:           uid(),
      predictionId: pred.id,
      coin,
      direction:    pred.direction as 'LONG' | 'SHORT',
      entryPrice:   price,
      entryTime:    Date.now(),
      stopLoss:     pred.stopPrice,
      takeProfit:   pred.targetPrice,
      size,
      status:       'OPEN',
      signals:      pred.signals,  // snapshot for self-learning
    };

    const next: StoredPortfolio = { ...port, openTrades: [...port.openTrades, trade] };
    lastEntryCandle.current = activeCandleCount.current;
    setPortfolio(next);
    onTradeOpened(next, trade);
    addLog({
      type: 'trade_open', level: 'info',
      message: `⚡ ${trade.coin} ${trade.direction} FORCE ENTRY at ${fmtPrice(trade.entryPrice)}`,
      detail: `TP ${fmtPrice(trade.takeProfit)} · SL ${fmtPrice(trade.stopLoss)}`,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions, tickers]);

  // ─── Reset ─────────────────────────────────────────────────────────────

  const resetPortfolio = useCallback(() => {
    const fresh: StoredPortfolio = { balance: START_BALANCE, startBalance: START_BALANCE, openTrades: [] };
    setPortfolio(fresh);
    setClosedTrades([]);
    setPredictions(Object.fromEntries(COINS.map(c => [c, null])) as Record<Coin, Prediction | null>);
    lastPredCandle.current        = {};
    slCooldownUntilCandle.current = {};
    lastEntryCandle.current       = 0;
    allLoaded.current             = false;
    resetStorage();
    resetWeights();
    setLearnedWeights(loadWeights());
    const cleared: LogEntry[] = [];
    setActivityLog(cleared);
    saveActivityLog(cleared);
    addLog({ type: 'system', level: 'warning', message: '↺ Portfolio reset — starting fresh at $1,000' });
  }, [addLog]);

  return { portfolio, closedTrades, predictions, tfMatrix, tradeMarkers, learnedWeights, activityLog, resetPortfolio, forceEntry };
}
