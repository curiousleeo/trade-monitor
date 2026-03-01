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
  Candle, Coin, FearGreed, FundingRate, NewsItem,
  Prediction, TFBias, TickerData, Trade, Timeframe,
} from '../types';
import { generatePrediction, getTFMatrix } from '../ai/engine';
import {
  StoredPortfolio,
  loadPortfolio, loadHistory,
  onTradeOpened, onTradeClosed, resetStorage,
} from '../lib/db';

// ─── Config ──────────────────────────────────────────────────────────────────

const COINS: Coin[] = ['BTC', 'ETH', 'SOL'];
const START_BALANCE        = 1000;
const RISK_PCT             = 0.02;   // 2% risk per trade
const MIN_CONFIDENCE       = 65;     // min score to enter
const MAX_OPEN             = 2;      // max concurrent positions
const PRED_CANDLE_GAP      = 5;      // predict once per 5 candles (not every close)
const POSTSL_COOLDOWN      = 4;      // candles to wait after SL before re-entering same coin
const ENTRY_GAP_CANDLES    = 2;      // min candles between any two entries (all coins)
const MAX_DD               = 0.25;   // halt if down 25% from start

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
  resetPortfolio: () => void;
}

interface Props {
  allCandles:      Record<Coin, Candle[]>;
  activeCandles:   Candle[];
  activeCoin:      Coin;
  activeTimeframe: Timeframe;
  tickers:         Record<Coin, TickerData | null>;
  fearGreed:       FearGreed | null;
  fundingRates:    Record<Coin, FundingRate | null>;
  news:            NewsItem[];
  prevDay:         { high: number; low: number } | null;
}

function uid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
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
  news,
  prevDay,
}: Props): AITraderResult {

  // Portfolio state — only trades + balance (predictions are ephemeral)
  const [portfolio,    setPortfolio]    = useState<StoredPortfolio>(loadPortfolio);
  const [closedTrades, setClosedTrades] = useState<Trade[]>(loadHistory);
  const [predictions,  setPredictions]  = useState<Record<Coin, Prediction | null>>({ BTC: null, ETH: null, SOL: null });
  const [tfMatrix,     setTfMatrix]     = useState<TFBias[]>([]);

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
    const next: Record<Coin, Prediction | null> = { BTC: null, ETH: null, SOL: null };

    COINS.forEach(coin => {
      if (!shouldPredict(coin, candleCount)) {
        next[coin] = predictions[coin]; // keep previous
        return;
      }

      const candles = coin === activeCoin ? activeCandles : allCandles[coin];
      if (candles.length < 20) { next[coin] = null; return; }

      const pred = generatePrediction(
        candles, coin, activeTimeframe,
        fearGreed, fundingRates[coin] ?? null, news,
        coin === activeCoin ? prevDay : null,
      );
      next[coin] = pred;
      lastPredCandle.current[coin] = candleCount;
    });

    setPredictions(next);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCoin, activeCandles, allCandles, activeTimeframe, fearGreed, fundingRates, news, prevDay, shouldPredict]);

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
  }, [allCandles.BTC.length, allCandles.ETH.length, allCandles.SOL.length]);

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

    COINS.forEach(coin => {
      const pred = predictions[coin];
      if (!pred || pred.direction === 'NEUTRAL') return;
      if (pred.confidence < MIN_CONFIDENCE) return;

      // Per-coin gates
      if (port.openTrades.some(t => t.coin === coin)) return; // already open
      const slUntil = slCooldownUntilCandle.current[coin] ?? 0;
      if (candleIdx < slUntil) return; // SL cooldown active

      const ticker = tickers[coin];
      if (!ticker) return;

      const price      = ticker.price;
      const riskDollars = port.balance * RISK_PCT;
      const stopDist   = Math.abs(price - pred.stopPrice);
      if (stopDist === 0) return;

      const coinUnits  = riskDollars / stopDist;
      const size       = coinUnits * price;
      if (size < 1 || size > port.balance * 0.5) return; // sanity: max 50% in one trade

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
      };

      const next: StoredPortfolio = {
        ...port,
        openTrades: [...port.openTrades, trade],
      };
      lastEntryCandle.current = candleIdx;
      setPortfolio(next);
      onTradeOpened(next, trade);  // ← only write to storage here
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions]);

  // ─── Monitor positions on price ticks (NO storage writes here) ─────────

  const prevPrices = useRef<Record<Coin, number | undefined>>({ BTC: undefined, ETH: undefined, SOL: undefined });

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

      if (!hitTP && !hitSL) { remaining.push(trade); return; }

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
    const nextHistory = [...newlyClosed, ...closedRef.current].slice(0, 200);

    setPortfolio(nextPortfolio);
    setClosedTrades(nextHistory);

    // Storage: write once per trade close event, never on ticks
    newlyClosed.forEach(t => onTradeClosed(nextPortfolio, t));

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickers.BTC?.price, tickers.ETH?.price, tickers.SOL?.price]);

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

  // ─── Reset ─────────────────────────────────────────────────────────────

  const resetPortfolio = useCallback(() => {
    const fresh: StoredPortfolio = { balance: START_BALANCE, startBalance: START_BALANCE, openTrades: [] };
    setPortfolio(fresh);
    setClosedTrades([]);
    setPredictions({ BTC: null, ETH: null, SOL: null });
    lastPredCandle.current        = {};
    slCooldownUntilCandle.current = {};
    lastEntryCandle.current       = 0;
    allLoaded.current             = false;
    resetStorage();
  }, []);

  return { portfolio, closedTrades, predictions, tfMatrix, tradeMarkers, resetPortfolio };
}
