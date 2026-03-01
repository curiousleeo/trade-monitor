/**
 * APEX AI Trader Hook
 * Manages portfolio state, runs predictions, executes/monitors trades.
 * Persists everything to localStorage.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Candle, Coin, FearGreed, FundingRate, NewsItem, Portfolio, Prediction, TFBias, TickerData, Trade, Timeframe } from '../types';
import { generatePrediction, getTFMatrix } from '../ai/engine';

const COINS: Coin[] = ['BTC', 'ETH', 'SOL'];
const STORAGE_KEY = 'apex-portfolio';
const START_BALANCE = 1000;
const RISK_PCT = 0.02;      // 2% of balance risked per trade
const MIN_SCORE = 65;       // minimum confidence to enter
const MAX_OPEN  = 2;        // max concurrent positions

function makePortfolio(): Portfolio {
  return {
    balance:      START_BALANCE,
    startBalance: START_BALANCE,
    openTrades:   [],
    closedTrades: [],
    predictions:  [],
  };
}

function loadPortfolio(): Portfolio {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as Portfolio;
  } catch { /* ignore */ }
  return makePortfolio();
}

function savePortfolio(p: Portfolio) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch { /* ignore */ }
}

function uid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export interface TradeMarkerData {
  time: number;
  coin: Coin;
  type: 'ENTRY_LONG' | 'ENTRY_SHORT' | 'EXIT_TP' | 'EXIT_SL';
}

export interface AITraderResult {
  portfolio: Portfolio;
  predictions: Record<Coin, Prediction | null>;
  tfMatrix: TFBias[];
  tradeMarkers: TradeMarkerData[];
  resetPortfolio: () => void;
}

interface Props {
  allCandles: Record<Coin, Candle[]>;
  activeCandles: Candle[];   // full 500-bar WS candles for active coin
  activeCoin: Coin;
  activeTimeframe: Timeframe;
  tickers: Record<Coin, TickerData | null>;
  fearGreed: FearGreed | null;
  fundingRates: Record<Coin, FundingRate | null>;
  news: NewsItem[];
  prevDay: { high: number; low: number } | null;
}

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
  const [portfolio, setPortfolio] = useState<Portfolio>(loadPortfolio);
  const [predictions, setPredictions] = useState<Record<Coin, Prediction | null>>({ BTC: null, ETH: null, SOL: null });
  const [tfMatrix, setTfMatrix] = useState<TFBias[]>([]);
  const portfolioRef = useRef<Portfolio>(portfolio);

  // Keep ref in sync for use inside closures
  useEffect(() => {
    portfolioRef.current = portfolio;
    savePortfolio(portfolio);
  }, [portfolio]);

  // ─── Generate predictions for each coin ──────────────────────────────────

  const runPredictions = useCallback(() => {
    const next: Record<Coin, Prediction | null> = { BTC: null, ETH: null, SOL: null };
    COINS.forEach(coin => {
      // Use full WS candles for active coin, REST snapshot for others
      const candles = coin === activeCoin ? activeCandles : allCandles[coin];
      if (candles.length < 20) return;

      const pred = generatePrediction(
        candles,
        coin,
        activeTimeframe,
        fearGreed,
        fundingRates[coin] ?? null,
        news,
        coin === activeCoin ? prevDay : null,
      );
      next[coin] = pred;

      // Update portfolio predictions list (keep last 100)
      setPortfolio(prev => {
        const updated = {
          ...prev,
          predictions: [...prev.predictions.slice(-99), pred],
        };
        return updated;
      });
    });
    setPredictions(next);
  }, [activeCoin, activeCandles, allCandles, activeTimeframe, fearGreed, fundingRates, news, prevDay]);

  // Run predictions when active candles change (new candle closes)
  const prevCandleCount = useRef(0);
  useEffect(() => {
    if (activeCandles.length === prevCandleCount.current) return;
    prevCandleCount.current = activeCandles.length;
    if (activeCandles.length < 20) return;
    runPredictions();
  }, [activeCandles.length, runPredictions]);

  // Also run when other coin candles arrive for first time
  useEffect(() => {
    const hasAll = COINS.every(c => allCandles[c].length >= 20);
    if (!hasAll) return;
    runPredictions();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allCandles.BTC.length, allCandles.ETH.length, allCandles.SOL.length]);

  // TF matrix for active coin
  useEffect(() => {
    if (activeCandles.length < 20) return;
    setTfMatrix(getTFMatrix(activeCandles, activeTimeframe));
  }, [activeCandles, activeTimeframe]);

  // ─── Enter trade when prediction is strong enough ─────────────────────────

  useEffect(() => {
    COINS.forEach(coin => {
      const pred = predictions[coin];
      if (!pred || pred.direction === 'NEUTRAL') return;
      if (pred.confidence < MIN_SCORE) return;

      const port = portfolioRef.current;
      if (port.openTrades.length >= MAX_OPEN) return;
      if (port.openTrades.some(t => t.coin === coin)) return; // already in this coin

      const ticker = tickers[coin];
      if (!ticker) return;

      const price = ticker.price;
      // Risk 2% of balance; position size = risk_$ / stop_distance
      const riskDollars = port.balance * RISK_PCT;
      const stopDist = Math.abs(price - pred.stopPrice);
      if (stopDist === 0) return;
      // size = how many $ of the coin we buy/short (spot, no leverage)
      const coinUnits = riskDollars / stopDist;
      const size = coinUnits * price; // USD value of position

      if (size < 1 || size > port.balance) return; // sanity check

      const trade: Trade = {
        id: uid(),
        predictionId: pred.id,
        coin,
        direction: pred.direction as 'LONG' | 'SHORT',
        entryPrice: price,
        entryTime:  Date.now(),
        stopLoss:   pred.stopPrice,
        takeProfit: pred.targetPrice,
        size,
        status: 'OPEN',
      };

      setPortfolio(prev => ({
        ...prev,
        openTrades: [...prev.openTrades, trade],
      }));
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions]);

  // ─── Monitor open positions against live prices ───────────────────────────

  useEffect(() => {
    const port = portfolioRef.current;
    if (port.openTrades.length === 0) return;

    const updated: Trade[] = [];
    let balanceChange = 0;
    let changed = false;

    port.openTrades.forEach(trade => {
      const ticker = tickers[trade.coin];
      if (!ticker) { updated.push(trade); return; }

      const price = ticker.price;
      const isLong = trade.direction === 'LONG';

      const hitTP = isLong ? price >= trade.takeProfit : price <= trade.takeProfit;
      const hitSL = isLong ? price <= trade.stopLoss   : price >= trade.stopLoss;

      if (hitTP || hitSL) {
        const exitPrice = hitTP ? trade.takeProfit : trade.stopLoss;
        const coinUnits = trade.size / trade.entryPrice;
        const pnl = isLong
          ? coinUnits * (exitPrice - trade.entryPrice)
          : coinUnits * (trade.entryPrice - exitPrice);
        const closed: Trade = {
          ...trade,
          status:     hitTP ? 'CLOSED_TP' : 'CLOSED_SL',
          exitPrice,
          exitTime:   Date.now(),
          pnl,
          pnlPct:     (pnl / trade.size) * 100,
        };
        balanceChange += pnl;
        // Mark prediction as resolved
        setPortfolio(prev => ({
          ...prev,
          predictions: prev.predictions.map(p =>
            p.id === trade.predictionId
              ? { ...p, resolved: true, resolvedAt: Date.now(), accurate: hitTP }
              : p,
          ),
        }));
        port.closedTrades = [closed, ...port.closedTrades];
        changed = true;
      } else {
        updated.push(trade);
      }
    });

    if (changed) {
      setPortfolio(prev => ({
        ...prev,
        balance:      Math.max(0, prev.balance + balanceChange),
        openTrades:   updated,
        closedTrades: port.closedTrades,
      }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickers.BTC?.price, tickers.ETH?.price, tickers.SOL?.price]);

  // ─── Build trade markers for chart ───────────────────────────────────────

  const tradeMarkers: TradeMarkerData[] = [
    ...portfolio.openTrades.map(t => ({
      time: Math.floor(t.entryTime / 1000),
      coin: t.coin,
      type: (t.direction === 'LONG' ? 'ENTRY_LONG' : 'ENTRY_SHORT') as TradeMarkerData['type'],
    })),
    ...portfolio.closedTrades.slice(0, 50).flatMap(t => {
      const markers: TradeMarkerData[] = [
        { time: Math.floor(t.entryTime / 1000), coin: t.coin, type: t.direction === 'LONG' ? 'ENTRY_LONG' : 'ENTRY_SHORT' },
      ];
      if (t.exitTime) {
        markers.push({ time: Math.floor(t.exitTime / 1000), coin: t.coin, type: t.status === 'CLOSED_TP' ? 'EXIT_TP' : 'EXIT_SL' });
      }
      return markers;
    }),
  ];

  const resetPortfolio = useCallback(() => {
    const fresh = makePortfolio();
    setPortfolio(fresh);
    savePortfolio(fresh);
    setPredictions({ BTC: null, ETH: null, SOL: null });
  }, []);

  return { portfolio, predictions, tfMatrix, tradeMarkers, resetPortfolio };
}
