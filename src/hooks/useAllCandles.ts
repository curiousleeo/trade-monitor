/**
 * Fetches 100 REST candles for all trade coins on the current timeframe.
 * Refreshes every 3 minutes. Used by AI trader for all coins.
 */

import { useEffect, useState, useRef } from 'react';
import { Candle, Coin, Timeframe } from '../types';

export type AllCandles = Record<Coin, Candle[]>;

const TRADE_COINS: Coin[] = [
  'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
  'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
  'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ',
  'PAXG',
];

const SYMBOLS: Record<Coin, string> = {
  BTC: 'BTCUSDT',  ETH: 'ETHUSDT',  BNB: 'BNBUSDT',  XRP: 'XRPUSDT',  LTC: 'LTCUSDT',  TRX: 'TRXUSDT',
  SOL: 'SOLUSDT',  AVAX: 'AVAXUSDT', DOT: 'DOTUSDT',  LINK: 'LINKUSDT', ATOM: 'ATOMUSDT', NEAR: 'NEARUSDT',
  UNI: 'UNIUSDT',  ADA: 'ADAUSDT',
  DOGE: 'DOGEUSDT', SUI: 'SUIUSDT',  APT: 'APTUSDT',  ARB: 'ARBUSDT',  OP: 'OPUSDT',    INJ: 'INJUSDT',
  PAXG: 'PAXGUSDT',
};

const REFRESH_MS = 3 * 60 * 1000; // 3 minutes

function parseRest(raw: unknown[][]): Candle[] {
  return raw.map((k) => ({
    time:   Math.floor((k[0] as number) / 1000),
    open:   parseFloat(k[1] as string),
    high:   parseFloat(k[2] as string),
    low:    parseFloat(k[3] as string),
    close:  parseFloat(k[4] as string),
    volume: parseFloat(k[5] as string),
  }));
}

async function fetchCandles(coin: Coin, tf: Timeframe): Promise<Candle[]> {
  const url = `https://api.binance.com/api/v3/klines?symbol=${SYMBOLS[coin]}&interval=${tf}&limit=500`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const data = await res.json();
  return parseRest(data as unknown[][]);
}

const EMPTY: AllCandles = Object.fromEntries(TRADE_COINS.map(c => [c, []])) as unknown as AllCandles;

export function useAllCandles(timeframe: Timeframe): AllCandles {
  const [all, setAll] = useState<AllCandles>(EMPTY);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const results = await Promise.all(TRADE_COINS.map(c => fetchCandles(c, timeframe)));
      if (cancelled) return;
      const next = Object.fromEntries(TRADE_COINS.map((c, i) => [c, results[i]])) as unknown as AllCandles;
      setAll(next);
      timerRef.current = setTimeout(load, REFRESH_MS);
    }

    load();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [timeframe]);

  return all;
}
