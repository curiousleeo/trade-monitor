import { useState, useEffect, useRef } from 'react';
import { Coin, TickerData } from '../types';

type Tickers = Record<Coin, TickerData | null>;

const STREAM = 'wss://stream.binance.com:9443/stream?streams=btcusdt@miniTicker/ethusdt@miniTicker/solusdt@miniTicker';

const STREAM_TO_COIN: Record<string, Coin> = {
  btcusdt: 'BTC',
  ethusdt: 'ETH',
  solusdt: 'SOL',
};

export function use24hTicker(): Tickers {
  const [tickers, setTickers] = useState<Tickers>({ BTC: null, ETH: null, SOL: null });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(STREAM);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const data = msg.data;
      if (!data || !data.s) return;

      const coin = STREAM_TO_COIN[data.s.toLowerCase()];
      if (!coin) return;

      setTickers(prev => ({
        ...prev,
        [coin]: {
          price:     parseFloat(data.c),
          change24h: parseFloat(data.P),
          volume24h: parseFloat(data.q), // quote volume in USDT
          high24h:   parseFloat(data.h),
          low24h:    parseFloat(data.l),
        },
      }));
    };

    wsRef.current = ws;
    return () => ws.close();
  }, []);

  return tickers;
}
