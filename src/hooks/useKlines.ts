import { useState, useEffect, useRef } from 'react';
import { Candle, Coin, Timeframe } from '../types';

const SYMBOLS: Record<Coin, string> = {
  BTC: 'btcusdt',
  ETH: 'ethusdt',
  SOL: 'solusdt',
};

function parseRestKline(k: string[]): Candle {
  return {
    time: Math.floor(Number(k[0]) / 1000),
    open: parseFloat(k[1]),
    high: parseFloat(k[2]),
    low: parseFloat(k[3]),
    close: parseFloat(k[4]),
  };
}

export function useKlines(coin: Coin, timeframe: Timeframe) {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [liveCandle, setLiveCandle] = useState<Candle | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    setCandles([]);
    setLiveCandle(null);

    const symbol = SYMBOLS[coin].toUpperCase();

    // Fetch historical candles
    fetch(
      `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${timeframe}&limit=200`
    )
      .then(res => res.json())
      .then((data: string[][]) => {
        setCandles(data.map(parseRestKline));
      })
      .catch(err => console.error('Klines fetch failed:', err));

    // Close existing WebSocket
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(
      `wss://stream.binance.com:9443/ws/${SYMBOLS[coin]}@kline_${timeframe}`
    );

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const k = msg.k;
      const candle: Candle = {
        time: Math.floor(k.t / 1000),
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
      };

      setLiveCandle(candle);

      if (k.x) {
        // Candle closed — merge into historical
        setCandles(prev => {
          const idx = prev.findIndex(c => c.time === candle.time);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = candle;
            return updated;
          }
          return [...prev, candle];
        });
      }
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [coin, timeframe]);

  return { candles, liveCandle };
}
