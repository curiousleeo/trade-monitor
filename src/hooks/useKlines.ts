import { useState, useEffect, useRef } from 'react';
import { Candle, Coin, Timeframe } from '../types';

const SYMBOLS: Record<Coin, string> = {
  BTC: 'btcusdt', ETH: 'ethusdt', SOL: 'solusdt',
  BNB: 'bnbusdt', XRP: 'xrpusdt', AVAX: 'avaxusdt',
  DOGE: 'dogeusdt', LINK: 'linkusdt', ADA: 'adausdt',
};

function parseRestKline(k: string[]): Candle {
  return {
    time: Math.floor(Number(k[0]) / 1000),
    open: parseFloat(k[1]),
    high: parseFloat(k[2]),
    low: parseFloat(k[3]),
    close: parseFloat(k[4]),
    volume: parseFloat(k[5]),
  };
}

export function useKlines(coin: Coin, timeframe: Timeframe) {
  const [candles, setCandles]       = useState<Candle[]>([]);
  const [liveCandle, setLiveCandle] = useState<Candle | null>(null);
  const wsRef          = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay = useRef(1000);
  // Each effect run gets a unique connId. Stale callbacks compare their captured
  // connId against the ref and bail out if it's been superseded — fixes the race
  // condition where the old WS's onclose fires after the new effect has started.
  const connIdRef = useRef(0);

  useEffect(() => {
    setCandles([]);
    setLiveCandle(null);
    reconnectDelay.current = 1000;

    const connId = ++connIdRef.current;
    const symbol = SYMBOLS[coin].toUpperCase();

    fetch(
      `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${timeframe}&limit=500`
    )
      .then(res => res.json())
      .then((data: string[][]) => {
        if (connIdRef.current !== connId) return; // stale: coin/TF changed mid-fetch
        if (!Array.isArray(data)) return;
        setCandles(data.map(parseRestKline));
      })
      .catch(err => console.error('Klines fetch failed:', err));

    function connect() {
      if (connIdRef.current !== connId) return; // superseded
      if (wsRef.current) wsRef.current.close();

      const ws = new WebSocket(
        `wss://stream.binance.com:9443/ws/${SYMBOLS[coin]}@kline_${timeframe}`
      );

      ws.onmessage = (event) => {
        if (connIdRef.current !== connId) return;
        const msg = JSON.parse(event.data);
        const k = msg.k;
        const candle: Candle = {
          time:   Math.floor(k.t / 1000),
          open:   parseFloat(k.o),
          high:   parseFloat(k.h),
          low:    parseFloat(k.l),
          close:  parseFloat(k.c),
          volume: parseFloat(k.v),
        };

        setLiveCandle(candle);

        if (k.x) {
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

      ws.onerror = () => console.error(`[useKlines] WS error (${coin} ${timeframe})`);

      ws.onclose = () => {
        if (connIdRef.current !== connId) return; // superseded — don't reconnect
        const delay = reconnectDelay.current;
        reconnectDelay.current = Math.min(30_000, delay * 2);
        console.warn(`[useKlines] WS closed — reconnecting in ${delay}ms (${coin} ${timeframe})`);
        reconnectTimer.current = setTimeout(connect, delay);
      };

      wsRef.current = ws;
    }

    connect();

    return () => {
      // Don't touch connIdRef here — the next effect will increment it.
      // Just stop in-flight work for this connection.
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [coin, timeframe]);

  return { candles, liveCandle };
}
