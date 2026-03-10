import { useState, useEffect, useRef } from 'react';
import { Coin, TickerData } from '../types';

type Tickers = Record<Coin, TickerData | null>;

const ALL_COINS: Coin[] = [
  'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
  'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
  'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ',
  'PAXG',
];

const STREAM = 'wss://stream.binance.com:9443/stream?streams=' +
  ALL_COINS.map(c => `${c.toLowerCase()}usdt@miniTicker`).join('/');

const STREAM_TO_COIN: Record<string, Coin> = {
  btcusdt: 'BTC',  ethusdt: 'ETH',  bnbusdt: 'BNB',  xrpusdt: 'XRP',  ltcusdt: 'LTC',  trxusdt: 'TRX',
  solusdt: 'SOL',  avaxusdt: 'AVAX', dotusdt: 'DOT',  linkusdt: 'LINK', atomusdt: 'ATOM', nearusdt: 'NEAR',
  uniusdt: 'UNI',  adausdt: 'ADA',
  dogeusdt: 'DOGE', suiusdt: 'SUI',  aptusdt: 'APT',  arbusdt: 'ARB',  opusdt: 'OP',    injusdt: 'INJ',
  paxgusdt: 'PAXG',
};

const INITIAL: Tickers = Object.fromEntries(ALL_COINS.map(c => [c, null])) as Tickers;

export function use24hTicker(): Tickers {
  const [tickers, setTickers] = useState<Tickers>(INITIAL);
  const wsRef           = useRef<WebSocket | null>(null);
  const reconnectTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelay  = useRef(1000);
  const destroyed       = useRef(false);

  useEffect(() => {
    destroyed.current      = false;
    reconnectDelay.current = 1000;

    function connect() {
      if (destroyed.current) return;
      if (wsRef.current) wsRef.current.close();

      const ws = new WebSocket(STREAM);

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const data = msg.data;
        if (!data || !data.s) return;

        const coin = STREAM_TO_COIN[data.s.toLowerCase()];
        if (!coin) return;

        const close = parseFloat(data.c);
        const open  = parseFloat(data.o);
        setTickers(prev => ({
          ...prev,
          [coin]: {
            price:     close,
            change24h: open > 0 ? ((close - open) / open) * 100 : 0,
            volume24h: parseFloat(data.q),
            high24h:   parseFloat(data.h),
            low24h:    parseFloat(data.l),
          },
        }));
      };

      ws.onerror = () => console.error('[use24hTicker] WS error');

      ws.onclose = () => {
        if (destroyed.current) return;
        const delay = reconnectDelay.current;
        reconnectDelay.current = Math.min(30_000, delay * 2);
        console.warn(`[use24hTicker] WS closed — reconnecting in ${delay}ms`);
        reconnectTimer.current = setTimeout(connect, delay);
      };

      wsRef.current = ws;
    }

    connect();

    return () => {
      destroyed.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return tickers;
}
