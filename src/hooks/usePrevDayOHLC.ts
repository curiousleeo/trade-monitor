import { useState, useEffect } from 'react';
import { Coin } from '../types';

const SYMBOLS: Record<Coin, string> = {
  BTC: 'BTCUSDT', ETH: 'ETHUSDT', SOL: 'SOLUSDT',
  BNB: 'BNBUSDT', XRP: 'XRPUSDT', AVAX: 'AVAXUSDT',
  DOGE: 'DOGEUSDT', LINK: 'LINKUSDT', ADA: 'ADAUSDT',
};

export interface PrevDay {
  high: number;
  low: number;
  open: number;
  close: number;
}

export function usePrevDayOHLC(coin: Coin) {
  const [prevDay, setPrevDay] = useState<PrevDay | null>(null);

  useEffect(() => {
    fetch(
      `https://api.binance.com/api/v3/klines?symbol=${SYMBOLS[coin]}&interval=1d&limit=2`
    )
      .then(res => res.json())
      .then((data: string[][]) => {
        // data[0] = completed previous day, data[1] = current forming day
        const d = data[0];
        setPrevDay({
          open: parseFloat(d[1]),
          high: parseFloat(d[2]),
          low: parseFloat(d[3]),
          close: parseFloat(d[4]),
        });
      })
      .catch(() => null);
  }, [coin]);

  return prevDay;
}
