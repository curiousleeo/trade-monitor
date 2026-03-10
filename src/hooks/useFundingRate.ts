import { useState, useEffect } from 'react';
import { Coin, FundingRate } from '../types';

const SYMBOLS: Record<Coin, string> = {
  BTC: 'BTCUSDT',  ETH: 'ETHUSDT',  BNB: 'BNBUSDT',  XRP: 'XRPUSDT',  LTC: 'LTCUSDT',  TRX: 'TRXUSDT',
  SOL: 'SOLUSDT',  AVAX: 'AVAXUSDT', DOT: 'DOTUSDT',  LINK: 'LINKUSDT', ATOM: 'ATOMUSDT', NEAR: 'NEARUSDT',
  UNI: 'UNIUSDT',  ADA: 'ADAUSDT',
  DOGE: 'DOGEUSDT', SUI: 'SUIUSDT',  APT: 'APTUSDT',  ARB: 'ARBUSDT',  OP: 'OPUSDT',    INJ: 'INJUSDT',
  PAXG: 'PAXGUSDT',
};

export function useFundingRate(coin: Coin) {
  const [data, setData] = useState<FundingRate | null>(null);

  async function fetchRate() {
    try {
      const res = await fetch(
        `https://fapi.binance.com/fapi/v1/premiumIndex?symbol=${SYMBOLS[coin]}`
      );
      const json = await res.json();
      setData({
        rate: parseFloat(json.lastFundingRate),
        nextFundingTime: json.nextFundingTime,
      });
    } catch {}
  }

  useEffect(() => {
    fetchRate();
    const id = setInterval(fetchRate, 60_000);
    return () => clearInterval(id);
  }, [coin]);

  return data;
}
