import { useEffect, useRef } from 'react';
import { Coin, Timeframe } from '../types';

const SYMBOL_MAP: Record<Coin, string> = {
  BTC: 'BINANCE:BTCUSDT',  ETH: 'BINANCE:ETHUSDT',  BNB: 'BINANCE:BNBUSDT',
  XRP: 'BINANCE:XRPUSDT',  LTC: 'BINANCE:LTCUSDT',  TRX: 'BINANCE:TRXUSDT',
  SOL: 'BINANCE:SOLUSDT',  AVAX: 'BINANCE:AVAXUSDT', DOT: 'BINANCE:DOTUSDT',
  LINK: 'BINANCE:LINKUSDT', ATOM: 'BINANCE:ATOMUSDT', NEAR: 'BINANCE:NEARUSDT',
  UNI: 'BINANCE:UNIUSDT',  ADA: 'BINANCE:ADAUSDT',
  DOGE: 'BINANCE:DOGEUSDT', SUI: 'BINANCE:SUIUSDT',  APT: 'BINANCE:APTUSDT',
  ARB: 'BINANCE:ARBUSDT',  OP: 'BINANCE:OPUSDT',    INJ: 'BINANCE:INJUSDT',
  PAXG: 'BINANCE:PAXGUSDT',
};

const TF_MAP: Record<Timeframe, string> = {
  '1m': '1', '5m': '5', '15m': '15',
  '1h': '60', '4h': '240', '1d': 'D',
};

// Load TV script once globally
let tvLoaded = false;
const tvQueue: (() => void)[] = [];

function loadTV(cb: () => void) {
  if (tvLoaded) { cb(); return; }
  tvQueue.push(cb);
  if (tvQueue.length > 1) return; // already loading
  const s = document.createElement('script');
  s.src = 'https://s3.tradingview.com/tv.js';
  s.async = true;
  s.onload = () => {
    tvLoaded = true;
    tvQueue.forEach(f => f());
    tvQueue.length = 0;
  };
  document.head.appendChild(s);
}

interface Props {
  coin: Coin;
  timeframe: Timeframe;
  theme: 'light' | 'dark';
}

export function TradingViewChart({ coin, timeframe, theme }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Unique ID per render to avoid stale container conflicts
    const id = `tv-apex-${Date.now()}`;
    containerRef.current.innerHTML = `<div id="${id}" style="width:100%;height:100%"></div>`;

    loadTV(() => {
      if (!containerRef.current || !document.getElementById(id)) return;
      new (window as any).TradingView.widget({
        container_id:        id,
        autosize:            true,
        symbol:              SYMBOL_MAP[coin],
        interval:            TF_MAP[timeframe],
        timezone:            'Etc/UTC',
        theme:               theme,
        style:               '1',          // candlestick
        locale:              'en',
        toolbar_bg:          theme === 'dark' ? '#0f0f0f' : '#ffffff',
        enable_publishing:   false,
        allow_symbol_change: false,
        hide_legend:         false,
        save_image:          false,
        withdateranges:      false,
        hide_volume:         false,
      });
    });

    return () => {
      if (containerRef.current) containerRef.current.innerHTML = '';
    };
  }, [coin, timeframe, theme]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
}
