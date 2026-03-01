import { useEffect, useRef, useState } from 'react';
import { Coin, TickerData } from '../types';

interface Props {
  coin: Coin;
  ticker: TickerData | null;
  active: boolean;
  flash: boolean;
  onClick: (coin: Coin) => void;
}

const COIN_COLORS: Record<Coin, string> = {
  BTC: '#f7931a',
  ETH: '#627eea',
  SOL: '#9945ff',
};

const COIN_ICONS: Record<Coin, string> = {
  BTC: '₿',
  ETH: 'Ξ',
  SOL: '◎',
};

function formatPrice(price: number, coin: Coin): string {
  if (coin === 'BTC') return price.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (coin === 'SOL') return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function CoinCard({ coin, ticker, active, flash, onClick }: Props) {
  const color = COIN_COLORS[coin];
  const isUp = (ticker?.change24h ?? 0) >= 0;

  const prevPrice   = useRef<number | null>(null);
  const tickTimer   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [tickDir, setTickDir] = useState<'up' | 'down' | null>(null);

  useEffect(() => {
    const price = ticker?.price ?? null;
    if (price === null) return;
    const prev = prevPrice.current;
    if (prev !== null && price !== prev) {
      setTickDir(price > prev ? 'up' : 'down');
      if (tickTimer.current) clearTimeout(tickTimer.current);
      tickTimer.current = setTimeout(() => setTickDir(null), 700);
    }
    prevPrice.current = price;
  }, [ticker?.price]);

  return (
    <button
      className={`coin-card ${active ? 'coin-card--active' : ''} ${flash ? 'coin-card--flash' : ''}`}
      style={{ '--coin-color': color } as React.CSSProperties}
      onClick={() => onClick(coin)}
    >
      <div className="coin-card-top">
        <span className="coin-card-icon" style={{ color }}>{COIN_ICONS[coin]}</span>
        <span className="coin-card-name">{coin}</span>
        <span className={`coin-card-price ${active ? '' : 'coin-card-price--muted'} ${tickDir ? `price-tick-${tickDir}` : ''}`}>
          {ticker ? `$${formatPrice(ticker.price, coin)}` : '—'}
        </span>
      </div>
      <div className={`coin-card-change ${isUp ? 'up' : 'down'}`}>
        {ticker ? `${isUp ? '▲' : '▼'} ${Math.abs(ticker.change24h).toFixed(2)}%` : '—'}
      </div>
    </button>
  );
}
