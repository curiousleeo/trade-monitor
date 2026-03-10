import { useEffect } from 'react';
import { Coin } from '../types';

interface Props {
  coin: Coin;
  direction: 'above' | 'below';
  alertPrice: number;
  currentPrice: number;
  onClose: () => void;
}

const COIN_COLORS: Record<Coin, string> = {
  BTC: '#f7931a', ETH: '#627eea', SOL: '#9945ff',
  BNB: '#f3ba2f', XRP: '#346aa9', AVAX: '#e84142',
  DOGE: '#c2a633', LINK: '#2a5ada', ADA: '#0033ad',
};

export function Toast({ coin, direction, alertPrice, currentPrice, onClose }: Props) {
  useEffect(() => {
    const t = setTimeout(onClose, 5000);
    return () => clearTimeout(t);
  }, [onClose]);

  const isAbove = direction === 'above';
  const coinColor = COIN_COLORS[coin];

  function fmt(v: number): string {
    return v >= 1000
      ? `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
      : `$${v.toFixed(2)}`;
  }

  return (
    <div className="toast" role="alert">
      <div className="toast-accent" style={{ background: isAbove ? 'var(--green)' : 'var(--red)' }} />
      <div className="toast-icon">🔔</div>
      <div className="toast-body">
        <div className="toast-title">
          <span style={{ color: coinColor, fontWeight: 700 }}>{coin}</span>
          {' '}Price Alert
        </div>
        <div className="toast-msg">
          <span className={isAbove ? 'up' : 'down'}>
            {isAbove ? '▲' : '▼'} {fmt(alertPrice)}
          </span>
          <span className="toast-now"> · Now {fmt(currentPrice)}</span>
        </div>
      </div>
      <button className="toast-close" onClick={onClose} aria-label="Dismiss">✕</button>
    </div>
  );
}
