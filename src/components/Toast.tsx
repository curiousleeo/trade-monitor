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
  BTC: '#f7931a', ETH: '#627eea', BNB: '#f3ba2f', XRP: '#346aa9', LTC: '#bfbbbb', TRX: '#ef0027',
  SOL: '#9945ff', AVAX: '#e84142', DOT: '#e6007a', LINK: '#2a5ada', ATOM: '#6f4cff', NEAR: '#00c08b', UNI: '#ff007a', ADA: '#0033ad',
  DOGE: '#c2a633', SUI: '#4da2ff', APT: '#00c2a8', ARB: '#28a0f0', OP: '#ff0420', INJ: '#00b4d8',
  PAXG: '#d4a843',
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
