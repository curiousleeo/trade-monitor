import { useState } from 'react';
import { PriceAlert, Coin } from '../types';

interface Props {
  coin: Coin;
  currentPrice: number | null;
  alerts: PriceAlert[];
  onAdd: (price: number, direction: 'above' | 'below') => void;
  onRemove: (id: string) => void;
}

export function PriceAlertPanel({ coin, currentPrice, alerts, onAdd, onRemove }: Props) {
  const [inputPrice, setInputPrice] = useState('');
  const [direction, setDirection] = useState<'above' | 'below'>('above');

  const handleAdd = () => {
    const p = parseFloat(inputPrice);
    if (!isNaN(p) && p > 0) {
      onAdd(p, direction);
      setInputPrice('');
    }
  };

  const coinAlerts = alerts.filter(a => a.coin === coin);

  return (
    <div className="alert-panel">
      <div className="panel-header">
        <span>🔔</span>
        <span>PRICE ALERTS</span>
        <span className="news-count">{coinAlerts.filter(a => !a.triggered).length}</span>
      </div>

      <div className="alert-form">
        <select
          className="alert-select"
          value={direction}
          onChange={e => setDirection(e.target.value as 'above' | 'below')}
        >
          <option value="above">Above</option>
          <option value="below">Below</option>
        </select>
        <input
          className="alert-input"
          type="number"
          placeholder={currentPrice ? `e.g. ${Math.round(currentPrice)}` : 'Price'}
          value={inputPrice}
          onChange={e => setInputPrice(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAdd()}
        />
        <button className="alert-add-btn" onClick={handleAdd}>+</button>
      </div>

      <div className="alert-list">
        {coinAlerts.length === 0 ? (
          <div className="news-empty" style={{ fontSize: '10px' }}>No alerts set</div>
        ) : (
          coinAlerts.map(alert => (
            <div key={alert.id} className={`alert-item ${alert.triggered ? 'alert-triggered' : ''}`}>
              <span className={`alert-dir ${alert.direction}`}>
                {alert.direction === 'above' ? '▲' : '▼'}
              </span>
              <span className="alert-price">
                ${alert.price.toLocaleString()}
              </span>
              {alert.triggered && <span className="alert-hit">HIT</span>}
              <button className="alert-remove" onClick={() => onRemove(alert.id)}>×</button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
