import { useState, useEffect, useCallback } from 'react';
import { PriceAlert, Coin } from '../types';

const STORAGE_KEY = 'trade-monitor-alerts';

function loadAlerts(): PriceAlert[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
  } catch {
    return [];
  }
}

function saveAlerts(alerts: PriceAlert[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts));
}

export function usePriceAlerts(currentPrice: number | null, currentCoin: Coin) {
  const [alerts, setAlerts] = useState<PriceAlert[]>(loadAlerts);
  const [triggered, setTriggered] = useState<string | null>(null); // id of last triggered alert

  // Check alerts on each price tick
  useEffect(() => {
    if (!currentPrice) return;

    let didTrigger = false;
    const updated = alerts.map(alert => {
      if (alert.triggered || alert.coin !== currentCoin) return alert;

      const hit =
        (alert.direction === 'above' && currentPrice >= alert.price) ||
        (alert.direction === 'below' && currentPrice <= alert.price);

      if (hit) {
        didTrigger = true;
        setTriggered(alert.id);

        if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
          new Notification(`⚡ ${alert.coin} Alert`, {
            body: `Price ${alert.direction === 'above' ? '▲' : '▼'} $${alert.price.toLocaleString()}  ·  Now: $${currentPrice.toLocaleString()}`,
          });
        }

        return { ...alert, triggered: true };
      }
      return alert;
    });

    if (didTrigger) {
      setAlerts(updated);
      saveAlerts(updated);
    }
  }, [currentPrice, currentCoin]);

  const addAlert = useCallback(
    (price: number, direction: 'above' | 'below') => {
      // Request notification permission on first add
      if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
        Notification.requestPermission();
      }

      const alert: PriceAlert = {
        id: `${Date.now()}`,
        coin: currentCoin,
        price,
        direction,
        triggered: false,
      };

      setAlerts(prev => {
        const next = [...prev, alert];
        saveAlerts(next);
        return next;
      });
    },
    [currentCoin]
  );

  const removeAlert = useCallback((id: string) => {
    setAlerts(prev => {
      const next = prev.filter(a => a.id !== id);
      saveAlerts(next);
      return next;
    });
  }, []);

  const clearTriggered = useCallback(() => setTriggered(null), []);

  return { alerts, addAlert, removeAlert, triggered, clearTriggered };
}
