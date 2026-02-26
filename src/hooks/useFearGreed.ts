import { useState, useEffect } from 'react';
import { FearGreed } from '../types';

export function useFearGreed() {
  const [data, setData] = useState<FearGreed | null>(null);

  async function fetchFG() {
    try {
      const res = await fetch('https://api.alternative.me/fng/?limit=1');
      const json = await res.json();
      const item = json.data?.[0];
      if (item) {
        setData({ value: Number(item.value), label: item.value_classification });
      }
    } catch {}
  }

  useEffect(() => {
    fetchFG();
    const id = setInterval(fetchFG, 5 * 60_000);
    return () => clearInterval(id);
  }, []);

  return data;
}
