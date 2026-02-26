import { useState, useEffect, useRef } from 'react';
import { NewsItem } from '../types';

const NEWS_URL =
  'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH,SOL&sortOrder=latest';

export function useNews() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const seenIds = useRef(new Set<string>());

  async function fetchNews() {
    try {
      const res = await fetch(NEWS_URL);
      const data = await res.json();
      if (!data.Data) return;

      const items: NewsItem[] = data.Data.slice(0, 40).map((item: Record<string, unknown>) => ({
        id: String(item.id),
        title: item.title as string,
        url: item.url as string,
        source: (item.source_info as Record<string, string> | undefined)?.name ?? (item.source as string),
        publishedAt: item.published_on as number,
        categories: item.categories as string,
      }));

      const fresh = items.filter(item => !seenIds.current.has(item.id));
      fresh.forEach(item => seenIds.current.add(item.id));

      if (fresh.length > 0) {
        setNews(prev => [...fresh, ...prev].slice(0, 60));
      }
    } catch (err) {
      console.error('News fetch failed:', err);
    }
  }

  useEffect(() => {
    fetchNews();
    const id = setInterval(fetchNews, 60_000);
    return () => clearInterval(id);
  }, []);

  return news;
}
