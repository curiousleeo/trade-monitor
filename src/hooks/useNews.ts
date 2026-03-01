/**
 * Fetches two streams in parallel and merges them:
 *
 *  1. Coin-specific  — BTC, ETH, SOL  (direct price drivers)
 *  2. Macro/market   — Regulation, Exchange, Hack, Stablecoin,
 *                      Market, Blockchain, Mining, Trading
 *                      (indirect but often bigger price movers)
 *
 * Both streams refresh every 90 s. Items are deduped by ID and
 * sorted newest-first, capped at 100.
 */

import { useState, useEffect, useRef } from 'react';
import { NewsItem } from '../types';

const BASE = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest';

const COIN_CATEGORIES  = 'BTC,ETH,SOL';
const MACRO_CATEGORIES = 'Regulation,Exchange,Hack,Stablecoin,Market,Blockchain,Mining,Trading';

function parseItems(data: Record<string, unknown>[]): NewsItem[] {
  return data.map(item => ({
    id:          String(item.id),
    title:       item.title as string,
    url:         item.url as string,
    source:      (item.source_info as Record<string, string> | undefined)?.name
                 ?? (item.source as string),
    publishedAt: item.published_on as number,
    categories:  item.categories as string,
  }));
}

async function fetchStream(categories: string): Promise<NewsItem[]> {
  const res  = await fetch(`${BASE}&categories=${categories}`);
  const data = await res.json();
  if (!Array.isArray(data?.Data)) return [];
  return parseItems(data.Data as Record<string, unknown>[]);
}

export function useNews() {
  const [news, setNews]   = useState<NewsItem[]>([]);
  const seenIds           = useRef(new Set<string>());

  async function refresh() {
    try {
      const [coinItems, macroItems] = await Promise.allSettled([
        fetchStream(COIN_CATEGORIES),
        fetchStream(MACRO_CATEGORIES),
      ]);

      const combined: NewsItem[] = [
        ...(coinItems.status  === 'fulfilled' ? coinItems.value  : []),
        ...(macroItems.status === 'fulfilled' ? macroItems.value : []),
      ];

      // Dedupe across both streams (same story tagged both ways)
      const fresh = combined.filter(item => !seenIds.current.has(item.id));
      if (fresh.length === 0) return;
      fresh.forEach(item => seenIds.current.add(item.id));

      setNews(prev => {
        const merged = [...fresh, ...prev];
        // Sort newest-first, dedup again (prev might have overlaps after refresh)
        const seen = new Set<string>();
        const deduped = merged.filter(item => {
          if (seen.has(item.id)) return false;
          seen.add(item.id);
          return true;
        });
        return deduped
          .sort((a, b) => b.publishedAt - a.publishedAt)
          .slice(0, 100);
      });
    } catch (err) {
      console.error('News fetch error:', err);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 90_000); // 90s — two streams = ~2 req/refresh
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return news;
}
