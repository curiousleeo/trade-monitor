/**
 * Three parallel news streams — all crypto-price-relevant:
 *
 *  1. Coin-specific     — BTC, ETH, SOL (direct drivers)
 *  2. Crypto macro      — Regulation, Exchange, Hack, Stablecoin,
 *                         Market, Blockchain, Mining, Trading
 *  3. Global macro/geo  — BBC Business + CNBC Economy RSS via rss2json proxy
 *                         Filtered to keywords that move risk assets:
 *                         Fed, inflation, war, sanctions, dollar, banking
 *
 * Streams 1+2 refresh every 90s.
 * Stream 3 refreshes every 5 min (rss2json free tier: 10k req/month).
 */

import { useState, useEffect, useRef } from 'react';
import { NewsItem } from '../types';

// ── Stream 1 & 2: CryptoCompare ───────────────────────────────────────────────

const CC_BASE          = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest';
const COIN_CATEGORIES  = 'BTC,ETH,SOL';
const MACRO_CATEGORIES = 'Regulation,Exchange,Hack,Stablecoin,Market,Blockchain,Mining,Trading';

function parseCCItems(data: Record<string, unknown>[]): NewsItem[] {
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

async function fetchCC(categories: string): Promise<NewsItem[]> {
  const res  = await fetch(`${CC_BASE}&categories=${categories}`);
  const data = await res.json();
  if (!Array.isArray(data?.Data)) return [];
  return parseCCItems(data.Data as Record<string, unknown>[]);
}

// ── Stream 3: Global macro / geopolitical via Reuters Business RSS ────────────

/**
 * Keywords that historically correlate with crypto price swings.
 * ANY match → item is included in the feed tagged as MACRO or GEO.
 */
const MACRO_KEYWORDS = [
  // Monetary policy
  'federal reserve', 'fed rate', 'interest rate', 'rate hike', 'rate cut',
  'jerome powell', 'fomc', 'quantitative', 'tightening', 'tapering',
  // Inflation / economy
  'inflation', 'cpi', 'ppi', 'gdp', 'recession', 'stagflation',
  'consumer price', 'job report', 'unemployment', 'nonfarm', 'payroll',
  // Dollar & liquidity
  'dollar index', 'dxy', 'us dollar', 'dollar strength', 'liquidity',
  'treasury yield', 'bond yield', '10-year', 'yield curve',
  // Banking & financial system
  'bank collapse', 'bank failure', 'banking crisis', 'svb', 'credit suisse',
  'fdic', 'financial crisis', 'bail', 'default', 'debt ceiling',
  // Geopolitics & war
  'war', 'conflict', 'invasion', 'missile', 'airstrike', 'military',
  'sanctions', 'russia', 'ukraine', 'iran', 'north korea', 'taiwan',
  'tariff', 'trade war', 'geopolit',
  // Stock market correlation
  'nasdaq', 's&p 500', 'risk-off', 'risk off', 'market crash', 'stock sell',
  'market selloff', 'equity sell',
  // Commodities (oil correlates with risk sentiment)
  'oil price', 'crude oil', 'opec', 'energy crisis',
];

const GEO_KEYWORDS = [
  'war', 'conflict', 'invasion', 'missile', 'airstrike', 'military',
  'sanctions', 'geopolit', 'nuclear', 'escalat',
];

// Free RSS→JSON proxy (no API key, 10k req/month free)
// Reuters RSS feeds were deprecated; using BBC Business + CNBC Economy instead
const RSS_FEEDS = [
  'http://feeds.bbci.co.uk/news/business/rss.xml',
  'https://www.cnbc.com/id/100003114/device/rss/rss.html',
];
const RSS2JSON = 'https://api.rss2json.com/v1/api.json?count=20&rss_url=';

interface RssItem {
  title: string;
  link: string;
  pubDate: string;
  author?: string;
}

function isMacroRelevant(title: string): boolean {
  const lower = title.toLowerCase();
  return MACRO_KEYWORDS.some(kw => lower.includes(kw));
}

function macroCategory(title: string): string {
  const lower = title.toLowerCase();
  if (GEO_KEYWORDS.some(kw => lower.includes(kw))) return 'GeoPolitic';
  return 'MacroEcon';
}

async function fetchMacroRss(): Promise<NewsItem[]> {
  const results: NewsItem[] = [];
  await Promise.allSettled(
    RSS_FEEDS.map(async feed => {
      const res  = await fetch(`${RSS2JSON}${encodeURIComponent(feed)}`);
      const data = await res.json();
      if (!Array.isArray(data?.items)) return;
      const sourceName = feed.includes('bbc') ? 'BBC' : 'CNBC';
      (data.items as RssItem[]).forEach(item => {
        if (!isMacroRelevant(item.title)) return;
        const pubSec = Math.floor(new Date(item.pubDate).getTime() / 1000);
        results.push({
          id:          `rss-${item.link}`,
          title:       item.title,
          url:         item.link,
          source:      sourceName,
          publishedAt: pubSec,
          categories:  macroCategory(item.title),
        });
      });
    })
  );
  return results;
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useNews() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const seenIds         = useRef(new Set<string>());

  function ingest(items: NewsItem[]) {
    const fresh = items.filter(item => !seenIds.current.has(item.id));
    if (fresh.length === 0) return;
    fresh.forEach(item => seenIds.current.add(item.id));

    setNews(prev => {
      const merged = [...fresh, ...prev];
      const seen   = new Set<string>();
      return merged
        .filter(item => { if (seen.has(item.id)) return false; seen.add(item.id); return true; })
        .sort((a, b) => b.publishedAt - a.publishedAt)
        .slice(0, 120);
    });
  }

  async function refreshCrypto() {
    const [coinRes, macroRes] = await Promise.allSettled([
      fetchCC(COIN_CATEGORIES),
      fetchCC(MACRO_CATEGORIES),
    ]);
    const items = [
      ...(coinRes.status  === 'fulfilled' ? coinRes.value  : []),
      ...(macroRes.status === 'fulfilled' ? macroRes.value : []),
    ];
    ingest(items);
  }

  async function refreshMacro() {
    const items = await fetchMacroRss().catch(() => []);
    ingest(items);
  }

  useEffect(() => {
    // Crypto: immediate + every 90s
    refreshCrypto();
    const cryptoId = setInterval(refreshCrypto, 90_000);

    // Global macro: immediate + every 5 min (rss2json free tier friendly)
    refreshMacro();
    const macroId = setInterval(refreshMacro, 300_000);

    return () => { clearInterval(cryptoId); clearInterval(macroId); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return news;
}
