export interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface NewsItem {
  id: string;
  title: string;
  url: string;
  source: string;
  publishedAt: number; // unix seconds
  categories: string;
}

export type Coin = 'BTC' | 'ETH' | 'SOL';
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
