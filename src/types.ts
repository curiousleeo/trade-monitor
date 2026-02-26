export interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface NewsItem {
  id: string;
  title: string;
  url: string;
  source: string;
  publishedAt: number; // unix seconds
  categories: string;
}

export interface PriceAlert {
  id: string;
  coin: Coin;
  price: number;
  direction: 'above' | 'below';
  triggered: boolean;
}

export interface FundingRate {
  rate: number;
  nextFundingTime: number;
}

export interface FearGreed {
  value: number;
  label: string;
}

export type Sentiment = 'bullish' | 'bearish' | 'neutral';
export type Coin = 'BTC' | 'ETH' | 'SOL';
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
