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

export interface TickerData {
  price: number;
  change24h: number;   // percentage e.g. 2.41
  volume24h: number;   // in USD
  high24h: number;
  low24h: number;
}

export type Sentiment = 'bullish' | 'bearish' | 'neutral';
export type Coin =
  // Large cap
  | 'BTC' | 'ETH' | 'BNB' | 'XRP' | 'LTC' | 'TRX'
  // Mid cap momentum
  | 'SOL' | 'AVAX' | 'DOT' | 'LINK' | 'ATOM' | 'NEAR' | 'UNI' | 'ADA'
  // High beta / momentum
  | 'DOGE' | 'SUI' | 'APT' | 'ARB' | 'OP' | 'INJ'
  // Gold proxy (PAXG tracks spot gold 1:1 via Paxos)
  | 'PAXG';
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
export type TradeDirection = 'LONG' | 'SHORT';
export type TradeStatus = 'OPEN' | 'CLOSED_TP' | 'CLOSED_SL' | 'CLOSED_MANUAL';
export type PredictionDirection = 'LONG' | 'SHORT' | 'NEUTRAL';

export interface Signal {
  name: string;
  value: number;       // -100 to +100
  weight: number;      // 0–1, sum of all = 1
  description: string;
}

export interface Prediction {
  id: string;
  coin: Coin;
  timeframe: Timeframe;
  timestamp: number;
  direction: PredictionDirection;
  confidence: number;               // 50–100
  entryZone: [number, number];      // [low, high]
  targetPrice: number;
  stopPrice: number;
  atr: number;
  signals: Signal[];
  reasoning: string;
  resolved: boolean;
  resolvedAt?: number;
  accurate?: boolean;               // price reached target before stop
}

export interface Trade {
  id: string;
  predictionId: string;
  coin: Coin;
  direction: TradeDirection;
  entryPrice: number;
  entryTime: number;
  stopLoss: number;
  takeProfit: number;
  size: number;          // USD amount
  status: TradeStatus;
  exitPrice?: number;
  exitTime?: number;
  pnl?: number;          // USD
  pnlPct?: number;       // percentage
}

export interface Portfolio {
  balance: number;
  startBalance: number;
  openTrades: Trade[];
  closedTrades: Trade[];
  predictions: Prediction[];
}

export interface TFBias {
  timeframe: Timeframe;
  direction: PredictionDirection;
  score: number;        // 0–100
}
