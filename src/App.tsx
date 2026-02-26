import { useState } from 'react';
import { Chart } from './components/Chart';
import { NewsFeed } from './components/NewsFeed';
import { useKlines } from './hooks/useKlines';
import { useNews } from './hooks/useNews';
import { Coin, Timeframe } from './types';

const COINS: Coin[] = ['BTC', 'ETH', 'SOL'];
const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

const COIN_COLORS: Record<Coin, string> = {
  BTC: '#f7931a',
  ETH: '#627eea',
  SOL: '#9945ff',
};

export default function App() {
  const [coin, setCoin] = useState<Coin>('BTC');
  const [timeframe, setTimeframe] = useState<Timeframe>('15m');

  const { candles, liveCandle } = useKlines(coin, timeframe);
  const news = useNews();

  const price = liveCandle?.close ?? candles[candles.length - 1]?.close;
  const prevClose = candles[candles.length - 2]?.close;
  const change = price && prevClose ? ((price - prevClose) / prevClose) * 100 : null;
  const isUp = change !== null ? change >= 0 : true;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <span className="brand-hex">⬡</span>
          TRADE MONITOR
        </div>

        <div className="controls">
          {/* Coin selector */}
          <div className="selector">
            {COINS.map(c => (
              <button
                key={c}
                className={`sel-btn ${coin === c ? 'active' : ''}`}
                style={coin === c ? { color: COIN_COLORS[c], borderColor: COIN_COLORS[c] } : {}}
                onClick={() => setCoin(c)}
              >
                {c}
              </button>
            ))}
          </div>

          {/* Timeframe selector */}
          <div className="selector">
            {TIMEFRAMES.map(tf => (
              <button
                key={tf}
                className={`sel-btn ${timeframe === tf ? 'active' : ''}`}
                onClick={() => setTimeframe(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        {/* Live price */}
        <div className="price-area">
          {price ? (
            <>
              <span className="price" style={{ color: COIN_COLORS[coin] }}>
                ${price.toLocaleString(undefined, {
                  minimumFractionDigits: coin === 'BTC' ? 0 : 2,
                  maximumFractionDigits: coin === 'BTC' ? 0 : 2,
                })}
              </span>
              {change !== null && (
                <span className={`change ${isUp ? 'up' : 'down'}`}>
                  {isUp ? '▲' : '▼'} {Math.abs(change).toFixed(2)}%
                </span>
              )}
            </>
          ) : (
            <span className="price-loading">—</span>
          )}
        </div>

        <div className="header-info">
          <span className="marker-legend">
            <span className="marker-dot">●</span> news event
          </span>
        </div>
      </header>

      {/* ── Body ── */}
      <main className="main">
        <aside className="sidebar">
          <NewsFeed news={news} />
        </aside>
        <section className="chart-area">
          <Chart
            candles={candles}
            liveCandle={liveCandle}
            news={news}
            timeframe={timeframe}
            coin={coin}
          />
        </section>
      </main>
    </div>
  );
}
