import { useState, useCallback, useEffect } from 'react';
import { Chart } from './components/Chart';
import { NewsFeed } from './components/NewsFeed';
import { PriceAlertPanel } from './components/PriceAlertPanel';
import { useKlines } from './hooks/useKlines';
import { useNews } from './hooks/useNews';
import { useFearGreed } from './hooks/useFearGreed';
import { useFundingRate } from './hooks/useFundingRate';
import { usePrevDayOHLC } from './hooks/usePrevDayOHLC';
import { usePriceAlerts } from './hooks/usePriceAlerts';
import { Coin, Timeframe } from './types';

const COINS: Coin[]           = ['BTC', 'ETH', 'SOL'];
const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

const COIN_COLORS: Record<Coin, string> = {
  BTC: '#f7931a', ETH: '#627eea', SOL: '#9945ff',
};

function fgColor(v: number) {
  if (v <= 25) return '#ef4444';
  if (v <= 45) return '#f97316';
  if (v <= 55) return '#eab308';
  if (v <= 75) return '#84cc16';
  return '#22c55e';
}

export default function App() {
  const [coin, setCoin]           = useState<Coin>('BTC');
  const [timeframe, setTimeframe] = useState<Timeframe>('15m');
  const [showEMA20, setShowEMA20]   = useState(true);
  const [showEMA50, setShowEMA50]   = useState(true);
  const [showEMA200, setShowEMA200] = useState(false);
  const [showRSI, setShowRSI]       = useState(false);
  const [priceFlash, setPriceFlash] = useState(false);

  const [scrollToTime, setScrollToTime]           = useState<number | null>(null);
  const [highlightedNewsId, setHighlightedNewsId] = useState<string | null>(null);

  const { candles, liveCandle } = useKlines(coin, timeframe);
  const news        = useNews();
  const fearGreed   = useFearGreed();
  const fundingRate = useFundingRate(coin);
  const prevDay     = usePrevDayOHLC(coin);
  const { alerts, addAlert, removeAlert, triggered, clearTriggered } =
    usePriceAlerts(liveCandle?.close ?? null, coin);

  useEffect(() => {
    if (triggered) {
      setPriceFlash(true);
      const t = setTimeout(() => { setPriceFlash(false); clearTriggered(); }, 1500);
      return () => clearTimeout(t);
    }
  }, [triggered, clearTriggered]);

  const price     = liveCandle?.close ?? candles[candles.length - 1]?.close;
  const prevClose = candles[candles.length - 2]?.close;
  const change    = price && prevClose ? ((price - prevClose) / prevClose) * 100 : null;
  const isUp      = change !== null ? change >= 0 : true;

  const handleCandleClick = useCallback((time: number) => {
    if (news.length === 0) return;
    const closest = news.reduce((prev, cur) =>
      Math.abs(cur.publishedAt - time) < Math.abs(prev.publishedAt - time) ? cur : prev
    );
    setHighlightedNewsId(closest.id);
  }, [news]);

  const handleNewsClick = useCallback((publishedAt: number, id: string) => {
    setScrollToTime(publishedAt);
    setHighlightedNewsId(id);
    setTimeout(() => setScrollToTime(null), 500);
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-hex">⬡</span>
          <span className="brand-text">TRADE MONITOR</span>
        </div>

        <div className="controls">
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

          <div className="selector indicator-toggles">
            <button className={`sel-btn ema20-btn ${showEMA20 ? 'ema20-on' : ''}`} onClick={() => setShowEMA20(v => !v)}>20</button>
            <button className={`sel-btn ema50-btn ${showEMA50 ? 'ema50-on' : ''}`} onClick={() => setShowEMA50(v => !v)}>50</button>
            <button className={`sel-btn ema200-btn ${showEMA200 ? 'ema200-on' : ''}`} onClick={() => setShowEMA200(v => !v)}>200</button>
            <button className={`sel-btn rsi-btn ${showRSI ? 'rsi-on' : ''}`} onClick={() => setShowRSI(v => !v)}>RSI</button>
          </div>
        </div>

        <div className="price-area">
          {price ? (
            <>
              <span className={`price ${priceFlash ? 'price--flash' : ''}`} style={{ color: COIN_COLORS[coin] }}>
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

        <div className="widgets">
          {fearGreed && (
            <div className="widget">
              <span className="widget-label">F&G</span>
              <span className="widget-value" style={{ color: fgColor(fearGreed.value) }}>
                {fearGreed.value}
              </span>
              <span className="widget-sub" style={{ color: fgColor(fearGreed.value) }}>
                {fearGreed.label.replace('Extreme ', 'EX ')}
              </span>
            </div>
          )}
          {fundingRate && (
            <div className="widget">
              <span className="widget-label">FUND</span>
              <span className={`widget-value ${fundingRate.rate >= 0 ? 'up' : 'down'}`}>
                {(fundingRate.rate * 100).toFixed(4)}%
              </span>
            </div>
          )}
          <div className="widget-legend">
            <span style={{ color: '#f59e0b' }}>●</span>
            <span>news</span>
          </div>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <NewsFeed
            news={news}
            highlightedId={highlightedNewsId}
            onItemClick={handleNewsClick}
          />
          <PriceAlertPanel
            coin={coin}
            currentPrice={price ?? null}
            alerts={alerts}
            onAdd={addAlert}
            onRemove={removeAlert}
          />
        </aside>

        <section className="chart-area">
          <Chart
            candles={candles}
            liveCandle={liveCandle}
            news={news}
            timeframe={timeframe}
            coin={coin}
            showEMA20={showEMA20}
            showEMA50={showEMA50}
            showEMA200={showEMA200}
            showRSI={showRSI}
            prevDay={prevDay}
            scrollToTime={scrollToTime}
            onCandleClick={handleCandleClick}
          />
        </section>
      </main>
    </div>
  );
}
