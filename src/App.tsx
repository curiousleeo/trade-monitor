import { useState, useCallback, useEffect } from 'react';
import { Chart }           from './components/Chart';
import { NewsFeed }        from './components/NewsFeed';
import { PriceAlertPanel } from './components/PriceAlertPanel';
import { CoinCard }        from './components/CoinCard';
import { StatsStrip }      from './components/StatsStrip';
import { ChartToolbar }    from './components/ChartToolbar';
import { AIPanel }         from './components/AIPanel';
import { useKlines }       from './hooks/useKlines';
import { useNews }         from './hooks/useNews';
import { useFearGreed }    from './hooks/useFearGreed';
import { useFundingRate }  from './hooks/useFundingRate';
import { usePrevDayOHLC }  from './hooks/usePrevDayOHLC';
import { usePriceAlerts }  from './hooks/usePriceAlerts';
import { use24hTicker }    from './hooks/use24hTicker';
import { useAllCandles }   from './hooks/useAllCandles';
import { useAITrader }     from './hooks/useAITrader';
import { Coin, Timeframe } from './types';

const COINS: Coin[] = ['BTC', 'ETH', 'SOL'];

export default function App() {
  const [coin, setCoin]                     = useState<Coin>('BTC');
  const [timeframe, setTimeframe]           = useState<Timeframe>('15m');
  const [showEMA20, setShowEMA20]           = useState(true);
  const [showEMA50, setShowEMA50]           = useState(true);
  const [showEMA200, setShowEMA200]         = useState(false);
  const [showRSI, setShowRSI]               = useState(false);
  const [showNewsMarkers, setShowNewsMarkers] = useState(true);
  const [priceFlash, setPriceFlash]         = useState(false);
  const [scrollToTime, setScrollToTime]     = useState<number | null>(null);
  const [highlightedNewsId, setHighlightedNewsId] = useState<string | null>(null);
  const [sidebarTab, setSidebarTab]         = useState<'news' | 'ai'>('news');

  // ── Data hooks ─────────────────────────────────────────────────────────
  const { candles, liveCandle } = useKlines(coin, timeframe);
  const tickers     = use24hTicker();
  const news        = useNews();
  const fearGreed   = useFearGreed();
  const fundingRate = useFundingRate(coin);

  // Funding rates for all 3 coins (AI trader needs them per-coin)
  const fundingBTC = useFundingRate('BTC');
  const fundingETH = useFundingRate('ETH');
  const fundingSOL = useFundingRate('SOL');
  const fundingRates = { BTC: fundingBTC, ETH: fundingETH, SOL: fundingSOL };

  const prevDay     = usePrevDayOHLC(coin);
  const { alerts, addAlert, removeAlert, triggered, clearTriggered } =
    usePriceAlerts(tickers[coin]?.price ?? null, coin);

  // REST candles snapshot for all 3 coins (AI uses for non-active coins)
  const allCandles = useAllCandles(timeframe);

  // ── AI Trader ──────────────────────────────────────────────────────────
  const { portfolio, predictions, tfMatrix, tradeMarkers, resetPortfolio } = useAITrader({
    allCandles,
    activeCandles:   candles,
    activeCoin:      coin,
    activeTimeframe: timeframe,
    tickers,
    fearGreed,
    fundingRates,
    news,
    prevDay,
  });

  // Flash coin card when alert triggers
  useEffect(() => {
    if (triggered) {
      setPriceFlash(true);
      const t = setTimeout(() => { setPriceFlash(false); clearTriggered(); }, 1500);
      return () => clearTimeout(t);
    }
  }, [triggered, clearTriggered]);

  // Chart click → highlight nearest news item
  const handleCandleClick = useCallback((time: number) => {
    if (news.length === 0) return;
    const closest = news.reduce((prev, cur) =>
      Math.abs(cur.publishedAt - time) < Math.abs(prev.publishedAt - time) ? cur : prev
    );
    setHighlightedNewsId(closest.id);
  }, [news]);

  // News click → scroll chart to that candle
  const handleNewsClick = useCallback((publishedAt: number, id: string) => {
    setScrollToTime(publishedAt);
    setHighlightedNewsId(id);
    setTimeout(() => setScrollToTime(null), 500);
  }, []);

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <span className="brand-hex">⬡</span>
          <span className="brand-text">Trade Monitor</span>
          <span className="brand-apex">· APEX AI</span>
        </div>

        <div className="coin-cards">
          {COINS.map(c => (
            <CoinCard
              key={c}
              coin={c}
              ticker={tickers[c]}
              active={coin === c}
              flash={coin === c && priceFlash}
              onClick={setCoin}
            />
          ))}
        </div>
      </header>

      {/* ── Stats strip ── */}
      <StatsStrip
        fearGreed={fearGreed}
        fundingRate={fundingRate}
        ticker={tickers[coin]}
      />

      {/* ── Body ── */}
      <main className="main">
        <aside className="sidebar">

          {/* Sidebar tab bar */}
          <div className="sidebar-tabs">
            <button
              className={`sidebar-tab ${sidebarTab === 'news' ? 'active' : ''}`}
              onClick={() => setSidebarTab('news')}
            >
              📰 NEWS
            </button>
            <button
              className={`sidebar-tab ${sidebarTab === 'ai' ? 'active' : ''}`}
              onClick={() => setSidebarTab('ai')}
            >
              ⚡ APEX
            </button>
          </div>

          {sidebarTab === 'news' ? (
            <>
              <NewsFeed
                news={news}
                highlightedId={highlightedNewsId}
                onItemClick={handleNewsClick}
              />
              <PriceAlertPanel
                coin={coin}
                currentPrice={tickers[coin]?.price ?? null}
                alerts={alerts}
                onAdd={addAlert}
                onRemove={removeAlert}
              />
            </>
          ) : (
            <AIPanel
              portfolio={portfolio}
              predictions={predictions}
              tfMatrix={tfMatrix}
              activeCoin={coin}
              livePrice={tickers[coin]?.price ?? null}
              onReset={resetPortfolio}
            />
          )}
        </aside>

        <section className="chart-area">
          <ChartToolbar
            timeframe={timeframe}    onTimeframe={setTimeframe}
            showEMA20={showEMA20}    onEMA20={setShowEMA20}
            showEMA50={showEMA50}    onEMA50={setShowEMA50}
            showEMA200={showEMA200}  onEMA200={setShowEMA200}
            showRSI={showRSI}        onRSI={setShowRSI}
            showNewsMarkers={showNewsMarkers} onNewsMarkers={setShowNewsMarkers}
          />
          <div className="chart-canvas">
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
              showNewsMarkers={showNewsMarkers}
              prevDay={prevDay}
              scrollToTime={scrollToTime}
              onCandleClick={handleCandleClick}
              tradeMarkers={tradeMarkers}
              openTrades={portfolio.openTrades}
            />
          </div>
        </section>
      </main>

    </div>
  );
}
