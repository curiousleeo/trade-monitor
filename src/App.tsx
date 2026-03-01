import { useState, useCallback, useEffect } from 'react';
import { Chart }           from './components/Chart';
import { NewsFeed }        from './components/NewsFeed';
import { PriceAlertPanel } from './components/PriceAlertPanel';
import { CoinCard }        from './components/CoinCard';
import { StatsStrip }      from './components/StatsStrip';
import { ChartToolbar }    from './components/ChartToolbar';
import { AIPanel }         from './components/AIPanel';
import { HelpModal }       from './components/HelpModal';
import { Toast }           from './components/Toast';
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

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('apex-theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('apex-theme', theme);
  }, [theme]);

  const toggle = useCallback(() => setTheme(t => t === 'dark' ? 'light' : 'dark'), []);
  return { theme, toggle };
}

export default function App() {
  const [coin, setCoin]                       = useState<Coin>('BTC');
  const [timeframe, setTimeframe]             = useState<Timeframe>('15m');
  const [showEMA20, setShowEMA20]             = useState(true);
  const [showEMA50, setShowEMA50]             = useState(true);
  const [showEMA200, setShowEMA200]           = useState(false);
  const [showBB, setShowBB]                   = useState(false);
  const [showRSI, setShowRSI]                 = useState(false);
  const [showNewsMarkers, setShowNewsMarkers] = useState(true);
  const [priceFlash, setPriceFlash]           = useState(false);
  const [scrollToTime, setScrollToTime]       = useState<number | null>(null);
  const [highlightedNewsId, setHighlightedNewsId] = useState<string | null>(null);
  const [sidebarTab, setSidebarTab]           = useState<'news' | 'ai'>('news');
  const [sidebarOpen, setSidebarOpen]         = useState(true);
  const [showHelp, setShowHelp]               = useState(false);
  const [toast, setToast]                     = useState<{ coin: Coin; direction: 'above' | 'below'; alertPrice: number; currentPrice: number } | null>(null);

  const now = useClock();
  const { theme, toggle: toggleTheme } = useTheme();

  // ── Data hooks ─────────────────────────────────────────────────────────
  const { candles, liveCandle } = useKlines(coin, timeframe);
  const tickers     = use24hTicker();
  const news        = useNews();
  const fearGreed   = useFearGreed();
  const fundingRate = useFundingRate(coin);

  const fundingBTC = useFundingRate('BTC');
  const fundingETH = useFundingRate('ETH');
  const fundingSOL = useFundingRate('SOL');
  const fundingRates = { BTC: fundingBTC, ETH: fundingETH, SOL: fundingSOL };

  const prevDay = usePrevDayOHLC(coin);
  const { alerts, addAlert, removeAlert, triggered, clearTriggered } =
    usePriceAlerts(tickers[coin]?.price ?? null, coin);

  const allCandles = useAllCandles(timeframe);

  // ── AI Trader ──────────────────────────────────────────────────────────
  const { portfolio, closedTrades, predictions, tfMatrix, tradeMarkers, resetPortfolio } = useAITrader({
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

  useEffect(() => {
    if (triggered) {
      const alert = alerts.find(a => a.id === triggered);
      if (alert) {
        setToast({
          coin:         alert.coin,
          direction:    alert.direction,
          alertPrice:   alert.price,
          currentPrice: tickers[alert.coin]?.price ?? alert.price,
        });
      }
      setPriceFlash(true);
      const t = setTimeout(() => { setPriceFlash(false); clearTriggered(); }, 1500);
      return () => clearTimeout(t);
    }
  }, [triggered, clearTriggered]);

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

  const timeStr = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const dateStr = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: '2-digit' });

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <span className="brand-hex">◈</span>
          <span className="brand-text">Apex</span>
          <span className="brand-apex">AI Terminal</span>
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

        <div className="header-right">
          <div className="header-status">
            <div className="header-status-dot" />
            <span className="header-status-label">Live</span>
          </div>
          <div className="header-clock">
            <span className="header-clock-time">{timeStr}</span>
            <span className="header-clock-date">{dateStr}</span>
          </div>
          <button
            className="help-btn"
            onClick={() => setShowHelp(true)}
            title="How to use Apex"
          >
            ?
          </button>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀' : '●'}
          </button>
        </div>
      </header>

      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}

      {toast && (
        <div className="toast-container">
          <Toast
            coin={toast.coin}
            direction={toast.direction}
            alertPrice={toast.alertPrice}
            currentPrice={toast.currentPrice}
            onClose={() => setToast(null)}
          />
        </div>
      )}

      {/* ── Stats strip ── */}
      <StatsStrip
        fearGreed={fearGreed}
        fundingRate={fundingRate}
        ticker={tickers[coin]}
      />

      {/* ── Body ── */}
      <main className="main">
        <aside className={`sidebar${sidebarOpen ? '' : ' sidebar--collapsed'}`}>

          <div className="sidebar-tabs">
            <button
              className={`sidebar-tab ${sidebarTab === 'news' ? 'active' : ''}`}
              onClick={() => setSidebarTab('news')}
            >
              News
            </button>
            <button
              className={`sidebar-tab ${sidebarTab === 'ai' ? 'active' : ''}`}
              onClick={() => setSidebarTab('ai')}
            >
              Apex AI
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
              closedTrades={closedTrades}
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
            showBB={showBB}          onBB={setShowBB}
            showRSI={showRSI}        onRSI={setShowRSI}
            showNewsMarkers={showNewsMarkers} onNewsMarkers={setShowNewsMarkers}
            sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(o => !o)}
          />
          <div className="chart-canvas">
            <Chart
              candles={candles}
              liveCandle={liveCandle}
              news={news}
              timeframe={timeframe}
              coin={coin}
              theme={theme}
              showEMA20={showEMA20}
              showEMA50={showEMA50}
              showEMA200={showEMA200}
              showBB={showBB}
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
