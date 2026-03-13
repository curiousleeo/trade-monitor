import { useState, useCallback, useEffect, useRef } from 'react';
import { KLineChart }       from './components/KLineChart';
import { CoinCard }        from './components/CoinCard';
import { StatsStrip }      from './components/StatsStrip';
import { ChartToolbar }    from './components/ChartToolbar';
import { AIPanel }         from './components/AIPanel';
import { BacktestPanel }   from './components/BacktestPanel';
import { HelpModal }       from './components/HelpModal';
import { useKlines }       from './hooks/useKlines';
import { useFearGreed }    from './hooks/useFearGreed';
import { useFundingRate }  from './hooks/useFundingRate';
import { usePrevDayOHLC }  from './hooks/usePrevDayOHLC';
import { use24hTicker }    from './hooks/use24hTicker';
import { useAllCandles }   from './hooks/useAllCandles';
import { useAITrader }     from './hooks/useAITrader';
import { Coin, FundingRate, Timeframe } from './types';

const COINS: Coin[] = [
  'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
  'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
  'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ',
  'PAXG',
];

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
  const [showEMA,  setShowEMA]                = useState(true);
  const [showBOLL, setShowBOLL]               = useState(false);
  const [showRSI,  setShowRSI]                = useState(false);
  const [showMACD, setShowMACD]               = useState(false);
  const [mobilePage, setMobilePage]           = useState<'chart' | 'ai'>('chart');
  const [scrollToTime, setScrollToTime]       = useState<number | null>(null);
  const [sidebarTab, setSidebarTab]           = useState<'ai' | 'backtest'>('ai');
  const [sidebarOpen, setSidebarOpen]         = useState(true);
  const [showHelp, setShowHelp]               = useState(false);
  const now = useClock();
  const { theme, toggle: toggleTheme } = useTheme();

  // ── Data hooks ─────────────────────────────────────────────────────────
  const { candles, liveCandle } = useKlines(coin, timeframe);
  const tickers     = use24hTicker();
  const fearGreed   = useFearGreed();
  const fundingRate = useFundingRate(coin);

  const fundingBTC = useFundingRate('BTC');
  const fundingETH = useFundingRate('ETH');
  const fundingSOL = useFundingRate('SOL');
  const fundingRates: Record<Coin, FundingRate | null> = {
    BTC: fundingBTC, ETH: fundingETH, SOL: fundingSOL,
    BNB: null, XRP: null, LTC: null, TRX: null,
    AVAX: null, DOT: null, LINK: null, ATOM: null, NEAR: null, UNI: null, ADA: null,
    DOGE: null, SUI: null, APT: null, ARB: null, OP: null, INJ: null,
    PAXG: null,
  };

  const prevDay = usePrevDayOHLC(coin);
  const allCandles = useAllCandles(timeframe);

  // ── AI Trader ──────────────────────────────────────────────────────────
  const { portfolio, closedTrades, predictions, tfMatrix, tradeMarkers, resetPortfolio, forceEntry } = useAITrader({
    allCandles,
    activeCandles:   candles,
    activeCoin:      coin,
    activeTimeframe: timeframe,
    tickers,
    fearGreed,
    fundingRates,
    prevDay,
  });


  const tickerRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft,  setCanScrollLeft]  = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const updateTickerScroll = useCallback(() => {
    const el = tickerRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 4);
  }, []);

  useEffect(() => {
    const el = tickerRef.current;
    if (!el) return;
    updateTickerScroll();
    el.addEventListener('scroll', updateTickerScroll, { passive: true });
    return () => el.removeEventListener('scroll', updateTickerScroll);
  }, [updateTickerScroll]);

  // Re-check once ticker prices arrive (coins widen and may now overflow)
  const hasTickerData = Object.values(tickers).some(t => t !== null);
  useEffect(() => {
    updateTickerScroll();
  }, [hasTickerData, updateTickerScroll]);

  const scrollTicker = useCallback((dir: 'left' | 'right') => {
    tickerRef.current?.scrollBy({ left: dir === 'left' ? -220 : 220, behavior: 'smooth' });
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

        <div className={`ticker-wrap${canScrollLeft ? ' ticker-wrap--left' : ''}${canScrollRight ? ' ticker-wrap--right' : ''}`}>
          <button className="ticker-arrow ticker-arrow--left" onClick={() => scrollTicker('left')}>‹</button>
          <div className="coin-cards" ref={tickerRef}>
            {COINS.map(c => (
              <CoinCard
                key={c}
                coin={c}
                ticker={tickers[c]}
                active={coin === c}
                onClick={setCoin}
              />
            ))}
          </div>
          <button className="ticker-arrow ticker-arrow--right" onClick={() => scrollTicker('right')}>›</button>
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

      {/* ── Stats strip ── */}
      <StatsStrip
        fearGreed={fearGreed}
        fundingRate={fundingRate}
        ticker={tickers[coin]}
      />

      {/* ── Body ── */}
      <main className={`main mobile-page--${mobilePage}`}>
        <aside className={`sidebar${sidebarOpen ? '' : ' sidebar--collapsed'}`}>

          <div className="sidebar-tabs">
            <button
              className={`sidebar-tab ${sidebarTab === 'ai' ? 'active' : ''}`}
              onClick={() => setSidebarTab('ai')}
            >
              Apex AI
            </button>
            <button
              className={`sidebar-tab ${sidebarTab === 'backtest' ? 'active' : ''}`}
              onClick={() => setSidebarTab('backtest')}
            >
              Backtest
            </button>
          </div>

          {sidebarTab === 'ai' ? (
            <AIPanel
              portfolio={portfolio}
              closedTrades={closedTrades}
              predictions={predictions}
              tfMatrix={tfMatrix}
              activeCoin={coin}
              tickers={tickers}
              onReset={resetPortfolio}
              onForceEntry={forceEntry}
            />
          ) : (
            <BacktestPanel
              candles={candles}
              coin={coin}
              timeframe={timeframe}
            />
          )}
        </aside>

        <section className="chart-area">
          <ChartToolbar
            timeframe={timeframe}  onTimeframe={setTimeframe}
            showEMA={showEMA}      onEMA={setShowEMA}
            showBOLL={showBOLL}    onBOLL={setShowBOLL}
            showRSI={showRSI}      onRSI={setShowRSI}
            showMACD={showMACD}    onMACD={setShowMACD}
            sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(o => !o)}
          />
          <div className="chart-canvas">
            <KLineChart
              candles={candles}
              liveCandle={liveCandle}
              coin={coin}
              timeframe={timeframe}
              theme={theme}
              prevDay={prevDay}
              showEMA={showEMA}
              showBOLL={showBOLL}
              showRSI={showRSI}
              showMACD={showMACD}
              tradeMarkers={tradeMarkers}
              openTrades={portfolio.openTrades}
              scrollToTime={scrollToTime}
            />
          </div>
        </section>
      </main>

      {/* ── Mobile bottom nav ── */}
      <nav className="mobile-nav">
        <div className="mobile-nav-coins">
          {COINS.map(c => {
            const t = tickers[c];
            const isUp = (t?.change24h ?? 0) >= 0;
            const price = t
              ? (c === 'BTC'
                  ? `$${Math.round(t.price).toLocaleString()}`
                  : `$${t.price.toFixed(2)}`)
              : '—';
            return (
              <button
                key={c}
                className={`mobile-coin-btn ${coin === c ? 'mobile-coin-btn--active' : ''}`}
                onClick={() => { setCoin(c); setMobilePage('chart'); }}
              >
                {c}
                <span className={`mobile-coin-price ${coin === c ? '' : (isUp ? 'up' : 'down')}`}>
                  {price}
                </span>
              </button>
            );
          })}
        </div>
        <div className="mobile-nav-tabs">
          <button
            className={`mobile-tab-btn ${mobilePage === 'chart' ? 'mobile-tab-btn--active' : ''}`}
            onClick={() => setMobilePage('chart')}
          >
            <span className="mobile-tab-icon">📈</span>
            <span>Chart</span>
          </button>
          <button
            className={`mobile-tab-btn ${mobilePage === 'ai' ? 'mobile-tab-btn--active' : ''}`}
            onClick={() => { setMobilePage('ai'); setSidebarTab('ai'); }}
          >
            <span className="mobile-tab-icon">⚡</span>
            <span>Apex AI</span>
          </button>
        </div>
      </nav>

    </div>
  );
}
