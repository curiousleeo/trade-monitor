import { Timeframe } from '../types';

interface Props {
  timeframe: Timeframe;
  onTimeframe: (tf: Timeframe) => void;
  showEMA20: boolean;   onEMA20:  (v: boolean) => void;
  showEMA50: boolean;   onEMA50:  (v: boolean) => void;
  showEMA200: boolean;  onEMA200: (v: boolean) => void;
  showBB: boolean;      onBB:     (v: boolean) => void;
  showRSI: boolean;     onRSI:    (v: boolean) => void;
  showNewsMarkers: boolean; onNewsMarkers: (v: boolean) => void;
  sidebarOpen: boolean; onToggleSidebar: () => void;
}

const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

export function ChartToolbar({
  timeframe, onTimeframe,
  showEMA20, onEMA20,
  showEMA50, onEMA50,
  showEMA200, onEMA200,
  showBB, onBB,
  showRSI, onRSI,
  showNewsMarkers, onNewsMarkers,
  sidebarOpen, onToggleSidebar,
}: Props) {
  return (
    <div className="chart-toolbar">

      <button
        className="sidebar-toggle"
        onClick={onToggleSidebar}
        title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
      >
        {sidebarOpen ? '◀' : '▶'}
      </button>

      <div className="toolbar-group">
        <span className="toolbar-label">TF</span>
        {TIMEFRAMES.map(tf => (
          <button
            key={tf}
            className={`tb-btn ${timeframe === tf ? 'tb-btn--active' : ''}`}
            onClick={() => onTimeframe(tf)}
          >
            {tf}
          </button>
        ))}
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-group">
        <span className="toolbar-label">EMA</span>
        <button className={`tb-btn ${showEMA20  ? 'tb-ema20'  : ''}`} onClick={() => onEMA20(!showEMA20)}>20</button>
        <button className={`tb-btn ${showEMA50  ? 'tb-ema50'  : ''}`} onClick={() => onEMA50(!showEMA50)}>50</button>
        <button className={`tb-btn ${showEMA200 ? 'tb-ema200' : ''}`} onClick={() => onEMA200(!showEMA200)}>200</button>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-group">
        <span className="toolbar-label">IND</span>
        <button className={`tb-btn ${showBB  ? 'tb-bb'  : ''}`} onClick={() => onBB(!showBB)} title="Bollinger Bands (20, 2)">BB</button>
        <button className={`tb-btn ${showRSI ? 'tb-rsi' : ''}`} onClick={() => onRSI(!showRSI)}>RSI</button>
        <button
          className={`tb-btn ${showNewsMarkers ? 'tb-news' : ''}`}
          onClick={() => onNewsMarkers(!showNewsMarkers)}
          title="Toggle news event markers"
        >
          ● NEWS
        </button>
      </div>

    </div>
  );
}
