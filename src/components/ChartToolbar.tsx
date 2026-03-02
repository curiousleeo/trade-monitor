import { Timeframe } from '../types';

interface Props {
  timeframe:   Timeframe;
  onTimeframe: (tf: Timeframe) => void;
  showEMA:     boolean;  onEMA:  (v: boolean) => void;
  showBOLL:    boolean;  onBOLL: (v: boolean) => void;
  showRSI:     boolean;  onRSI:  (v: boolean) => void;
  showMACD:    boolean;  onMACD: (v: boolean) => void;
  sidebarOpen: boolean;  onToggleSidebar: () => void;
}

const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

export function ChartToolbar({
  timeframe, onTimeframe,
  showEMA, onEMA,
  showBOLL, onBOLL,
  showRSI, onRSI,
  showMACD, onMACD,
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
        <button className={`tb-btn ${showEMA ? 'tb-ema20' : ''}`} onClick={() => onEMA(!showEMA)}>6/12/20</button>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-group">
        <span className="toolbar-label">IND</span>
        <button className={`tb-btn ${showBOLL ? 'tb-bb'  : ''}`} onClick={() => onBOLL(!showBOLL)}  title="Bollinger Bands">BB</button>
        <button className={`tb-btn ${showRSI  ? 'tb-rsi' : ''}`} onClick={() => onRSI(!showRSI)}   >RSI</button>
        <button className={`tb-btn ${showMACD ? 'tb-rsi' : ''}`} onClick={() => onMACD(!showMACD)} >MACD</button>
      </div>

    </div>
  );
}
