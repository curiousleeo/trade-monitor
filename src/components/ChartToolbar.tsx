import { Timeframe } from '../types';

interface Props {
  timeframe: Timeframe;
  onTimeframe: (tf: Timeframe) => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d'];

export function ChartToolbar({ timeframe, onTimeframe, sidebarOpen, onToggleSidebar }: Props) {
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
    </div>
  );
}
