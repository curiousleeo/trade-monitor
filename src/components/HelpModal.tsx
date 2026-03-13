interface Props {
  onClose: () => void;
}

interface Section {
  icon: string;
  title: string;
  items: { label: string; desc: string }[];
}

const SECTIONS: Section[] = [
  {
    icon: '🪙',
    title: 'Coin Ticker (top bar)',
    items: [
      { label: '21 coins supported', desc: 'BTC, ETH, BNB, XRP, LTC, TRX, SOL, AVAX, DOT, LINK, ATOM, NEAR, UNI, ADA, DOGE, SUI, APT, ARB, OP, INJ, PAXG — all tracked live from Binance.' },
      { label: 'Price cards', desc: 'Each card shows the coin\'s current price and 24-hour change. Green = price is up vs yesterday. Red = down. Click any card to switch to that coin.' },
      { label: 'Scrolling the ticker', desc: 'Hover over the ticker to reveal the ‹ › arrow buttons, or use your mouse wheel to scroll through all coins.' },
    ],
  },
  {
    icon: '🧠',
    title: 'Stats Strip (below header)',
    items: [
      { label: 'Fear & Greed', desc: 'A score from 0–100 measuring overall market mood. 0 = extreme fear (panic selling). 100 = extreme greed (everyone rushing to buy). Extremes often signal reversals.' },
      { label: 'Funding Rate', desc: 'A fee paid between futures traders every 8 hours. Positive = crowd is betting price goes up. Negative = crowd bets it goes down. Very high positive rates can mean the market is overleveraged long.' },
      { label: '24h High / Low', desc: 'The highest and lowest price reached by the selected coin in the past 24 hours.' },
    ],
  },
  {
    icon: '📊',
    title: 'The Price Chart',
    items: [
      { label: 'Candles', desc: 'Each bar shows price action for one time period. Green candle = price closed higher than it opened. Red = closed lower. The thin wick shows the highest and lowest point reached.' },
      { label: 'Timeframes (1m 5m 15m 1h 4h 1d)', desc: 'How long each candle represents. "15m" = each candle is 15 minutes of price action. "1d" = one full day.' },
      { label: 'Prev Day H/L lines', desc: 'Dashed green and red horizontal lines showing yesterday\'s high and low price — key levels traders watch for support and resistance.' },
      { label: 'Trade markers', desc: 'Apex\'s entry and exit points appear directly on the chart. L↑ = long entry, S↓ = short entry, TP = take profit hit, SL = stop loss hit.' },
      { label: 'TP / SL lines', desc: 'When Apex has an open trade, dashed green (take profit) and red (stop loss) lines are drawn on the chart at the target levels.' },
    ],
  },
  {
    icon: '📈',
    title: 'Chart Indicators',
    items: [
      { label: 'EMA 6 / 12 / 20', desc: 'Exponential Moving Averages. Smooth lines tracking the average price over the last 6, 12, or 20 candles. Price above the lines = uptrend. Below = downtrend. When shorter EMA crosses above longer EMA, it\'s often a buy signal.' },
      { label: 'BB — Bollinger Bands', desc: 'Three bands around the price. When the bands squeeze together, a big move is brewing. When price touches the outer band, it may bounce back toward the middle.' },
      { label: 'RSI — Relative Strength Index', desc: 'Shown in a sub-panel below the chart. Ranges 0–100. Above 70 = overbought (price may pull back). Below 30 = oversold (price may bounce). Apex uses RSI as one of its 5 scoring signals.' },
      { label: 'MACD', desc: 'Shown in a sub-panel. Measures momentum by comparing two moving averages. When the MACD line crosses above the signal line, momentum is turning bullish. Below = bearish.' },
    ],
  },
  {
    icon: '⚡',
    title: 'Apex AI — Signals Tab',
    items: [
      { label: 'What is Apex?', desc: 'A quantitative momentum trader that reads chart data, funding rates, Fear & Greed, and market structure every few minutes to decide whether to go LONG, SHORT, or stay NEUTRAL.' },
      { label: 'Coin pills (top row)', desc: 'Shows Apex\'s current bias for all 21 coins at a glance. Green pill = bullish, red = bearish. Click a coin in the main ticker to see its full prediction card.' },
      { label: 'Prediction card', desc: 'Shows direction (LONG/SHORT/NEUTRAL), confidence %, entry zone, target price, stop price, and risk-to-reward ratio for the active coin.' },
      { label: '5 signal bars', desc: 'The prediction is built from 5 weighted components: EMA Trend (35%), RSI Momentum (25%), Sentiment/Fear&Greed (20%), Market Structure (10%), Funding Bias (10%). Each bar shows the individual score.' },
      { label: 'Confidence %', desc: 'How certain Apex is. 90% = very strong signal. Below 65% = Apex stays out (no trade taken).' },
      { label: 'FORCE ENTRY button', desc: 'Lets you manually trigger Apex to enter a trade on the current signal, even mid-candle. Only appears when signal is LONG or SHORT and no open trade exists for that coin.' },
      { label: 'Timeframe Matrix', desc: 'Shows Apex\'s directional bias across all 6 timeframes (1m → 1d) using EMA alignment. Green = uptrend bias, red = downtrend. Helps spot confluence when multiple timeframes agree.' },
    ],
  },
  {
    icon: '📋',
    title: 'Apex AI — Trades Tab',
    items: [
      { label: 'Portfolio', desc: 'Apex starts with a simulated $1,000 and tries to grow it to $100,000. All trades are fake — no real money ever involved. Balance updates in real time as open trades move.' },
      { label: 'Risk management', desc: 'Apex risks exactly 2% of its balance per trade with a minimum 2:1 reward-to-risk ratio. Max 2 open positions at once. Position size is calculated using ATR so stops are always volatility-adjusted.' },
      { label: 'Open Positions', desc: 'Active trades with live unrealized P&L updating from the current market price.' },
      { label: 'Trade History', desc: 'All closed trades. ✓ TP HIT = Apex hit its target (win). ✗ SL HIT = stop loss triggered (loss). Shows entry price, exit price, and final P&L.' },
      { label: 'Win Rate', desc: 'The percentage of closed trades that hit take profit. A good systematic trader aims for 40–60% win rate with strong R:R.' },
      { label: 'Reset Portfolio', desc: 'Wipes all trade history and resets Apex\'s balance back to $1,000 to start fresh.' },
    ],
  },
  {
    icon: '🔬',
    title: 'Backtest Tab',
    items: [
      { label: 'What is backtesting?', desc: 'Running Apex\'s trading logic on historical candle data to see how it would have performed in the past. Useful for validating whether the strategy has an edge.' },
      { label: 'How to use it', desc: 'Select the coin and timeframe in the main chart, then switch to the Backtest tab. Results show simulated trades on the loaded candle history.' },
    ],
  },
];

export function HelpModal({ onClose }: Props) {
  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-modal" onClick={e => e.stopPropagation()}>

        <div className="help-header">
          <div className="help-title">
            <span className="help-title-icon">◈</span>
            <span>How to Use Apex</span>
          </div>
          <button className="help-close" onClick={onClose}>✕</button>
        </div>

        <p className="help-intro">
          New here? No trading experience needed. This guide explains everything on the screen in plain English.
        </p>

        <div className="help-sections">
          {SECTIONS.map(section => (
            <div key={section.title} className="help-section">
              <div className="help-section-title">
                <span>{section.icon}</span>
                <span>{section.title}</span>
              </div>
              <div className="help-items">
                {section.items.map(item => (
                  <div key={item.label} className="help-item">
                    <div className="help-item-label">{item.label}</div>
                    <div className="help-item-desc">{item.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="help-footer">
          Apex is for education only — all trades are simulated. No real money is ever used.
        </div>

      </div>
    </div>
  );
}
