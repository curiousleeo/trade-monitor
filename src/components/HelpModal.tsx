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
    title: 'The 3 Coins',
    items: [
      { label: 'BTC — Bitcoin',  desc: 'The original and largest cryptocurrency. Think of it as digital gold.' },
      { label: 'ETH — Ethereum', desc: 'The second biggest. Powers thousands of apps and smart contracts.' },
      { label: 'SOL — Solana',   desc: 'A fast, low-fee blockchain popular for games, NFTs, and trading.' },
      { label: 'Price cards (top bar)', desc: 'Show the current price of each coin and how much it has moved in the last 24 hours. Green = up. Red = down.' },
    ],
  },
  {
    icon: '📊',
    title: 'The Price Chart',
    items: [
      { label: 'What am I looking at?', desc: 'Each bar on the chart is called a candle. It shows the price action for one time period (e.g. 15 minutes, 1 hour).' },
      { label: 'Green candle',  desc: 'The price went UP during that time period. The coin closed higher than it opened.' },
      { label: 'Red candle',    desc: 'The price went DOWN. The coin closed lower than it opened.' },
      { label: 'The thin line (wick)', desc: 'The thin line sticking out of a candle shows the highest and lowest price reached during that period.' },
      { label: 'Timeframes (1m 5m 15m 1h 4h 1d)', desc: 'How long each candle represents. "15m" means each candle = 15 minutes of price action. "1d" = one full day.' },
    ],
  },
  {
    icon: '📈',
    title: 'Chart Indicators (the lines)',
    items: [
      { label: 'EMA 20 / 50 / 200',   desc: 'Exponential Moving Average. A smooth line tracking the average price over the last 20, 50, or 200 candles. When the price is above the line → the trend is up. Below → the trend is down.' },
      { label: 'BB — Bollinger Bands', desc: 'Three lines that show how much the price is moving. When the bands squeeze together, a big move is likely coming. When price touches the outer band, it may bounce back.' },
      { label: 'RSI — Relative Strength Index', desc: 'A number from 0 to 100 shown in the panel below the chart. Above 70 = everyone is buying (possibly overbought, price may drop soon). Below 30 = everyone is selling (possibly oversold, price may bounce up).' },
      { label: 'NEWS markers (●)', desc: 'Dots on the chart showing when a news article was published. Big news often causes big price moves.' },
    ],
  },
  {
    icon: '🧠',
    title: 'The Numbers Strip (below header)',
    items: [
      { label: 'Fear & Greed',  desc: 'A score from 0 to 100 measuring overall market mood. 0 = extreme fear (people are panic selling). 100 = extreme greed (everyone is rushing to buy). Extreme fear can be a buying opportunity; extreme greed can signal a top.' },
      { label: 'Funding Rate',  desc: 'A small fee paid between traders every 8 hours. Positive = more people are betting the price will go up. Negative = more people are betting it will go down. Very high positive rates often mean the market is too crowded on one side.' },
      { label: '24h High / Low', desc: 'The highest and lowest price the coin reached in the last 24 hours.' },
    ],
  },
  {
    icon: '📰',
    title: 'News Feed',
    items: [
      { label: 'What is it?', desc: 'Real headlines from crypto and financial news sources, updated every 90 seconds. News about regulations, hacks, or economic data can cause prices to move fast.' },
      { label: 'Clicking a headline', desc: 'Jumps the chart to the time that article was published, so you can see how the price reacted to the news.' },
      { label: 'MACRO / GEO tags', desc: 'Articles tagged MACRO are about the economy (inflation, interest rates). GEO = geopolitical events (wars, sanctions). Both can affect crypto prices significantly.' },
    ],
  },
  {
    icon: '⚡',
    title: 'Apex AI',
    items: [
      { label: 'What is Apex?', desc: 'An AI that reads the chart, news, and market data every few minutes and decides whether it thinks the price will go UP, DOWN, or stay flat.' },
      { label: 'LONG',    desc: 'Apex thinks the price will go UP. It may "buy" the coin in the simulation.' },
      { label: 'SHORT',   desc: 'Apex thinks the price will go DOWN. It may "sell" the coin in the simulation.' },
      { label: 'NEUTRAL', desc: 'Apex is not confident either way and stays out of the market.' },
      { label: 'Confidence %', desc: 'How sure Apex is about its prediction. 90% = very sure. 51% = barely leaning one way.' },
      { label: 'Entry / Target / Stop', desc: 'Entry = the price Apex would buy/sell at. Target = where it expects the price to go (profit). Stop = where it would cut the loss if wrong.' },
      { label: 'Portfolio & trades', desc: 'Apex starts with a simulated $1,000. Every trade is fake — no real money. You can watch its balance grow or shrink and see if its predictions are accurate.' },
      { label: 'Timeframe Matrix', desc: 'A quick summary of whether Apex sees an uptrend or downtrend on each timeframe (from 1-minute to 1-day view).' },
    ],
  },
  {
    icon: '🔔',
    title: 'Price Alerts',
    items: [
      { label: 'What is it?', desc: 'Set a price level and the app will flash when the coin reaches it. Useful so you don\'t have to stare at the screen all day.' },
      { label: 'How to set one', desc: 'Type a price number into the alert box in the News tab and press Add. The alert will fire once and then clear itself.' },
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
