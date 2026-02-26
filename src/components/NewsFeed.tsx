import { useEffect, useRef } from 'react';
import { NewsItem } from '../types';

interface Props {
  news: NewsItem[];
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000) - ts;
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

const COIN_COLORS: Record<string, string> = {
  BTC: '#f7931a',
  ETH: '#627eea',
  SOL: '#9945ff',
};

function getCoins(categories: string): string[] {
  return ['BTC', 'ETH', 'SOL'].filter(c => categories.includes(c));
}

export function NewsFeed({ news }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);

  // Flash new items at the top when news updates
  useEffect(() => {
    if (news.length > prevLengthRef.current && listRef.current) {
      listRef.current.scrollTop = 0;
    }
    prevLengthRef.current = news.length;
  }, [news.length]);

  return (
    <div className="news-panel">
      <div className="panel-header">
        <span className="live-dot" />
        <span>LIVE NEWS</span>
        <span className="news-count">{news.length}</span>
      </div>

      <div className="news-list" ref={listRef}>
        {news.length === 0 ? (
          <div className="news-empty">FETCHING FEED...</div>
        ) : (
          news.map(item => {
            const coins = getCoins(item.categories);
            return (
              <a
                key={item.id}
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="news-item"
              >
                <div className="news-top-row">
                  <span className="news-source">{item.source}</span>
                  <span className="news-time">{timeAgo(item.publishedAt)}</span>
                </div>
                <div className="news-title">{item.title}</div>
                {coins.length > 0 && (
                  <div className="news-tags">
                    {coins.map(c => (
                      <span
                        key={c}
                        className="news-tag"
                        style={{ color: COIN_COLORS[c], borderColor: COIN_COLORS[c] }}
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                )}
              </a>
            );
          })
        )}
      </div>
    </div>
  );
}
