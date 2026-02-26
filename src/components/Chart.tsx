import { useEffect, useRef } from 'react';
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  SeriesMarker,
  UTCTimestamp,
  CandlestickData,
} from 'lightweight-charts';
import { Candle, NewsItem, Timeframe } from '../types';

interface ChartProps {
  candles: Candle[];
  liveCandle: Candle | null;
  news: NewsItem[];
  timeframe: Timeframe;
  coin: string;
}

const TF_SECONDS: Record<Timeframe, number> = {
  '1m': 60,
  '5m': 300,
  '15m': 900,
  '1h': 3600,
  '4h': 14400,
  '1d': 86400,
};

export function Chart({ candles, liveCandle, news, timeframe, coin }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  // Mount chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#6b7280',
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#111827' },
        horzLines: { color: '#111827' },
      },
      crosshair: {
        vertLine: { color: '#374151', labelBackgroundColor: '#1f2937' },
        horzLine: { color: '#374151', labelBackgroundColor: '#1f2937' },
      },
      rightPriceScale: {
        borderColor: '#1f2937',
      },
      timeScale: {
        borderColor: '#1f2937',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth || window.innerWidth,
      height: containerRef.current.clientHeight || window.innerHeight - 48,
    });

    const series = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const observer = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.resize(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        );
      }
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Set historical data
  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return;
    const data: CandlestickData[] = candles.map(c => ({
      time: c.time as UTCTimestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    seriesRef.current.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  // Update live candle
  useEffect(() => {
    if (!seriesRef.current || !liveCandle) return;
    seriesRef.current.update({
      time: liveCandle.time as UTCTimestamp,
      open: liveCandle.open,
      high: liveCandle.high,
      low: liveCandle.low,
      close: liveCandle.close,
    });
  }, [liveCandle]);

  // Place news markers on chart
  useEffect(() => {
    if (!seriesRef.current || candles.length === 0 || news.length === 0) return;

    const tfSec = TF_SECONDS[timeframe];
    const firstTime = candles[0].time;
    const lastTime = candles[candles.length - 1].time;

    // Group news items by candle timestamp
    const grouped = new Map<number, NewsItem[]>();
    news.forEach(item => {
      const candleTime = Math.floor(item.publishedAt / tfSec) * tfSec;
      if (candleTime < firstTime || candleTime > lastTime + tfSec) return;
      if (!grouped.has(candleTime)) grouped.set(candleTime, []);
      grouped.get(candleTime)!.push(item);
    });

    const markers: SeriesMarker<UTCTimestamp>[] = Array.from(grouped.entries())
      .map(([time, items]) => ({
        time: time as UTCTimestamp,
        position: 'aboveBar' as const,
        color: '#f59e0b',
        shape: 'circle' as const,
        text: items.length > 1 ? `${items.length}` : '',
        size: 1,
      }))
      .sort((a, b) => a.time - b.time);

    seriesRef.current.setMarkers(markers);
  }, [news, candles, timeframe]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {candles.length === 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#374151',
            fontSize: '13px',
            letterSpacing: '0.1em',
            fontFamily: 'monospace',
          }}
        >
          LOADING {coin} DATA...
        </div>
      )}
    </div>
  );
}
