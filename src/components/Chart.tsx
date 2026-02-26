import { useEffect, useRef } from 'react';
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  IPriceLine,
  LineStyle,
  SeriesMarker,
  UTCTimestamp,
  CandlestickData,
  HistogramData,
} from 'lightweight-charts';
import { Candle, NewsItem, Timeframe } from '../types';
import { calcEMA, calcRSI } from '../utils/indicators';
import { PrevDay } from '../hooks/usePrevDayOHLC';

interface ChartProps {
  candles: Candle[];
  liveCandle: Candle | null;
  news: NewsItem[];
  timeframe: Timeframe;
  coin: string;
  showEMA20: boolean;
  showEMA50: boolean;
  showEMA200: boolean;
  showRSI: boolean;
  prevDay: PrevDay | null;
  scrollToTime: number | null;
  onCandleClick: (time: number) => void;
}

const TF_SECONDS: Record<Timeframe, number> = {
  '1m': 60, '5m': 300, '15m': 900,
  '1h': 3600, '4h': 14400, '1d': 86400,
};

const BASE_CHART_OPTS = {
  layout: {
    background: { type: ColorType.Solid as const, color: '#0a0a0a' },
    textColor: '#6b7280',
    fontFamily: "'SF Mono','Fira Code',monospace",
    fontSize: 11,
  },
  grid: { vertLines: { color: '#111827' }, horzLines: { color: '#111827' } },
  crosshair: {
    vertLine: { color: '#374151', labelBackgroundColor: '#1f2937' },
    horzLine: { color: '#374151', labelBackgroundColor: '#1f2937' },
  },
  rightPriceScale: { borderColor: '#1f2937' },
  timeScale: { borderColor: '#1f2937', timeVisible: true, secondsVisible: false },
};

export function Chart({
  candles, liveCandle, news, timeframe, coin,
  showEMA20, showEMA50, showEMA200, showRSI,
  prevDay, scrollToTime, onCandleClick,
}: ChartProps) {
  const mainRef = useRef<HTMLDivElement>(null);
  const rsiRef  = useRef<HTMLDivElement>(null);

  const chartRef      = useRef<IChartApi | null>(null);
  const rsiChartRef   = useRef<IChartApi | null>(null);
  const isSyncing     = useRef(false);

  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ema20Ref  = useRef<ISeriesApi<'Line'> | null>(null);
  const ema50Ref  = useRef<ISeriesApi<'Line'> | null>(null);
  const ema200Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const rsiSeriesRef  = useRef<ISeriesApi<'Line'> | null>(null);
  const pdHighRef = useRef<IPriceLine | null>(null);
  const pdLowRef  = useRef<IPriceLine | null>(null);

  // ── Mount main chart ──────────────────────────────────────────────────
  useEffect(() => {
    if (!mainRef.current) return;

    const chart = createChart(mainRef.current, {
      ...BASE_CHART_OPTS,
      width:  mainRef.current.clientWidth  || window.innerWidth,
      height: mainRef.current.clientHeight || window.innerHeight - 84,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });

    const volumeSeries = chart.addHistogramSeries({
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    const makeEMA = (color: string) => chart.addLineSeries({
      color, lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });

    chartRef.current        = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    ema20Ref.current        = makeEMA('#3b82f6');
    ema50Ref.current        = makeEMA('#f59e0b');
    ema200Ref.current       = makeEMA('#ec4899');

    chart.subscribeClick(param => {
      if (param.time) onCandleClick(param.time as number);
    });

    const obs = new ResizeObserver(() => {
      if (mainRef.current && chartRef.current) {
        chartRef.current.resize(mainRef.current.clientWidth, mainRef.current.clientHeight);
      }
    });
    obs.observe(mainRef.current);

    return () => {
      obs.disconnect();
      chart.remove();
      chartRef.current = candleSeriesRef.current = volumeSeriesRef.current = null;
      ema20Ref.current = ema50Ref.current = ema200Ref.current = null;
    };
  }, []);

  // ── Mount / unmount RSI chart ─────────────────────────────────────────
  useEffect(() => {
    if (!showRSI || !rsiRef.current) return;

    const rsiChart = createChart(rsiRef.current, {
      ...BASE_CHART_OPTS,
      width:  rsiRef.current.clientWidth  || window.innerWidth,
      height: rsiRef.current.clientHeight || 130,
      timeScale: { ...BASE_CHART_OPTS.timeScale, visible: false },
      rightPriceScale: {
        borderColor: '#1f2937',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
    });

    const rsiSeries = rsiChart.addLineSeries({
      color: '#7c3aed', lineWidth: 1,
      priceLineVisible: false, lastValueVisible: true,
      crosshairMarkerVisible: false,
    });

    rsiSeries.createPriceLine({ price: 70, color: '#374151', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '' });
    rsiSeries.createPriceLine({ price: 30, color: '#374151', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '' });

    rsiChartRef.current  = rsiChart;
    rsiSeriesRef.current = rsiSeries;

    // Sync main → RSI
    if (chartRef.current) {
      chartRef.current.timeScale().subscribeVisibleTimeRangeChange(range => {
        if (isSyncing.current || !range) return;
        isSyncing.current = true;
        rsiChart.timeScale().setVisibleRange(range);
        isSyncing.current = false;
      });
    }
    // Sync RSI → main
    rsiChart.timeScale().subscribeVisibleTimeRangeChange(range => {
      if (isSyncing.current || !range || !chartRef.current) return;
      isSyncing.current = true;
      chartRef.current.timeScale().setVisibleRange(range);
      isSyncing.current = false;
    });

    const obs = new ResizeObserver(() => {
      if (rsiRef.current && rsiChartRef.current) {
        rsiChartRef.current.resize(rsiRef.current.clientWidth, rsiRef.current.clientHeight);
      }
    });
    obs.observe(rsiRef.current);

    return () => {
      obs.disconnect();
      rsiChart.remove();
      rsiChartRef.current = rsiSeriesRef.current = null;
    };
  }, [showRSI]);

  // ── Set chart data ────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;

    candleSeriesRef.current.setData(
      candles.map<CandlestickData>(c => ({
        time: c.time as UTCTimestamp,
        open: c.open, high: c.high, low: c.low, close: c.close,
      }))
    );
    chartRef.current?.timeScale().fitContent();

    volumeSeriesRef.current?.setData(
      candles.map<HistogramData>(c => ({
        time: c.time as UTCTimestamp,
        value: c.volume,
        color: c.close >= c.open ? '#22c55e28' : '#ef444428',
      }))
    );

    const closes = candles.map(c => c.close);

    const toLineData = (vals: (number | null)[]): { time: UTCTimestamp; value: number }[] => {
      const pts: { time: UTCTimestamp; value: number }[] = [];
      vals.forEach((v, i) => { if (v !== null) pts.push({ time: candles[i].time as UTCTimestamp, value: v }); });
      return pts;
    };

    const applyEMA = (ref: typeof ema20Ref, period: number) => {
      if (!ref.current) return;
      ref.current.setData(toLineData(calcEMA(closes, period)));
    };
    applyEMA(ema20Ref, 20);
    applyEMA(ema50Ref, 50);
    applyEMA(ema200Ref, 200);

    if (rsiSeriesRef.current) {
      rsiSeriesRef.current.setData(toLineData(calcRSI(closes)));
    }
  }, [candles]);

  // ── Live candle ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!liveCandle) return;
    const t = liveCandle.time as UTCTimestamp;
    candleSeriesRef.current?.update({ time: t, open: liveCandle.open, high: liveCandle.high, low: liveCandle.low, close: liveCandle.close });
    volumeSeriesRef.current?.update({ time: t, value: liveCandle.volume, color: liveCandle.close >= liveCandle.open ? '#22c55e28' : '#ef444428' });
  }, [liveCandle]);

  // ── EMA visibility toggles ────────────────────────────────────────────
  useEffect(() => { ema20Ref.current?.applyOptions({ visible: showEMA20 }); }, [showEMA20]);
  useEffect(() => { ema50Ref.current?.applyOptions({ visible: showEMA50 }); }, [showEMA50]);
  useEffect(() => { ema200Ref.current?.applyOptions({ visible: showEMA200 }); }, [showEMA200]);

  // ── Prev day H/L ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current || !prevDay) return;
    try { if (pdHighRef.current) candleSeriesRef.current.removePriceLine(pdHighRef.current); } catch {}
    try { if (pdLowRef.current)  candleSeriesRef.current.removePriceLine(pdLowRef.current);  } catch {}
    pdHighRef.current = candleSeriesRef.current.createPriceLine({ price: prevDay.high, color: '#4ade80', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PD H' });
    pdLowRef.current  = candleSeriesRef.current.createPriceLine({ price: prevDay.low,  color: '#f87171', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PD L' });
  }, [prevDay]);

  // ── News markers ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;
    const tfSec = TF_SECONDS[timeframe];
    const firstT = candles[0].time, lastT = candles[candles.length - 1].time;
    const grouped = new Map<number, number>();
    news.forEach(item => {
      const t = Math.floor(item.publishedAt / tfSec) * tfSec;
      if (t >= firstT && t <= lastT + tfSec) grouped.set(t, (grouped.get(t) ?? 0) + 1);
    });
    const markers: SeriesMarker<UTCTimestamp>[] = Array.from(grouped.entries())
      .map(([time, count]) => ({ time: time as UTCTimestamp, position: 'aboveBar' as const, color: '#f59e0b', shape: 'circle' as const, text: count > 1 ? `${count}` : '', size: 1 }))
      .sort((a, b) => a.time - b.time);
    candleSeriesRef.current.setMarkers(markers);
  }, [news, candles, timeframe]);

  // ── Scroll chart to news time ─────────────────────────────────────────
  useEffect(() => {
    if (!scrollToTime || !chartRef.current) return;
    const tfSec = TF_SECONDS[timeframe];
    try {
      chartRef.current.timeScale().setVisibleRange({
        from: (scrollToTime - tfSec * 50) as UTCTimestamp,
        to:   (scrollToTime + tfSec * 50) as UTCTimestamp,
      });
    } catch {}
  }, [scrollToTime, timeframe]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div ref={mainRef} style={{ flex: showRSI ? '0 0 70%' : '1', position: 'relative' }}>
        {candles.length === 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: '12px', letterSpacing: '0.1em', fontFamily: 'monospace', pointerEvents: 'none' }}>
            LOADING {coin}...
          </div>
        )}
      </div>
      {showRSI && (
        <div ref={rsiRef} style={{ flex: '0 0 30%', borderTop: '1px solid #1a1a1a', position: 'relative' }}>
          <span style={{ position: 'absolute', top: 4, left: 8, fontSize: '9px', color: '#7c3aed', letterSpacing: '0.1em', fontFamily: 'monospace', zIndex: 1, pointerEvents: 'none' }}>RSI(14)</span>
        </div>
      )}
    </div>
  );
}
