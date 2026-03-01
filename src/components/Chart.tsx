import { useEffect, useRef, useState } from 'react';
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
import { Candle, Timeframe, Trade } from '../types';
import { calcEMA, calcRSI, calcBollingerBands } from '../utils/indicators';
import { PrevDay } from '../hooks/usePrevDayOHLC';
import { TradeMarkerData } from '../hooks/useAITrader';

interface ChartProps {
  candles: Candle[];
  liveCandle: Candle | null;
  timeframe: Timeframe;
  coin: string;
  theme: 'light' | 'dark';
  showEMA20: boolean;
  showEMA50: boolean;
  showEMA200: boolean;
  showBB: boolean;
  showRSI: boolean;
  prevDay: PrevDay | null;
  scrollToTime: number | null;
  onCandleClick: (time: number) => void;
  tradeMarkers?: TradeMarkerData[];
  openTrades?: Trade[];
}

interface HoverData { o: number; h: number; l: number; c: number; v: number }

const TF_SECONDS: Record<Timeframe, number> = {
  '1m': 60, '5m': 300, '15m': 900,
  '1h': 3600, '4h': 14400, '1d': 86400,
};

function chartColors(theme: 'light' | 'dark') {
  const dark = theme === 'dark';
  return {
    bg:         dark ? '#0a0a0a' : '#ffffff',
    text:       dark ? '#6b7280' : '#6b7280',
    grid:       dark ? '#111827' : '#f3f4f6',
    crosshair:  dark ? '#374151' : '#d1d5db',
    labelBg:    dark ? '#1f2937' : '#f3f4f6',
    border:     dark ? '#1f2937' : '#e5e7eb',
  };
}

function buildChartOpts(theme: 'light' | 'dark') {
  const c = chartColors(theme);
  return {
    layout: {
      background: { type: ColorType.Solid as const, color: c.bg },
      textColor: c.text,
      fontFamily: "'SF Mono','Fira Code',monospace",
      fontSize: 11,
    },
    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
    crosshair: {
      vertLine: { color: c.crosshair, labelBackgroundColor: c.labelBg },
      horzLine: { color: c.crosshair, labelBackgroundColor: c.labelBg },
    },
    rightPriceScale: { borderColor: c.border },
    timeScale: { borderColor: c.border, timeVisible: true, secondsVisible: false },
  };
}

function fmtPrice(v: number): string {
  if (v >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (v >= 1)    return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return v.toFixed(4);
}

function fmtVol(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(2);
}

export function Chart({
  candles, liveCandle, timeframe, coin, theme,
  showEMA20, showEMA50, showEMA200, showBB, showRSI,
  prevDay, scrollToTime, onCandleClick,
  tradeMarkers = [], openTrades = [],
}: ChartProps) {
  const mainRef = useRef<HTMLDivElement>(null);
  const rsiRef  = useRef<HTMLDivElement>(null);

  const [hover, setHover] = useState<HoverData | null>(null);
  const [countdown, setCountdown] = useState<string | null>(null);
  const [countdownX, setCountdownX] = useState<number | null>(null);

  const chartRef      = useRef<IChartApi | null>(null);
  const rsiChartRef   = useRef<IChartApi | null>(null);
  const isSyncing     = useRef(false);

  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ema20Ref    = useRef<ISeriesApi<'Line'> | null>(null);
  const ema50Ref    = useRef<ISeriesApi<'Line'> | null>(null);
  const ema200Ref   = useRef<ISeriesApi<'Line'> | null>(null);
  const bbUpperRef  = useRef<ISeriesApi<'Line'> | null>(null);
  const bbMiddleRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bbLowerRef  = useRef<ISeriesApi<'Line'> | null>(null);
  const rsiSeriesRef  = useRef<ISeriesApi<'Line'> | null>(null);
  const pdHighRef     = useRef<IPriceLine | null>(null);
  const pdLowRef      = useRef<IPriceLine | null>(null);
  const tradeLinesRef = useRef<IPriceLine[]>([]);

  // ── Mount main chart ──────────────────────────────────────────────────
  useEffect(() => {
    if (!mainRef.current) return;

    const chart = createChart(mainRef.current, {
      ...buildChartOpts(theme),
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

    // Bollinger Bands — dashed purple/violet lines
    const bbOpts = {
      color: '#7c3aed66', lineWidth: 1 as const,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    };
    const bbUpper  = chart.addLineSeries(bbOpts);
    const bbMiddle = chart.addLineSeries({ ...bbOpts, color: '#7c3aed33', lineStyle: LineStyle.Solid });
    const bbLower  = chart.addLineSeries(bbOpts);

    chartRef.current        = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    ema20Ref.current        = makeEMA('#3b82f6');
    ema50Ref.current        = makeEMA('#f59e0b');
    ema200Ref.current       = makeEMA('#ec4899');
    bbUpperRef.current      = bbUpper;
    bbMiddleRef.current     = bbMiddle;
    bbLowerRef.current      = bbLower;

    // Crosshair hover — populate OHLCV overlay
    chart.subscribeCrosshairMove(param => {
      if (!param.time) { setHover(null); return; }
      const cd = param.seriesData.get(candleSeries) as CandlestickData | undefined;
      const vd = param.seriesData.get(volumeSeries) as HistogramData | undefined;
      if (cd) setHover({ o: cd.open, h: cd.high, l: cd.low, c: cd.close, v: vd?.value ?? 0 });
    });

    chart.subscribeClick(param => {
      if (param.time) onCandleClick(param.time as number);
    });

    // Track live candle x position for countdown badge
    const updateCountdownX = () => {
      const t = liveTimeRef.current;
      if (t === null) { setCountdownX(null); return; }
      const x = chart.timeScale().timeToCoordinate(t as UTCTimestamp);
      setCountdownX(x !== null ? x : null);
    };
    chart.timeScale().subscribeVisibleTimeRangeChange(updateCountdownX);

    const obs = new ResizeObserver(() => {
      updateCountdownX();
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
      bbUpperRef.current = bbMiddleRef.current = bbLowerRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Theme sync (applyOptions — no remount needed) ─────────────────────
  useEffect(() => {
    const opts = buildChartOpts(theme);
    chartRef.current?.applyOptions(opts);
    rsiChartRef.current?.applyOptions({ ...opts, timeScale: { ...opts.timeScale, visible: false } });
  }, [theme]);

  // ── Mount / unmount RSI chart ─────────────────────────────────────────
  useEffect(() => {
    if (!showRSI || !rsiRef.current) return;

    const rsiChart = createChart(rsiRef.current, {
      ...buildChartOpts(theme),
      width:  rsiRef.current.clientWidth  || window.innerWidth,
      height: rsiRef.current.clientHeight || 130,
      timeScale: { ...buildChartOpts(theme).timeScale, visible: false },
      rightPriceScale: {
        borderColor: chartColors(theme).border,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
    });

    const rsiSeries = rsiChart.addLineSeries({
      color: '#7c3aed', lineWidth: 1,
      priceLineVisible: false, lastValueVisible: true,
      crosshairMarkerVisible: false,
    });

    // Reference lines: 70, 50, 30
    rsiSeries.createPriceLine({ price: 70, color: '#ef4444', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '' });
    rsiSeries.createPriceLine({ price: 50, color: '#6b7280', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '' });
    rsiSeries.createPriceLine({ price: 30, color: '#22c55e', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '' });

    rsiChartRef.current  = rsiChart;
    rsiSeriesRef.current = rsiSeries;

    // Seed with current candles
    if (candles.length > 0) {
      const closes = candles.map(c => c.close);
      const pts: { time: UTCTimestamp; value: number }[] = [];
      calcRSI(closes).forEach((v, i) => {
        if (v !== null) pts.push({ time: candles[i].time as UTCTimestamp, value: v });
      });
      rsiSeries.setData(pts);
    }

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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showRSI]);

  // ── Set chart data ────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    // Clear all series when candles reset (coin/TF switch) to prevent
    // "Cannot update oldest data" errors from stale timestamps
    if (candles.length === 0) {
      candleSeriesRef.current.setData([]);
      volumeSeriesRef.current?.setData([]);
      ema20Ref.current?.setData([]);
      ema50Ref.current?.setData([]);
      ema200Ref.current?.setData([]);
      bbUpperRef.current?.setData([]);
      bbMiddleRef.current?.setData([]);
      bbLowerRef.current?.setData([]);
      rsiSeriesRef.current?.setData([]);
      return;
    }

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
        color: c.close >= c.open ? '#22c55e33' : '#ef444433',
      }))
    );

    const closes = candles.map(c => c.close);

    const toLine = (vals: (number | null)[]) =>
      vals.reduce<{ time: UTCTimestamp; value: number }[]>((acc, v, i) => {
        if (v !== null) acc.push({ time: candles[i].time as UTCTimestamp, value: v });
        return acc;
      }, []);

    ema20Ref.current?.setData(toLine(calcEMA(closes, 20)));
    ema50Ref.current?.setData(toLine(calcEMA(closes, 50)));
    ema200Ref.current?.setData(toLine(calcEMA(closes, 200)));

    // Bollinger Bands
    const bands = calcBollingerBands(closes);
    const bbU: { time: UTCTimestamp; value: number }[] = [];
    const bbM: { time: UTCTimestamp; value: number }[] = [];
    const bbL: { time: UTCTimestamp; value: number }[] = [];
    bands.forEach((b, i) => {
      if (b.upper !== null && b.middle !== null && b.lower !== null) {
        const t = candles[i].time as UTCTimestamp;
        bbU.push({ time: t, value: b.upper });
        bbM.push({ time: t, value: b.middle });
        bbL.push({ time: t, value: b.lower });
      }
    });
    bbUpperRef.current?.setData(bbU);
    bbMiddleRef.current?.setData(bbM);
    bbLowerRef.current?.setData(bbL);

    if (rsiSeriesRef.current) {
      rsiSeriesRef.current.setData(toLine(calcRSI(closes)));
    }
  }, [candles]);

  // ── Live candle ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!liveCandle) return;
    const t = liveCandle.time as UTCTimestamp;
    // Guard against stale-data errors during coin/TF transitions
    try {
      candleSeriesRef.current?.update({ time: t, open: liveCandle.open, high: liveCandle.high, low: liveCandle.low, close: liveCandle.close });
      volumeSeriesRef.current?.update({ time: t, value: liveCandle.volume, color: liveCandle.close >= liveCandle.open ? '#22c55e33' : '#ef444433' });
    } catch { /* swallow: series cleared mid-transition */ }
  }, [liveCandle]);

  // ── EMA / BB visibility toggles ───────────────────────────────────────
  useEffect(() => { ema20Ref.current?.applyOptions({ visible: showEMA20 }); }, [showEMA20]);
  useEffect(() => { ema50Ref.current?.applyOptions({ visible: showEMA50 }); }, [showEMA50]);
  useEffect(() => { ema200Ref.current?.applyOptions({ visible: showEMA200 }); }, [showEMA200]);
  useEffect(() => {
    bbUpperRef.current?.applyOptions({ visible: showBB });
    bbMiddleRef.current?.applyOptions({ visible: showBB });
    bbLowerRef.current?.applyOptions({ visible: showBB });
  }, [showBB]);

  // ── Prev day H/L ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current || !prevDay) return;
    try { if (pdHighRef.current) candleSeriesRef.current.removePriceLine(pdHighRef.current); } catch {}
    try { if (pdLowRef.current)  candleSeriesRef.current.removePriceLine(pdLowRef.current);  } catch {}
    pdHighRef.current = candleSeriesRef.current.createPriceLine({ price: prevDay.high, color: '#4ade80', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PD H' });
    pdLowRef.current  = candleSeriesRef.current.createPriceLine({ price: prevDay.low,  color: '#f87171', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PD L' });
  }, [prevDay]);

  // ── Trade markers ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;
    const tfSec = TF_SECONDS[timeframe];
    const firstT = candles[0].time, lastT = candles[candles.length - 1].time;

    const tradeMs: SeriesMarker<UTCTimestamp>[] = tradeMarkers
      .filter(m => m.coin === coin && m.time >= firstT && m.time <= lastT + tfSec)
      .map(m => {
        const isEntry = m.type === 'ENTRY_LONG' || m.type === 'ENTRY_SHORT';
        const isLong  = m.type === 'ENTRY_LONG' || m.type === 'EXIT_TP';
        const isTP    = m.type === 'EXIT_TP';
        return {
          time:     m.time as UTCTimestamp,
          position: (isEntry ? (isLong ? 'belowBar' : 'aboveBar') : (isTP ? 'aboveBar' : 'belowBar')) as SeriesMarker<UTCTimestamp>['position'],
          color:    isEntry ? (isLong ? '#22c55e' : '#ef4444') : (isTP ? '#22c55e' : '#ef4444'),
          shape:    (isEntry ? (isLong ? 'arrowUp' : 'arrowDown') : 'circle') as SeriesMarker<UTCTimestamp>['shape'],
          text:     isEntry ? (m.type === 'ENTRY_LONG' ? 'L' : 'S') : (isTP ? 'TP' : 'SL'),
          size:     2,
        };
      });

    const all = [...tradeMs].sort((a, b) => (a.time as number) - (b.time as number));
    candleSeriesRef.current.setMarkers(all);
  }, [candles, timeframe, tradeMarkers, coin]);

  // ── Open trade TP/SL price lines ─────────────────────────────────────
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    tradeLinesRef.current.forEach(line => {
      try { candleSeriesRef.current!.removePriceLine(line); } catch {}
    });
    tradeLinesRef.current = [];
    openTrades.filter(t => t.coin === coin && t.status === 'OPEN').forEach(t => {
      const tp = candleSeriesRef.current!.createPriceLine({ price: t.takeProfit, color: '#22c55e99', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'APEX TP' });
      const sl = candleSeriesRef.current!.createPriceLine({ price: t.stopLoss,   color: '#ef444499', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'APEX SL' });
      tradeLinesRef.current.push(tp, sl);
    });
  }, [openTrades, coin]);

  // ── Scroll to news ────────────────────────────────────────────────────
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

  // ── Candle countdown timer ────────────────────────────────────────────
  const liveTimeRef = useRef<number | null>(null);
  liveTimeRef.current = liveCandle?.time ?? null;

  useEffect(() => {
    const tfSec = TF_SECONDS[timeframe];
    function update() {
      const openTime = liveTimeRef.current;
      if (openTime === null) { setCountdown(null); setCountdownX(null); return; }
      const remaining = Math.max(0, openTime + tfSec - Math.floor(Date.now() / 1000));
      const h = Math.floor(remaining / 3600);
      const m = Math.floor((remaining % 3600) / 60);
      const s = remaining % 60;
      setCountdown(
        h > 0
          ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
          : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
      );
      // Keep badge pinned under the live candle
      const x = chartRef.current?.timeScale().timeToCoordinate(openTime as UTCTimestamp);
      setCountdownX(x ?? null);
    }
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, [liveCandle?.time, timeframe]);

  // ── Derive display data (hover or last candle) ────────────────────────
  const last = candles.length > 0 ? candles[candles.length - 1] : null;
  const display: HoverData | null = hover ?? (last ? { o: last.open, h: last.high, l: last.low, c: last.close, v: last.volume } : null);
  const isUp = display ? display.c >= display.o : true;
  const chgPct = display && display.o > 0 ? ((display.c - display.o) / display.o) * 100 : 0;

  const overlayColor = isUp ? '#22c55e' : '#ef4444';
  const labelStyle: React.CSSProperties = { color: 'var(--text3)', marginRight: 4, fontSize: 10 };
  const valStyle: React.CSSProperties   = { color: 'var(--text)', marginRight: 12, fontSize: 11, fontFamily: 'monospace' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div ref={mainRef} style={{ flex: showRSI ? '0 0 70%' : '1', position: 'relative' }}>

        {/* OHLCV overlay */}
        {display && (
          <div style={{
            position: 'absolute', top: 8, left: 8, zIndex: 10,
            display: 'flex', alignItems: 'center', gap: 0,
            pointerEvents: 'none', userSelect: 'none',
          }}>
            <span style={labelStyle}>O</span><span style={valStyle}>{fmtPrice(display.o)}</span>
            <span style={labelStyle}>H</span><span style={valStyle}>{fmtPrice(display.h)}</span>
            <span style={labelStyle}>L</span><span style={valStyle}>{fmtPrice(display.l)}</span>
            <span style={labelStyle}>C</span>
            <span style={{ ...valStyle, color: overlayColor }}>{fmtPrice(display.c)}</span>
            <span style={{ fontSize: 10, color: overlayColor, marginRight: 12, fontFamily: 'monospace' }}>
              {chgPct >= 0 ? '+' : ''}{chgPct.toFixed(2)}%
            </span>
            <span style={labelStyle}>V</span>
            <span style={{ ...valStyle, marginRight: 0 }}>{fmtVol(display.v)}</span>
          </div>
        )}

        {candles.length === 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: '12px', letterSpacing: '0.1em', fontFamily: 'monospace', pointerEvents: 'none' }}>
            LOADING {coin}...
          </div>
        )}

        {/* Candle countdown — pinned under live candle on time axis */}
        {countdown && countdownX !== null && (
          <div style={{
            position: 'absolute', bottom: 0,
            left: countdownX, transform: 'translateX(-50%)',
            background: isUp ? '#22c55e' : '#ef4444',
            color: '#fff',
            fontSize: 11, fontFamily: "'SF Mono','Fira Code',monospace",
            fontWeight: 600, letterSpacing: '0.04em',
            padding: '2px 6px', borderRadius: 3,
            pointerEvents: 'none', userSelect: 'none',
            zIndex: 5, whiteSpace: 'nowrap',
          }}>
            {countdown}
          </div>
        )}
      </div>

      {showRSI && (
        <div ref={rsiRef} style={{ flex: '0 0 30%', borderTop: '1px solid var(--border)', position: 'relative' }}>
          <span style={{ position: 'absolute', top: 4, left: 8, fontSize: '9px', color: '#7c3aed', letterSpacing: '0.1em', fontFamily: 'monospace', zIndex: 1, pointerEvents: 'none' }}>RSI(14)</span>
        </div>
      )}
    </div>
  );
}
