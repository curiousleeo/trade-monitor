import { useEffect, useRef } from 'react';
import { init, dispose } from 'klinecharts';
import type { Chart } from 'klinecharts';
import { Candle, Coin, Timeframe, Trade } from '../types';
import { PrevDay } from '../hooks/usePrevDayOHLC';
import { TradeMarkerData } from '../hooks/useAITrader';

interface Props {
  candles:      Candle[];
  liveCandle:   Candle | null;
  coin:         Coin;
  timeframe:    Timeframe;
  theme:        'light' | 'dark';
  prevDay:      PrevDay | null;
  showEMA:      boolean;
  showBOLL:     boolean;
  showRSI:      boolean;
  showMACD:     boolean;
  tradeMarkers?: TradeMarkerData[];
  openTrades?:   Trade[];
  scrollToTime?: number | null;
}

function toKLine(c: Candle) {
  return { timestamp: c.time * 1000, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume };
}

function buildStyles(theme: 'light' | 'dark') {
  const dark = theme === 'dark';
  return {
    grid: {
      horizontal: { color: dark ? '#111827' : '#f3f4f6' },
      vertical:   { color: dark ? '#111827' : '#f3f4f6' },
    },
    candle: {
      bar: {
        upColor:          '#22c55e', downColor:          '#ef4444',
        noChangeColor:    '#6b7280',
        upBorderColor:    '#22c55e', downBorderColor:    '#ef4444',
        noChangeBorderColor: '#6b7280',
        upWickColor:      '#22c55e', downWickColor:      '#ef4444',
        noChangeWickColor:'#6b7280',
      },
      tooltip: {
        labels: ['T', 'O', 'H', 'L', 'C', 'V'],
        text:   { color: dark ? '#9ca3af' : '#6b7280', size: 11, family: "'SF Mono','Fira Code',monospace" },
      },
    },
    indicator: {
      tooltip: { text: { color: dark ? '#9ca3af' : '#6b7280', size: 11, family: "'SF Mono','Fira Code',monospace" } },
    },
    xAxis: {
      axisLine: { color: dark ? '#1f2937' : '#e5e7eb' },
      tickLine: { color: dark ? '#1f2937' : '#e5e7eb' },
      tickText: { color: dark ? '#6b7280' : '#9ca3af', size: 11, family: "'SF Mono','Fira Code',monospace" },
    },
    yAxis: {
      axisLine: { color: dark ? '#1f2937' : '#e5e7eb' },
      tickLine: { color: dark ? '#1f2937' : '#e5e7eb' },
      tickText: { color: dark ? '#6b7280' : '#9ca3af', size: 11, family: "'SF Mono','Fira Code',monospace" },
    },
    crosshair: {
      horizontal: {
        line: { color: dark ? '#374151' : '#d1d5db' },
        text: { backgroundColor: dark ? '#1f2937' : '#f3f4f6', color: dark ? '#e5e7eb' : '#374151', borderColor: dark ? '#374151' : '#d1d5db' },
      },
      vertical: {
        line: { color: dark ? '#374151' : '#d1d5db' },
        text: { backgroundColor: dark ? '#1f2937' : '#f3f4f6', color: dark ? '#e5e7eb' : '#374151', borderColor: dark ? '#374151' : '#d1d5db' },
      },
    },
    background: { color: dark ? '#0a0a0a' : '#ffffff' },
  };
}

const CANDLE_PANE = 'candle_pane';

export function KLineChart({
  candles, liveCandle, theme,
  prevDay, showEMA, showBOLL, showRSI, showMACD,
  tradeMarkers = [], openTrades = [],
  scrollToTime,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<Chart | null>(null);

  // Indicator pane ID refs
  const rsiPaneRef  = useRef<string | null>(null);
  const macdPaneRef = useRef<string | null>(null);

  // Overlay ID refs
  const pdLineIds   = useRef<string[]>([]);
  const tpSlIds     = useRef<string[]>([]);
  const markerIds   = useRef<string[]>([]);

  // ── Mount / unmount ──────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = init(containerRef.current, { styles: buildStyles(theme) });
    if (!chart) return;
    chartRef.current = chart;

    const obs = new ResizeObserver(() => chart.resize());
    obs.observe(containerRef.current);

    return () => {
      obs.disconnect();
      if (containerRef.current) dispose(containerRef.current);
      chartRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Theme ─────────────────────────────────────────────────────────────
  useEffect(() => {
    chartRef.current?.setStyles(buildStyles(theme));
  }, [theme]);

  // ── Candle data ───────────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.applyNewData(candles.length ? candles.map(toKLine) : []);
  }, [candles]);

  // ── Live candle ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!liveCandle || !chartRef.current) return;
    try { chartRef.current.updateData(toKLine(liveCandle)); } catch {}
  }, [liveCandle]);

  // ── EMA overlay ───────────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (showEMA) {
      chart.createIndicator('EMA', false, { id: CANDLE_PANE });
    } else {
      chart.removeIndicator(CANDLE_PANE, 'EMA');
    }
  }, [showEMA]);

  // ── Bollinger Bands ───────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (showBOLL) {
      chart.createIndicator('BOLL', false, { id: CANDLE_PANE });
    } else {
      chart.removeIndicator(CANDLE_PANE, 'BOLL');
    }
  }, [showBOLL]);

  // ── RSI (sub-pane) ────────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (showRSI) {
      if (!rsiPaneRef.current) {
        const paneId = chart.createIndicator('RSI', false, { height: 80 });
        rsiPaneRef.current = paneId ?? null;
      }
    } else {
      if (rsiPaneRef.current) {
        chart.removeIndicator(rsiPaneRef.current);
        rsiPaneRef.current = null;
      }
    }
  }, [showRSI]);

  // ── MACD (sub-pane) ───────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (showMACD) {
      if (!macdPaneRef.current) {
        const paneId = chart.createIndicator('MACD', false, { height: 100 });
        macdPaneRef.current = paneId ?? null;
      }
    } else {
      if (macdPaneRef.current) {
        chart.removeIndicator(macdPaneRef.current);
        macdPaneRef.current = null;
      }
    }
  }, [showMACD]);

  // ── PD H/L lines ─────────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    // Clear old
    pdLineIds.current.forEach(id => chart.removeOverlay(id));
    pdLineIds.current = [];
    if (!prevDay) return;

    const highId = chart.createOverlay({
      name: 'horizontalStraightLine',
      lock: true,
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: true,
      points: [{ timestamp: 0, value: prevDay.high }],
      styles: { line: { color: '#4ade80', size: 1, style: 'dashed' } },
    } as any);
    const lowId = chart.createOverlay({
      name: 'horizontalStraightLine',
      lock: true,
      needDefaultPointFigure: false,
      needDefaultXAxisFigure: false,
      needDefaultYAxisFigure: true,
      points: [{ timestamp: 0, value: prevDay.low }],
      styles: { line: { color: '#f87171', size: 1, style: 'dashed' } },
    } as any);
    if (typeof highId === 'string') pdLineIds.current.push(highId);
    if (typeof lowId  === 'string') pdLineIds.current.push(lowId);
  }, [prevDay]);

  // ── Open trade TP/SL lines ────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    tpSlIds.current.forEach(id => chart.removeOverlay(id));
    tpSlIds.current = [];

    openTrades.filter(t => t.status === 'OPEN').forEach(t => {
      const tpId = chart.createOverlay({
        name: 'horizontalStraightLine', lock: true,
        points: [{ timestamp: 0, value: t.takeProfit }],
        styles: { line: { color: '#22c55e99', size: 1, style: 'dashed' } },
      } as any);
      const slId = chart.createOverlay({
        name: 'horizontalStraightLine', lock: true,
        points: [{ timestamp: 0, value: t.stopLoss }],
        styles: { line: { color: '#ef444499', size: 1, style: 'dashed' } },
      } as any);
      if (typeof tpId === 'string') tpSlIds.current.push(tpId);
      if (typeof slId === 'string') tpSlIds.current.push(slId);
    });
  }, [openTrades]);

  // ── AI trade markers ─────────────────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || candles.length === 0) return;
    markerIds.current.forEach(id => chart.removeOverlay(id));
    markerIds.current = [];

    // Build a quick lookup from candle time → candle
    const candleMap = new Map(candles.map(c => [c.time, c]));

    tradeMarkers.forEach(m => {
      const candle = candleMap.get(m.time);
      if (!candle) return;
      const isEntry = m.type === 'ENTRY_LONG' || m.type === 'ENTRY_SHORT';
      const isLong  = m.type === 'ENTRY_LONG'  || m.type === 'EXIT_TP';
      const color   = isEntry ? (isLong ? '#22c55e' : '#ef4444') : (m.type === 'EXIT_TP' ? '#22c55e' : '#ef4444');
      const label   = m.type === 'ENTRY_LONG' ? 'L↑' : m.type === 'ENTRY_SHORT' ? 'S↓' : m.type === 'EXIT_TP' ? 'TP' : 'SL';
      // Entry longs below bar, entry shorts above bar; exits reversed
      const price   = isEntry ? (isLong ? candle.low : candle.high) : (isLong ? candle.high : candle.low);

      const id = chart.createOverlay({
        name: 'simpleAnnotation',
        points: [{ timestamp: m.time * 1000, value: price }],
        extendData: label,
        styles: { line: { color }, text: { color, size: 10 } },
      } as any);
      if (typeof id === 'string') markerIds.current.push(id);
    });
  }, [candles, tradeMarkers]);

  // ── Scroll to time ────────────────────────────────────────────────────
  useEffect(() => {
    if (!scrollToTime || !chartRef.current) return;
    chartRef.current.scrollToTimestamp(scrollToTime * 1000);
  }, [scrollToTime]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
