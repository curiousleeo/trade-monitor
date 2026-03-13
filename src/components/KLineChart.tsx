import { useEffect, useRef } from 'react';
import { init, dispose, registerOverlay } from 'klinecharts';
import type { Chart } from 'klinecharts';

// ── Register custom TP/SL line overlay (once at module level) ─────────────────
let _tpslRegistered = false;
function ensureTpSlOverlay() {
  if (_tpslRegistered) return;
  _tpslRegistered = true;
  registerOverlay({
    name: 'tpslLine',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures({ coordinates, bounding, overlay }: any) {
      const y          = coordinates[0].y;
      const label      = (overlay.extendData ?? '') as string;
      const isTP       = label.startsWith('TP');
      const lineColor  = isTP ? '#22c55e' : '#ef4444';
      const bgColor    = isTP ? '#22c55ecc' : '#ef4444cc';
      return [
        // Dashed horizontal line
        {
          type: 'line',
          attrs: { coordinates: [{ x: 0, y }, { x: bounding.width, y }] },
          styles: { style: 'dashed', dashedValue: [4, 3], color: lineColor, size: 1 },
          ignoreEvent: true,
        },
        // Inline label box (auto-draws background via backgroundColor)
        {
          type: 'text',
          attrs: { x: bounding.width - 4, y, text: label, align: 'right', baseline: 'middle' },
          styles: {
            color: '#ffffff',
            backgroundColor: bgColor,
            size: 10,
            weight: 'bold',
            family: "'Inter', -apple-system, sans-serif",
            paddingLeft: 7,
            paddingRight: 7,
            paddingTop: 3,
            paddingBottom: 3,
            borderRadius: 3,
          },
          ignoreEvent: true,
        },
      ];
    },
  });
}
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
  // Colors aligned with CSS design tokens
  const bg         = dark ? '#131722' : '#f5f6fa';
  const gridLine   = dark ? '#1a1f2e' : '#eef0f8';
  const axisLine   = dark ? '#2a2e39' : '#dde0f0';
  const tickText   = dark ? '#787b86' : '#4a5070';
  const crossLine  = dark ? '#363c4e' : '#c8cce0';
  const crossBg    = dark ? '#1e2130' : '#eef0f8';
  const crossText  = dark ? '#d1d4dc' : '#0d0f1a';
  const tooltipTxt = dark ? '#787b86' : '#4a5070';
  const upColor    = dark ? '#26a69a' : '#00a854';
  const dnColor    = dark ? '#ef5350' : '#e0152e';
  const neutColor  = dark ? '#4c525e' : '#8a90b0';
  const monoFont   = "'JetBrains Mono','SF Mono','Fira Code','Consolas',monospace";
  return {
    grid: {
      horizontal: { color: gridLine },
      vertical:   { color: gridLine },
    },
    candle: {
      bar: {
        upColor:             upColor,   downColor:             dnColor,
        noChangeColor:       neutColor,
        upBorderColor:       upColor,   downBorderColor:       dnColor,
        noChangeBorderColor: neutColor,
        upWickColor:         upColor,   downWickColor:         dnColor,
        noChangeWickColor:   neutColor,
      },
      tooltip: {
        labels: ['T', 'O', 'H', 'L', 'C', 'V'],
        text:   { color: tooltipTxt, size: 11, family: monoFont },
      },
    },
    indicator: {
      tooltip: { text: { color: tooltipTxt, size: 11, family: monoFont } },
    },
    xAxis: {
      axisLine: { color: axisLine },
      tickLine: { color: axisLine },
      tickText: { color: tickText, size: 11, family: monoFont },
    },
    yAxis: {
      axisLine: { color: axisLine },
      tickLine: { color: axisLine },
      tickText: { color: tickText, size: 11, family: monoFont },
    },
    crosshair: {
      horizontal: {
        line: { color: crossLine },
        text: { backgroundColor: crossBg, color: crossText, borderColor: crossLine },
      },
      vertical: {
        line: { color: crossLine },
        text: { backgroundColor: crossBg, color: crossText, borderColor: crossLine },
      },
    },
    background: { color: bg },
  };
}

const CANDLE_PANE = 'candle_pane';

export function KLineChart({
  candles, liveCandle, coin, theme,
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
    ensureTpSlOverlay();
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

    openTrades.filter(t => t.status === 'OPEN' && t.coin === coin).forEach(t => {
      const fmtPrice = (v: number) =>
        v >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(2);
      const tpId = chart.createOverlay({
        name: 'tpslLine', lock: true,
        points: [{ timestamp: 0, value: t.takeProfit }],
        extendData: `TP  ${fmtPrice(t.takeProfit)}`,
      } as any);
      const slId = chart.createOverlay({
        name: 'tpslLine', lock: true,
        points: [{ timestamp: 0, value: t.stopLoss }],
        extendData: `SL  ${fmtPrice(t.stopLoss)}`,
      } as any);
      if (typeof tpId === 'string') tpSlIds.current.push(tpId);
      if (typeof slId === 'string') tpSlIds.current.push(slId);
    });
  }, [openTrades, coin]);

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
