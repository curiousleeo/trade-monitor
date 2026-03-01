/**
 * APEX Storage Layer
 *
 * Auto-selects backend:
 *   • Supabase  — when VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY are set in Vercel
 *   • localStorage — fallback for local dev or before Supabase is connected
 *
 * Writes only on trade open / trade close. Never on price ticks or predictions.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { Trade } from '../types';

// ─── Supabase client (lazy, only if env vars present) ─────────────────────────

const SUPABASE_URL  = import.meta.env.VITE_SUPABASE_URL  as string | undefined;
const SUPABASE_KEY  = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

let _supabase: SupabaseClient | null = null;

function supabase(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_KEY) return null;
  if (!_supabase) _supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
  return _supabase;
}

export const isSupabaseEnabled = !!(SUPABASE_URL && SUPABASE_KEY);

// ─── Types ────────────────────────────────────────────────────────────────────

export interface StoredPortfolio {
  balance:      number;
  startBalance: number;
  openTrades:   Trade[];
}

// ─── localStorage keys (fallback) ─────────────────────────────────────────────

const KEY_PORTFOLIO = 'apex-v2-portfolio';
const KEY_HISTORY   = 'apex-v2-history';

// ─── Load ─────────────────────────────────────────────────────────────────────

export async function loadPortfolioAsync(): Promise<StoredPortfolio> {
  const sb = supabase();
  if (sb) {
    try {
      const { data } = await sb.from('portfolio').select('*').eq('id', 'apex').single();
      if (data) {
        // Fetch open trades
        const { data: trades } = await sb
          .from('trades')
          .select('*')
          .eq('status', 'OPEN');
        return {
          balance:      data.balance,
          startBalance: data.start_balance,
          openTrades:   (trades ?? []).map(dbToTrade),
        };
      }
    } catch { /* fall through to localStorage */ }
  }
  return loadPortfolio();
}

export async function loadHistoryAsync(): Promise<Trade[]> {
  const sb = supabase();
  if (sb) {
    try {
      const { data } = await sb
        .from('trades')
        .select('*')
        .neq('status', 'OPEN')
        .order('entry_time', { ascending: false })
        .limit(200);
      if (data) return data.map(dbToTrade);
    } catch { /* fall through */ }
  }
  return loadHistory();
}

// ─── Sync write: localStorage (instant, used in hook for optimistic UI) ────────

export function loadPortfolio(): StoredPortfolio {
  try {
    const raw = localStorage.getItem(KEY_PORTFOLIO);
    if (raw) return JSON.parse(raw) as StoredPortfolio;
  } catch { /* ignore */ }
  return { balance: 1000, startBalance: 1000, openTrades: [] };
}

export function loadHistory(): Trade[] {
  try {
    const raw = localStorage.getItem(KEY_HISTORY);
    if (raw) return JSON.parse(raw) as Trade[];
  } catch { /* ignore */ }
  return [];
}

// ─── Write events — called only on trade open / close ─────────────────────────

export function onTradeOpened(portfolio: StoredPortfolio, trade: Trade): void {
  // Optimistic local write (instant)
  try { localStorage.setItem(KEY_PORTFOLIO, JSON.stringify(portfolio)); } catch { /* ignore */ }

  // Async Supabase write (fire-and-forget)
  const sb = supabase();
  if (!sb) return;
  sb.from('portfolio').upsert({
    id:            'apex',
    balance:       portfolio.balance,
    start_balance: portfolio.startBalance,
    updated_at:    new Date().toISOString(),
  }).then(() => {});
  sb.from('trades').insert(tradeToDb(trade)).then(() => {});
}

export function onTradeClosed(portfolio: StoredPortfolio, closedTrade: Trade): void {
  // Optimistic local write
  try {
    localStorage.setItem(KEY_PORTFOLIO, JSON.stringify(portfolio));
    const history = loadHistory();
    localStorage.setItem(KEY_HISTORY, JSON.stringify([closedTrade, ...history].slice(0, 200)));
  } catch { /* ignore */ }

  // Async Supabase write
  const sb = supabase();
  if (!sb) return;
  sb.from('portfolio').upsert({
    id:            'apex',
    balance:       portfolio.balance,
    start_balance: portfolio.startBalance,
    updated_at:    new Date().toISOString(),
  }).then(() => {});
  sb.from('trades').update(tradeToDb(closedTrade)).eq('id', closedTrade.id).then(() => {});
}

export function resetStorage(): void {
  try {
    localStorage.removeItem(KEY_PORTFOLIO);
    localStorage.removeItem(KEY_HISTORY);
  } catch { /* ignore */ }

  const sb = supabase();
  if (!sb) return;
  // Wipe portfolio row and all trades
  sb.from('trades').delete().neq('id', '').then(() => {});
  sb.from('portfolio').upsert({
    id:            'apex',
    balance:       1000,
    start_balance: 1000,
    updated_at:    new Date().toISOString(),
  }).then(() => {});
}

// ─── DB ↔ Trade mappers ───────────────────────────────────────────────────────

function tradeToDb(t: Trade) {
  return {
    id:            t.id,
    prediction_id: t.predictionId,
    coin:          t.coin,
    direction:     t.direction,
    entry_price:   t.entryPrice,
    entry_time:    t.entryTime,
    stop_loss:     t.stopLoss,
    take_profit:   t.takeProfit,
    size:          t.size,
    status:        t.status,
    exit_price:    t.exitPrice ?? null,
    exit_time:     t.exitTime  ?? null,
    pnl:           t.pnl       ?? null,
    pnl_pct:       t.pnlPct    ?? null,
  };
}

function dbToTrade(r: Record<string, unknown>): Trade {
  return {
    id:           r.id as string,
    predictionId: r.prediction_id as string,
    coin:         r.coin as Trade['coin'],
    direction:    r.direction as Trade['direction'],
    entryPrice:   r.entry_price as number,
    entryTime:    r.entry_time as number,
    stopLoss:     r.stop_loss as number,
    takeProfit:   r.take_profit as number,
    size:         r.size as number,
    status:       r.status as Trade['status'],
    exitPrice:    r.exit_price as number | undefined,
    exitTime:     r.exit_time  as number | undefined,
    pnl:          r.pnl        as number | undefined,
    pnlPct:       r.pnl_pct    as number | undefined,
  };
}
