/**
 * APEX Storage Layer
 *
 * Thin abstraction over localStorage — built to swap for Supabase with minimal changes.
 * Only writes on meaningful events (trade open / trade close), never on price ticks.
 *
 * To upgrade to Supabase later:
 *   1. npm install @supabase/supabase-js
 *   2. Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to Vercel env vars
 *   3. Replace the functions below with Supabase calls (same signatures)
 */

import { Trade } from '../types';

const KEY_PORTFOLIO = 'apex-v2-portfolio';
const KEY_HISTORY   = 'apex-v2-history';   // closed trades only (separate key)

export interface StoredPortfolio {
  balance:      number;
  startBalance: number;
  openTrades:   Trade[];
}

// ─── Load ─────────────────────────────────────────────────────────────────────

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

// ─── Write — only called on trade events ─────────────────────────────────────

/** Called once when a trade is opened. */
export function onTradeOpened(portfolio: StoredPortfolio): void {
  try {
    localStorage.setItem(KEY_PORTFOLIO, JSON.stringify(portfolio));
  } catch { /* ignore */ }
}

/** Called once when a trade closes (TP or SL). Appends to history, caps at 200 records. */
export function onTradeClosed(portfolio: StoredPortfolio, closedTrade: Trade): void {
  try {
    localStorage.setItem(KEY_PORTFOLIO, JSON.stringify(portfolio));

    // History is stored separately to keep the main portfolio key small
    const history = loadHistory();
    const next = [closedTrade, ...history].slice(0, 200);
    localStorage.setItem(KEY_HISTORY, JSON.stringify(next));
  } catch { /* ignore */ }
}

/** Full reset — wipes both keys. */
export function resetStorage(): void {
  try {
    localStorage.removeItem(KEY_PORTFOLIO);
    localStorage.removeItem(KEY_HISTORY);
  } catch { /* ignore */ }
}
