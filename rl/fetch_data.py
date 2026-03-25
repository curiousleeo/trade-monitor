"""
fetch_data.py
─────────────
Downloads historical OHLCV candles + funding rates from Binance
and saves them as Parquet files.

Usage:
    python fetch_data.py                            # BTC 15m since 2023
    python fetch_data.py --coin ETH --tf 1h
    python fetch_data.py --coin BTC --since 2021-01-01
    python fetch_data.py --funding                  # also download funding rates
    python fetch_data.py --coin BTC --funding-only  # funding rates only

Output:
    data/BTC_15m.parquet
    data/BTC_funding.parquet   (if --funding or --funding-only)
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# ── Binance endpoints ─────────────────────────────────────────────────────────
SPOT_URL    = "https://api.binance.com/api/v3/klines"
FUTURES_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
LIMIT_OHLCV    = 1000   # max candles per klines request
LIMIT_FUNDING  = 1000   # max records per fundingRate request

TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

FUNDING_INTERVAL_MS = 8 * 3_600_000   # Binance settles funding every 8h


# ── OHLCV ─────────────────────────────────────────────────────────────────────

def _fetch_ohlcv_chunk(symbol: str, interval: str, start_ms: int) -> list:
    resp = requests.get(SPOT_URL, params={
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "limit":     LIMIT_OHLCV,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_ohlcv(coin: str, timeframe: str, since: str) -> pd.DataFrame:
    """Download all OHLCV candles for a coin/timeframe since a given date."""
    symbol   = f"{coin}USDT"
    start_ms = int(datetime.strptime(since, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    step_ms  = TF_MS[timeframe] * LIMIT_OHLCV

    total  = max(1, (end_ms - start_ms) // step_ms + 1)
    cursor = start_ms
    rows   = []
    batch  = 0

    print(f"Fetching {symbol} {timeframe} since {since}", flush=True)

    while cursor < end_ms:
        chunk = _fetch_ohlcv_chunk(symbol, timeframe, cursor)
        if not chunk:
            break
        rows.extend(chunk)
        cursor = chunk[-1][0] + TF_MS[timeframe]
        batch += 1
        pct = min(100, int(batch / total * 100))
        bar = "#" * (pct // 3) + "-" * (34 - pct // 3)
        print(f"\r  [{bar}] {pct:3d}%  {batch}/{total} batches", end="", flush=True)
        time.sleep(0.12)

    print(flush=True)

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["time", "open", "high", "low", "close", "volume",
             "taker_buy_base", "trades"]].copy()
    for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


# ── Funding rate ──────────────────────────────────────────────────────────────

def _fetch_funding_chunk(symbol: str, start_ms: int) -> list:
    resp = requests.get(FUTURES_URL, params={
        "symbol":    symbol,
        "startTime": start_ms,
        "limit":     LIMIT_FUNDING,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_funding(coin: str, since: str) -> pd.DataFrame:
    """
    Download funding rate history from Binance futures.
    Funding settles every 8h (00:00, 08:00, 16:00 UTC).
    Returns a DataFrame with columns: time, funding_rate
    """
    symbol   = f"{coin}USDT"
    start_ms = int(datetime.strptime(since, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)

    total  = max(1, (end_ms - start_ms) // (FUNDING_INTERVAL_MS * LIMIT_FUNDING) + 1)
    cursor = start_ms
    rows   = []
    batch  = 0

    print(f"Fetching {symbol} funding rates since {since}", flush=True)

    while cursor < end_ms:
        chunk = _fetch_funding_chunk(symbol, cursor)
        if not chunk:
            break
        rows.extend(chunk)
        cursor = int(chunk[-1]["fundingTime"]) + FUNDING_INTERVAL_MS
        batch += 1
        pct = min(100, int(batch / total * 100))
        bar = "#" * (pct // 3) + "-" * (34 - pct // 3)
        print(f"\r  [{bar}] {pct:3d}%  {batch}/{total} batches", end="", flush=True)
        time.sleep(0.12)

    print(flush=True)

    df = pd.DataFrame(rows)
    df = df[["fundingTime", "fundingRate"]].copy()
    df.columns = ["time", "funding_rate"]
    df["time"]         = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["funding_rate"].astype(float)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def align_funding_to_15m(df_15m: pd.DataFrame, df_funding: pd.DataFrame) -> pd.Series:
    """
    Forward-fill 8h funding rates onto a 15m candle index.
    At each 15m step, the agent sees the most recently SETTLED funding rate.
    Returns a Series aligned to df_15m's index.
    """
    funding = df_funding.set_index("time")["funding_rate"]
    # Reindex to 15m timestamps, then forward-fill (rate is constant until next settlement)
    aligned = funding.reindex(df_15m["time"]).ffill().bfill()
    aligned.index = df_15m.index
    return aligned


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",        default="BTC")
    parser.add_argument("--tf",          default="15m")
    parser.add_argument("--since",       default="2023-01-01")
    parser.add_argument("--funding",     action="store_true",
                        help="Also download funding rates")
    parser.add_argument("--funding-only", action="store_true",
                        help="Download funding rates only (skip OHLCV)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not args.funding_only:
        out = DATA_DIR / f"{args.coin}_{args.tf}.parquet"
        df  = fetch_ohlcv(args.coin, args.tf, args.since)
        df.to_parquet(out, index=False)
        print(f"✓ Saved {len(df):,} candles → {out}", flush=True)
        print(df.tail(3).to_string(), flush=True)

    if args.funding or args.funding_only:
        out_f = DATA_DIR / f"{args.coin}_funding.parquet"
        df_f  = fetch_funding(args.coin, args.since)
        df_f.to_parquet(out_f, index=False)
        print(f"✓ Saved {len(df_f):,} funding records → {out_f}", flush=True)
        print(df_f.tail(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
