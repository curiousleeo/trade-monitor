"""
fetch_data.py
─────────────
Downloads historical OHLCV candles from Binance public REST API
and saves them as Parquet files (fast, compressed, column-oriented).

Usage:
    python fetch_data.py                        # BTC 15m since 2021
    python fetch_data.py --coin ETH --tf 1h     # ETH 1h since 2021
    python fetch_data.py --coin BTC --since 2020-01-01

Output:
    data/BTC_15m.parquet
    data/ETH_1h.parquet
    ...
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

BASE_URL  = "https://api.binance.com/api/v3/klines"
LIMIT     = 1000   # max candles per request
DATA_DIR  = Path(__file__).parent / "data"

TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


def fetch_chunk(symbol: str, interval: str, start_ms: int) -> list[list]:
    resp = requests.get(BASE_URL, params={
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "limit":     LIMIT,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all(coin: str, timeframe: str, since: str) -> pd.DataFrame:
    symbol   = f"{coin}USDT"
    start_ms = int(datetime.strptime(since, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc)
                   .timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    step_ms  = TF_MS[timeframe] * LIMIT

    rows = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=console,
    ) as progress:
        total = max(1, (end_ms - start_ms) // step_ms + 1)
        task  = progress.add_task(f"Fetching {symbol} {timeframe}", total=total)

        cursor = start_ms
        while cursor < end_ms:
            chunk = fetch_chunk(symbol, timeframe, cursor)
            if not chunk:
                break
            rows.extend(chunk)
            cursor = chunk[-1][0] + TF_MS[timeframe]
            progress.advance(task)
            time.sleep(0.12)   # stay well under Binance rate limit (1200 req/min)

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",  default="BTC")
    parser.add_argument("--tf",    default="15m")
    parser.add_argument("--since", default="2021-01-01")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{args.coin}_{args.tf}.parquet"

    console.print(f"[bold]Fetching {args.coin} {args.tf} since {args.since}[/bold]")
    df = fetch_all(args.coin, args.tf, args.since)
    df.to_parquet(out, index=False)

    console.print(f"[green]✓ Saved {len(df):,} candles → {out}[/green]")
    console.print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
