"""Small demo runner for financeAnalysis module.

Usage (PowerShell):
  python .\financeAnalysis\run_demo.py AAPL

This will print JSON outputs for a few functions.
"""
import sys
import json
from financeAnalysis import (
    get_snapshot,
    get_current_price,
    get_price_history,
    get_moving_average,
    get_RSI,
)


def pretty_print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py SYMBOL")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    print(f"Running demo for {symbol}...\n")

    print("Snapshot:")
    pretty_print(get_snapshot(symbol))

    print("\nCurrent price:")
    pretty_print(get_current_price(symbol))

    print("\nRecent price history (1mo):")
    pretty_print(get_price_history(symbol, period='1mo', interval='1d'))

    print("\n20-day MA:")
    pretty_print(get_moving_average(symbol, window=20, period='3mo'))

    print("\n14-day RSI:")
    pretty_print(get_RSI(symbol, window=14, period='3mo'))


if __name__ == '__main__':
    main()
