"""Quick smoke tests for financeAnalysis package.

Run from project root:
  python .\test.py

This script performs a few basic checks and exits non-zero on failure.
"""
import sys
import os
import json

# Ensure project root is on sys.path so `import financeAnalysis` works.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def fail(msg, exc=None):
    print("TEST FAILED:", msg)
    if exc:
        print(str(exc))
    sys.exit(2)

def main():
    try:
        from financeAnalysis import get_snapshot, get_current_price, get_price_history, get_moving_average
    except Exception as e:
        fail("Importing financeAnalysis failed", e)

    symbol = 'AAPL'

    print('Running snapshot test...')
    snap = get_snapshot(symbol)
    if not isinstance(snap, dict):
        fail('get_snapshot did not return a dict', type(snap))
    if 'error' in snap:
        fail('get_snapshot returned error', snap.get('error'))
    if snap.get('symbol', '').upper() != symbol:
        fail('get_snapshot symbol mismatch', snap)

    print('Running current price test...')
    price = get_current_price(symbol)
    if not isinstance(price, dict):
        fail('get_current_price did not return a dict', type(price))
    if 'error' in price:
        fail('get_current_price returned error', price.get('error'))

    print('Running price history test...')
    hist = get_price_history(symbol, period='1mo', interval='1d')
    if not isinstance(hist, dict):
        fail('get_price_history did not return a dict', type(hist))
    if 'error' in hist:
        fail('get_price_history returned error', hist.get('error'))
    if 'prices' not in hist:
        fail('get_price_history missing prices key', hist)

    print('Running MA test...')
    ma = get_moving_average(symbol, window=5, period='1mo')
    if not isinstance(ma, dict):
        fail('get_moving_average did not return a dict', type(ma))
    if 'error' in ma:
        fail('get_moving_average returned error', ma.get('error'))

    print('\nALL TESTS PASSED')
    print(json.dumps({
        'snapshot': snap,
        'price': price,
        'history_count': len(hist.get('prices', [])),
        'ma_points': len(ma.get('result', [])),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
