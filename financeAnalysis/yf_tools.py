"""Lightweight yfinance wrappers for S&P500-focused function-calling.

Assumptions:
- Users will pass valid S&P500 ticker symbols (e.g. AAPL, MSFT).
- Internet access is available for yfinance to fetch live data.

Design: functions return JSON-serializable dicts so they can be passed back into an LLM as tool responses.
"""
from typing import List, Dict, Any, Optional
import datetime
import traceback

import numpy as np
import pandas as pd
import yfinance as yf


def _safe_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def _df_to_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df['date'] = df.iloc[:, 0].dt.strftime('%Y-%m-%d')
        df = df.drop(df.columns[0], axis=1)
    return df.to_dict(orient='records')


def get_snapshot(symbol: str) -> Dict[str, Any]:
    """Return a compact snapshot: price, pe, market cap, previous close.

    Output is JSON-serializable.
    """
    try:
        t = _safe_ticker(symbol)
        info = t.info if hasattr(t, 'info') else t.get_info()
        # keep keys safe
        return {
            "symbol": symbol,
            "last_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "timestamp": int(datetime.datetime.utcnow().timestamp()),
        }
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_current_price(symbol: str) -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        info = t.info if hasattr(t, 'info') else t.get_info()
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        return {
            "symbol": symbol,
            "last_price": price,
            "previous_close": info.get('previousClose'),
            "timestamp": int(datetime.datetime.utcnow().timestamp()),
        }
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_price_history(symbol: str, period: str = '1y', interval: str = '1d') -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        df = t.history(period=period, interval=interval)
        records = []
        if not df.empty:
            df2 = df.reset_index()
            for _, r in df2.iterrows():
                records.append({
                    "date": r['Date'].strftime('%Y-%m-%d'),
                    "open": float(r['Open']) if not pd.isna(r['Open']) else None,
                    "high": float(r['High']) if not pd.isna(r['High']) else None,
                    "low": float(r['Low']) if not pd.isna(r['Low']) else None,
                    "close": float(r['Close']) if not pd.isna(r['Close']) else None,
                    "volume": int(r['Volume']) if 'Volume' in r and not pd.isna(r['Volume']) else None,
                })
        return {"symbol": symbol, "prices": records}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_financials(symbol: str, period: str = 'annual') -> Dict[str, Any]:
    """Return income_statement, balance_sheet, cash_flow as dicts keyed by year (latest first).

    period: 'annual' or 'quarterly'
    """
    try:
        t = _safe_ticker(symbol)
        # yfinance stores .financials (annual) and .quarterly_financials
        if period == 'quarterly':
            inc = t.quarterly_financials
            bal = t.quarterly_balance_sheet
            cash = t.quarterly_cashflow
        else:
            inc = t.financials
            bal = t.balance_sheet
            cash = t.cashflow

        def df_to_col_dict(df: pd.DataFrame) -> Dict[str, Any]:
            if df is None or df.empty:
                return {}
            # columns are timestamps; convert to string year or date
            out = {}
            for col in df.columns:
                key = str(col) if not hasattr(col, 'year') else str(col.year)
                out[key] = df[col].dropna().to_dict()
            return out

        return {
            "symbol": symbol,
            "income_statement": df_to_col_dict(inc),
            "balance_sheet": df_to_col_dict(bal),
            "cash_flow": df_to_col_dict(cash),
        }
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_moving_average(symbol: str, window: int = 20, period: str = '6mo', interval: str = '1d') -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return {"symbol": symbol, "window": window, "result": []}
        ma = df['Close'].rolling(window=window).mean()
        records = []
        for idx, val in ma.dropna().items():
            records.append({"date": idx.strftime('%Y-%m-%d'), "value": float(val)})
        return {"symbol": symbol, "window": window, "result": records}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_RSI(symbol: str, window: int = 14, period: str = '6mo', interval: str = '1d') -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return {"symbol": symbol, "window": window, "result": []}
        close = df['Close']
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=window, min_periods=window).mean()
        ma_down = down.rolling(window=window, min_periods=window).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        records = []
        for idx, val in rsi.dropna().items():
            records.append({"date": idx.strftime('%Y-%m-%d'), "value": float(val)})
        return {"symbol": symbol, "window": window, "result": records}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_bollinger_bands(symbol: str, window: int = 20, period: str = '6mo', interval: str = '1d', num_std: float = 2.0) -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return {"symbol": symbol, "window": window, "result": []}
        close = df['Close']
        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        records = []
        for idx in ma.dropna().index:
            records.append({
                "date": idx.strftime('%Y-%m-%d'),
                "middle": float(ma.loc[idx]),
                "upper": float(upper.loc[idx]),
                "lower": float(lower.loc[idx]),
            })
        return {"symbol": symbol, "window": window, "result": records}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_dividends(symbol: str) -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        info = t.info if hasattr(t, 'info') else t.get_info()
        dividends = t.dividends
        latest_dividend = None
        if (dividends is not None) and (not dividends.empty):
            last_idx = dividends.index[-1]
            latest_dividend = {"date": last_idx.strftime('%Y-%m-%d'), "value": float(dividends.iloc[-1])}
        return {
            "symbol": symbol,
            "annual_dividend": info.get('dividendRate') or None,
            "trailing_yield": info.get('dividendYield') or None,
            "latest": latest_dividend,
        }
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_company_news(symbol: str, count: int = 5) -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        # yfinance has get_news()
        news = []
        if hasattr(t, 'get_news'):
            raw = t.get_news() or []
            for item in raw[:count]:
                news.append({
                    "title": item.get('title'),
                    "link": item.get('link'),
                    "publisher": item.get('publisher'),
                    "providerPublishTime": item.get('providerPublishTime'),
                })
        return {"symbol": symbol, "news": news}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}


def get_analyst_recommendations(symbol: str) -> Dict[str, Any]:
    try:
        t = _safe_ticker(symbol)
        rec = t.recommendations
        if rec is None or rec.empty:
            return {"symbol": symbol, "recommendations": []}
        rec = rec.reset_index()
        # group by period (year-month)
        rec['period'] = rec['Date'].dt.to_period('M').astype(str)
        grouped = rec.groupby('period')
        out = []
        for period, g in grouped:
            counts = g['To Grade'].value_counts().to_dict() if 'To Grade' in g else {}
            out.append({"period": period, "counts": counts})
        return {"symbol": symbol, "recommendations": out}
    except Exception:
        return {"symbol": symbol, "error": traceback.format_exc()}
