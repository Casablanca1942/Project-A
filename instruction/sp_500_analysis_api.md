# 标普500成分股分析函数设计文档

本文档设计了一套基于[yfinance](https://ranaroussi.github.io/yfinance/reference/)库的股票分析函数，适用于本地 LLM 分析标普500成分股。

## 公用入参
- `symbol` (string): 美股标码，如 `AAPL` ，`MSFT`
- `period` (string): 历史周期，如 `1mo`, `6mo`, `1y`, `5y`
- `interval` (string): 周期频率，如 `1d`, `1wk`, `1mo`
- `window` (int): 技术指标窗口，如 14, 20

## 一、长线价值投资

### get_pe_ratio
获取 PE 指标
```json
{
  "symbol": "AAPL",
  "forward_pe": 22.8,
  "trailing_pe": 28.4
}
```

### get_pb_ratio
获取 PB 指标
```json
{
  "symbol": "AAPL",
  "price_to_book": 42.58
}
```

### get_dividend_yield
获取股息率
```json
{
  "symbol": "AAPL",
  "annual_dividend": 0.92,
  "trailing_yield": 0.6
}
```

### get_roe
获取 ROE
```json
{
  "symbol": "AAPL",
  "return_on_equity": 147.94
}
```

### get_profit_growth_rate
获取纯利增长率
```json
{
  "symbol": "AAPL",
  "latest_year": 2023,
  "previous_year": 2022,
  "net_income_latest": 95.0,
  "net_income_previous": 94.7,
  "growth_rate": 0.32
}
```

### get_financials
获取整体资料包括 income/balance/cashflow
```json
{
  "symbol": "AAPL",
  "income_statement": {"totalRevenue": 365, "netIncome": 95},
  "balance_sheet": {"totalAssets": 350, "totalLiab": 280},
  "cash_flow": {"operatingCashFlow": 122}
}
```

### get_income_statement / get_balance_sheet / get_cash_flow
根据 period = annual/quarterly 分别获取损益、资产负债、现金流量主要指标（统一格式结构。结果同 get_financials 中的字段。

---

## 二、短线分析

### get_company_news
获取最新公司新闻
```json
{
  "symbol": "AAPL",
  "news": [
    {
      "title": "...",
      "link": "...",
      "publisher": "...",
      "pubTime": 1680000000
    }
  ]
}
```

### get_analyst_recommendations
分析师评级分布
```json
{
  "symbol": "AAPL",
  "recommendations": [
    {
      "period": "2023-11",
      "strongBuy": 10,
      "buy": 12,
      "hold": 5,
      "sell": 0,
      "strongSell": 0
    }
  ]
}
```

### get_upgrades_downgrades
分析师评级调整历史
```json
{
  "symbol": "AAPL",
  "changes": [
    {
      "date": "2024-08-01",
      "firm": "Goldman Sachs",
      "fromGrade": "Hold",
      "toGrade": "Buy",
      "action": "upgrade"
    }
  ]
}
```

### get_earnings_calendar
资本日历
```json
{
  "symbol": "AAPL",
  "next_earnings_date": "2024-11-02",
  "last_earnings_date": "2024-08-02",
  "dividend_date": "2024-05-09"
}
```

### get_earnings_estimate / get_revenue_estimate
未来两年总收益 / 营收预期
```json
{
  "symbol": "AAPL",
  "current_qtr_eps_est": 1.20,
  "next_qtr_eps_est": 1.35
}
```

---

## 三、技术分析

### get_current_price
获取当前价
```json
{
  "symbol": "AAPL",
  "last_price": 172.35,
  "previous_close": 170.50
}
```

### get_price_history
历史行情
```json
{
  "symbol": "AAPL",
  "prices": [
    {
      "date": "2024-10-01",
      "open": 172.0,
      "high": 173.5,
      "low": 170.8,
      "close": 172.3,
      "volume": 35000000
    }
  ]
}
```

### get_moving_average / get_RSI / get_bollinger_bands / get_volume
统一格式：
```json
{
  "symbol": "AAPL",
  "window": 20,
  "result": [
    {
      "date": "2024-10-01",
      "value": 169.8
    }
  ]
}
```
> 当中 value 为 MA 值，或 RSI 值，或上下轨等。

---

## 后续扩展
- 加入 get_symbol_from_name 基于公司名查找 ticker
- 方便 LLM 把 "请分析英特尔" 转成 symbol=`INTC`
- 每个函数都应选择给出请求时间和周期，保持统一设计

---

本文档可直接注册为 function-calling 工具系统，适用于 Llama 3 / ChatGLM3 / Qwen2.5 等支持工具调用的本地分析系统。

