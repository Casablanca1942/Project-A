# 本地财经助手：函数调用式智能分析系统设计方案

> 目标：让本地 LLM 助手能根据用户模糊提问，自动决定调用哪些 yfinance 函数、提取哪些数据，从而生成结构化的投资建议。

---

## 🎯 背景问题

传统对话系统只有在用户明确提出问题（如“请告诉我AAPL的PE”）时，才会去调用接口获取数据。但现实中：

- 用户提问往往模糊、开放，比如：“你觉得这家公司值不值得买？”
- AI 助手无法预知需要哪些数据，除非它能**理解意图 → 自动调用数据接口**。

---

## ✅ 总体解决方案：Function Calling + yfinance 封装

> "让模型先输出调用请求，再由程序执行，并将结果回传模型综合生成最终答案"

### 系统流程图：

```
[用户提问]
   ↓
[本地 LLM 理解意图]
   ↓
[输出函数调用 JSON 请求]
   ↓
[后端执行 yfinance 函数]
   ↓
[将结果作为 context 送回 LLM]
   ↓
[生成最终分析建议]
```

---

## 🧠 实现模块

### 1. 工具函数定义（Tool Schema）

用 JSON 格式定义模型可调用的函数列表（参考 OpenAI function calling 规范）：

```json
{
  "name": "get_stock_snapshot",
  "description": "获取股票快照，包括价格、市盈率、市值等",
  "parameters": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string", "description": "股票代码，如 AAPL"}
    },
    "required": ["symbol"]
  }
}
```

可以注册多个工具函数：

- `get_snapshot(symbol)` → 当前价、市值、PE 等
- `get_news(symbol, count)` → 最新新闻摘要
- `get_price_targets(symbol)` → 分析师目标价
- `get_dividends(symbol)` → 股息率
- `get_recommendations(symbol)` → 买入/持有/卖出评级

每个函数你只需要写对应的 Python/yfinance 封装。

---

### 2. 模型推理：生成调用请求

LLM 接收到提问后，输出如下结构：

```json
{
  "tool_call": {
    "name": "get_news",
    "parameters": {
      "symbol": "AAPL",
      "count": 5
    }
  }
}
```

如果需要多步信息：

```json
[
  {"tool_call": {"name": "get_snapshot", "parameters": {"symbol": "AAPL"}}},
  {"tool_call": {"name": "get_news", "parameters": {"symbol": "AAPL", "count": 3}}},
  {"tool_call": {"name": "get_price_targets", "parameters": {"symbol": "AAPL"}}}
]
```

你将解析这个 JSON，执行相应函数，将结果作为 `tool_response` 回传模型。

---

### 3. 后端函数执行

每个 tool 函数都封装对 yfinance 的调用，例如：

```python
def get_snapshot(symbol):
    t = yf.Ticker(symbol)
    info = t.info
    return {
        "price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap")
    }
```

再比如：

```python
def get_news(symbol, count=5):
    t = yf.Ticker(symbol)
    return t.get_news(count=count)
```

---

### 4. 模型合成回答（第二轮生成）

把这些数据打包入新的提示词，再送回模型，生成完整分析：

> “比亚迪当前市盈率为 23.7，略高于行业中位数，但分析师维持‘买入’评级，12个月目标价高于现价 15%。根据最新新闻，公司电池业务取得突破性订单，因此建议关注。”

---

## 🔧 模块职责总结

| 模块 | 作用 |
|------|------|
| 模型 | 解析用户意图，决定调用哪些工具、用什么参数 |
| 后端 | 注册工具函数，接收调用请求，执行 yfinance 操作 |
| 数据层 | 实际查询 yfinance 数据，返回结构化内容 |
| 回传层 | 将执行结果送回模型形成完整上下文 |
| 生成层 | 模型二次生成完整建议（支持结构化 JSON 输出） |

---

## ✅ 推荐技术选型

- LLM：Qwen2.5 / Llama3 / ChatGLM3（支持 function calling）
- 推理部署：Ollama / Transformers / LMDeploy
- 工具层：FastAPI + Python 函数注册 + JSON 验证
- 数据源：yfinance
- 前端：Streamlit（展示过程 + JSON 输出 + 新闻可视化）

---

## 📌 示例用户对话流程

用户提问：
> "你怎么看比亚迪？"

模型 → 生成 tool_call：
```json
[
  {"tool_call": {"name": "get_snapshot", "parameters": {"symbol": "002594.SZ"}}},
  {"tool_call": {"name": "get_news", "parameters": {"symbol": "002594.SZ", "count": 3}}},
  {"tool_call": {"name": "get_price_targets", "parameters": {"symbol": "002594.SZ"}}}
]
```

后端执行 → 拿到数据 → 模型生成完整回答：

> "比亚迪当前 PE 为 21.5，分析师平均目标价为现价的 120%。新闻显示其新能源汽车销量连续增长，建议关注，评级：增持。"

---

## ✅ 下一步行动

你可以开始：

1. 编写 `tool_schema.json` 注册所有 yfinance 工具函数
2. 实现对应 Python 函数（调用 yfinance + 错误处理）
3. 接入支持 Tool Calling 的 LLM（如 Ollama + Qwen2.5）
4. 将模型输出的 tool_call JSON 交由后端调度器执行
5. 将执行结果送回模型，完成整体链条

如需我生成工具注册框架与样例代码，可随时告诉我 ✅

