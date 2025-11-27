"""演示脚本：连接向量数据库（Chroma）、financeAnalysis 的 yfinance 封装、以及可选的模型生成。

该脚本演示了 generate_answer 的完整流程：
- 从本地 Chroma 向量库（textExtract/chroma_db -> collection 'investor_book'）检索相关知识
- 使用 financeAnalysis 中的封装函数获取行情快照与新闻
- 构建提示词并优先调用本地 Ollama（或其它模型）生成结论
- 若缺少模型或外部依赖，则回退到确定性（non-LLM）生成逻辑，保证可运行性

用法示例:
    python generate_demo.py --query "What do you think about Apple?" --symbol AAPL

选项说明:
    --model-name  可选：指定一个 Transformers 模型名称用于生成；若不指定则使用回退机制或本地 Ollama。

注意事项:
    请先确保已在 textExtract 目录下用 build_vector_db.py 创建了向量库（textExtract/chroma_db）。
"""

import argparse
import json
import os
import sys
from typing import Optional

# Directory for debug outputs (raw model replies and final replies)
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'debugOutput')

# 本地模块（可选）- 如果缺少依赖则优雅降级，保证 demo 在精简环境中也能运行
have_yf_tools = True
try:
    from financeAnalysis.yf_tools import get_snapshot, get_company_news, get_RSI, get_price_history
except Exception:
    have_yf_tools = False
    print("[!] financeAnalysis.yf_tools not available. The demo will use lightweight dummy stock/news responses.")

    def get_snapshot(symbol: str):
        # 最小快照数据（保证后续流程可运行）
        return {
            "symbol": symbol,
            "last_price": 150.0,
            "previous_close": 149.0,
            "market_cap": None,
            "trailing_pe": None,
            "forward_pe": None,
            "timestamp": None,
        }

    def get_company_news(symbol: str, count: int = 3):
        return {
            "symbol": symbol,
            "news": [
                {"title": f"{symbol}: quarterly results beat expectations", "link": "", "publisher": "DemoNews", "providerPublishTime": None},
                {"title": f"{symbol}: new product announcement", "link": "", "publisher": "DemoNews", "providerPublishTime": None},
            ][:count]
        }

    def get_RSI(symbol: str, window: int = 14, period: str = '6mo', interval: str = '1d'):
        # 简单的 RSI 回退实现，保证离线/测试环境可用
        return {"symbol": symbol, "window": window, "result": [{"date": "2025-01-01", "value": 55.0}]}

    def get_price_history(symbol: str, period: str = '1y', interval: str = '1d'):
        # 简单的历史价格回退数据，供演示使用
        return {"symbol": symbol, "prices": [{"date": "2025-11-25", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000000}]}

# Chroma 向量数据库与嵌入模型
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception:
    chromadb = None

# 优先使用本地 Ollama (llama3) 作为文本生成器；若不可用则使用轻量级的回退方案
try:
    import requests
except Exception:
    requests = None


def query_vector_db(query: str, client_path: str = "textExtract/chroma_db", collection_name: str = "investor_book", n_results: int = 3):
    """查询持久化的 Chroma 向量库并返回检索到的 top 文档文本列表。
    返回值：list[str]（文档列表），若不可用则返回空列表。
    """
    if chromadb is None:
        print("[!] chromadb or sentence-transformers not installed. Skipping vector lookup.")
        return []

    if not os.path.exists(client_path):
        print(f"[!] Chroma DB path not found: {client_path}. Did you run textExtract/build_vector_db.py ?")
        return []

    client = chromadb.PersistentClient(path=client_path)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        # 尝试获取或创建集合（兼容不同 chroma 版本）
        collection = client.get_or_create_collection(collection_name)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qvec = embedder.encode([query], convert_to_numpy=True)

    results = collection.query(query_embeddings=qvec, n_results=n_results)
    docs = results.get("documents", [[]])[0]
    return docs


def build_prompt(context: str, stock_data: dict, news: list, user_query: str) -> str:
    # 请求模型返回可读的纯文本结论（不要 JSON）
    return f"""
You are a professional financial analyst. Based on the following inputs prepare a concise, plain-text analysis.
Context: {context}
Snapshot: {json.dumps(stock_data, ensure_ascii=False)}
News: {json.dumps(news, ensure_ascii=False)}
Question: {user_query}

Please output a multi-line plain text answer including: recommendation (buy/hold/sell), confidence (0-100), and a brief explanation. Do not output JSON or other wrappers.
"""


def ollama_generate(prompt: str, model: str = "llama3", max_tokens: int = 500, ollama_url: Optional[str] = None):
    """向本地 Ollama REST API 发起单次请求并返回生成的文本。

    - 默认访问 http://localhost:11434。
    - 使用最简化的 payload，以兼容常见 Ollama 配置。
    - 若返回不可解析为 JSON 的内容，则直接返回原始文本。
    """
    if requests is None:
        return None

    url = ollama_url or os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate"
    # 请求非流式响应，避免收到按 token 切片的 NDJSON 片段
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        # 优先尝试解析 JSON，解析失败则返回原始文本
        try:
            j = r.json()
            # 常见的返回字段有 `text` 或 `result`，否则把整个对象字符串化
            if isinstance(j, dict):
                # Ollama 的返回可能有多种形式，做一些兼容处理：
                # 1) {"text": "..."}
                # 2) {"generated":[{"role":"assistant","content":"..."}], ...}
                # 3) 嵌套 result 字段
                text = j.get("text") or j.get("result")
                if not text and "generated" in j and isinstance(j["generated"], list) and len(j["generated"])>0:
                    gen = j["generated"][0]
                    if isinstance(gen, dict):
                        # 有些实现把文本放在 content 或 text 字段里
                        text = gen.get("content") or gen.get("text")
                    else:
                        text = str(gen)

                if text:
                    return text
                # 如果找不到文本字段，就返回 JSON 字符串
                return json.dumps(j, ensure_ascii=False)
            return str(j)
        except Exception:
            return r.text
    except Exception as e:
        print("[!] Ollama generation failed:", e)
        return None


def decide_tools_with_model(user_query: str, symbol: str, ollama_model: str = "llama3", ollama_url: Optional[str] = None) -> Optional[list]:
    """让模型判断需要调用哪些工具并返回工具调用列表（tool_call）。若失败返回 None。

    要求模型仅返回形如：
        {"tool_call": [{"name":"get_snapshot","parameters":{"symbol":"AAPL"}}, ...]}

    若模型返回流式 tokens 或其它文本格式，本函数会尝试抽取有效的 JSON 子串并解析。
    """
    tool_schema_text = (
        "Available tools:\n"
        "1) get_snapshot(symbol: string) -> basic snapshot (price, pe, market cap).\n"
        "2) get_company_news(symbol: string, count: int) -> latest news headlines.\n"
        "3) get_RSI(symbol: string, window: int) -> RSI series.\n"
        "4) get_price_history(symbol: string, period: string, interval: string) -> price history.\n"
    )

    prompt = (
        f"Decide which tools should be called to answer the user's question.\n"
        f"Return exactly a JSON object and nothing else in this format:\n"
        f"{{\"tool_call\": [ {{\"name\": <tool-name>, \"parameters\": {{...}} }} ] }}\n\n"
        f"{tool_schema_text}\n"
        f"User question: {user_query}\n"
        f"Target symbol: {symbol}\n"
    )

    reply = ollama_generate(prompt, model=ollama_model, max_tokens=200, ollama_url=ollama_url)
    if not reply:
        return None
    print("[DEBUG] raw model reply:")
    print(reply)
    # ensure debug output directory exists and write raw reply there
    try:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        raw_path = os.path.join(DEBUG_OUTPUT_DIR, 'debug_raw_reply.txt')
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(reply)
    except Exception as _e:
        # If we cannot write to debug dir, fall back to writing to the CWD
        try:
            with open("debug_raw_reply.txt", "w", encoding="utf-8") as f:
                f.write(reply)
        except Exception:
            pass
    # The model (or back-end API) may return different shapes:
    # - a plain JSON string representing the tool_call dict
    # - a top-level JSON object that includes a `response` / `text` / `result` field that contains the JSON
    # - the JSON encoded inside markdown code fences ```{...}```
    # - `generated` arrays with nested content strings
    # We'll attempt several extraction strategies to find a JSON object that contains 'tool_call'.

    def try_parse_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def extract_from_code_fence(s: str) -> Optional[str]:
        # find triple-backtick fences and return first JSON-looking block inside
        start = s.find('```')
        if start == -1:
            return None
        end = s.find('```', start + 3)
        if end == -1:
            # if only single fence found, try to take rest of string
            content = s[start + 3 :]
        else:
            content = s[start + 3 : end]
        return content.strip()

    def extract_json_substr_with_keyword(s: str, keyword: str = '"tool_call"') -> Optional[str]:
        idx = s.find(keyword)
        if idx == -1:
            return None
        # Find left brace before the keyword
        left = s.rfind('{', 0, idx)
        if left == -1:
            left = s.find('{')
            if left == -1:
                return None
        # Find matching right brace by scanning
        depth = 0
        for i in range(left, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return s[left:i+1]
        return None

    obj = try_parse_json(reply)
    # If top-level JSON is a dict, try to find the tool_call inside nested fields
    if isinstance(obj, dict):
        if 'tool_call' in obj:
            return obj['tool_call']

        # Check common payload locations
        candidates = []
        for k in ('response', 'text', 'result'):
            if k in obj and isinstance(obj[k], str):
                candidates.append(obj[k])

        # generated -> list of dicts
        if 'generated' in obj and isinstance(obj['generated'], list):
            for gen in obj['generated']:
                if isinstance(gen, dict):
                    if 'content' in gen and isinstance(gen['content'], str):
                        candidates.append(gen['content'])
                    if 'text' in gen and isinstance(gen['text'], str):
                        candidates.append(gen['text'])
                elif isinstance(gen, str):
                    candidates.append(gen)

        # Try candidates: extract code fences then raw JSON substring
        for c in candidates:
            # try code fence extraction first
            fenced = extract_from_code_fence(c)
            if fenced:
                parsed = try_parse_json(fenced)
                if isinstance(parsed, dict) and 'tool_call' in parsed:
                    return parsed['tool_call']
                # maybe JSON is embedded inside fenced text
                sub = extract_json_substr_with_keyword(fenced)
                if sub:
                    parsed = try_parse_json(sub)
                    if isinstance(parsed, dict) and 'tool_call' in parsed:
                        return parsed['tool_call']

            # no code fence or fence failed -> try direct substring search for tool_call JSON
            sub = extract_json_substr_with_keyword(c)
            if sub:
                parsed = try_parse_json(sub)
                if isinstance(parsed, dict) and 'tool_call' in parsed:
                    return parsed['tool_call']

    # If initial json.loads failed or no nested object found, attempt to pull JSON from raw text
    # check for a code-fence block
    fenced = extract_from_code_fence(reply)
    if fenced:
        parsed = try_parse_json(fenced)
        if isinstance(parsed, dict) and 'tool_call' in parsed:
            return parsed['tool_call']
        sub = extract_json_substr_with_keyword(fenced)
        if sub:
            parsed = try_parse_json(sub)
            if isinstance(parsed, dict) and 'tool_call' in parsed:
                return parsed['tool_call']

    # final fallback: try to find the curly-brace JSON substring that contains tool_call
    sub = extract_json_substr_with_keyword(reply)
    if sub:
        parsed = try_parse_json(sub)
        if isinstance(parsed, dict) and 'tool_call' in parsed:
            return parsed['tool_call']

    if not isinstance(obj, dict) or 'tool_call' not in obj:
        return None
    return obj['tool_call']


def execute_tool_call(tool_call: dict):
    """执行单个工具调用（tool_call）并返回结果字典。

    支持的工具名会映射到本地 financeAnalysis 的函数；成功返回工具结果 dict，失败返回包含 error 字段的 dict。
    """
    name = tool_call.get('name')
    params = tool_call.get('parameters', {}) or {}

    try:
        if name == 'get_snapshot':
            return get_snapshot(params.get('symbol'))
        if name == 'get_company_news':
            return get_company_news(params.get('symbol'), count=params.get('count', 3))
        if name == 'get_RSI':
            return get_RSI(params.get('symbol'), window=params.get('window', 14))
        if name == 'get_price_history':
            return get_price_history(params.get('symbol'), period=params.get('period', '1y'), interval=params.get('interval', '1d'))
    except Exception as e:
        return {"error": str(e)}

    return {"error": f"unsupported tool: {name}"}


def fallback_generator(context: str, stock_data: dict, news: list, user_query: str) -> str:
    """基于输入生成确定性的纯文本答案用于测试（无 LLM）。"""
    # 简单启发式规则
    price = stock_data.get("last_price") if isinstance(stock_data, dict) else None
    pe = stock_data.get("trailing_pe") if isinstance(stock_data, dict) else None

    # 基于新闻标题的情绪判断
    news_titles = "\n".join([item.get('title', '') for item in news[:3]])
    if any(k in news_titles.lower() for k in ["beat", "upgrade", "record", "raise", "growth"]):
        rating = "buy"
        confidence = 78
    elif any(k in news_titles.lower() for k in ["downgrade", "miss", "restruct", "lawsuit", "concern"]):
        rating = "sell"
        confidence = 62
    else:
        rating = "hold"
        confidence = 55

    explanation = (
        f"Based on retrieved context ({len(context.split())} items), current price={price}, trailing PE={pe}. Recent news: {news_titles[:240]}..."
    )

    # 格式化为可读的纯文本（多行）
    lines = []
    lines.append(f"Recommendation: {rating.upper()}")
    lines.append(f"Confidence: {confidence}%")
    lines.append(f"Explanation: {explanation}")
    lines.append("")
    lines.append("--- Details ---")
    lines.append(f"Query: {user_query}")
    # 包含简短的上下文片段
    snippets = context.split("\n\n")[:3]
    if snippets and snippets[0] != "(no context found)":
        lines.append("Context snippets:")
        for s in snippets:
            lines.append(f" - {s[:220].strip().replace('\n', ' ')}")
    lines.append("Stock snapshot:")
    lines.append(f" - symbol: {stock_data.get('symbol')}")
    lines.append(f" - last_price: {stock_data.get('last_price')}")
    lines.append(f" - trailing_pe: {stock_data.get('trailing_pe')}")
    if news:
        lines.append("Top news headlines:")
        for item in news[:3]:
            title = item.get('title') or item.get('link') or "(no title)"
            lines.append(f" - {title}")

    return "\n".join(lines)


def generate_answer(user_query: str, symbol: str, use_model_name: Optional[str] = None) -> str:
    # 1. 从向量数据库检索
    docs = query_vector_db(user_query)
    context = "\n\n".join(docs) if docs else "(no context found)"

    # 2. 获取行情快照与相关新闻
    stock_data = get_snapshot(symbol)
    news_data = get_company_news(symbol).get('news', []) if isinstance(get_company_news(symbol), dict) else []

    # 3. 使用模型决定需要调用哪些工具（function-calling）
    model_name = use_model_name or os.environ.get("OLLAMA_MODEL") or "llama3"
    ollama_url = os.environ.get("OLLAMA_URL")

    print(f"[*] Asking model to decide tools (model={model_name})")
    tool_calls = decide_tools_with_model(user_query, symbol, ollama_model=model_name, ollama_url=ollama_url)

    tool_outputs = []
    if tool_calls:
        print(f"[*] Model requested {len(tool_calls)} tools: {tool_calls}")
        for t in tool_calls:
            print(f"[*] Executing tool: {t}")
            res = execute_tool_call(t)
            tool_outputs.append({"tool": t, "result": res})
    else:
        print("[!] Model did not provide valid tool_call JSON, falling back to default snapshot+news calls")
        # 使用默认的快速调用作为降级策略
        tool_outputs.append({"tool": {"name": "get_snapshot", "parameters": {"symbol": symbol}}, "result": stock_data})
        tool_outputs.append({"tool": {"name": "get_company_news", "parameters": {"symbol": symbol, "count": 3}}, "result": {"news": news_data}})

    # 4. 将工具输出拼接进最终提示词，向模型请求纯文本答案
    tools_context = json.dumps(tool_outputs, ensure_ascii=False)
    final_prompt = (
        "The model has retrieved the following tool outputs. Use them together with the knowledge context to prepare a concise plain-text analysis.\n\n"
        f"Knowledge context:\n{context}\n\n"
        f"Tool outputs:\n{tools_context}\n\n"
        f"User question: {user_query}\n\n"
        "Please produce a short multi-line plain text answer including: recommendation (buy/hold/sell), confidence (0-100), and a brief explanation. Do not output JSON or code blocks."
    )

    print(f"[*] Generating final answer using model {model_name}")
    final = ollama_generate(final_prompt, model=model_name, max_tokens=400, ollama_url=ollama_url)

    if final:
        return final

    # 回退：确定性生成答案（无模型）
    return fallback_generator(context, stock_data, news_data, user_query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo generate_answer flow: vectors -> finance -> model/fallback")
    parser.add_argument("--query", type=str, required=True, help="User natural language query")
    parser.add_argument("--symbol", type=str, required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--model-name", type=str, default=None, help="(optional) HF model name to use")
    args = parser.parse_args()

    out = generate_answer(args.query, args.symbol, use_model_name=args.model_name)
    # Write the full (raw) final response to a debug file for later inspection
    try:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        final_path = os.path.join(DEBUG_OUTPUT_DIR, 'debug_final_reply.txt')
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))
    except Exception as e:
        print(f"[!] Failed to write debug_final_reply.txt: {e}", file=sys.stderr)

    # Per caller request: print only the final response text
    if isinstance(out, str):
        print(out)
    else:
        # if the final result is a JSON-serialized string, print a compact representation
        print(json.dumps(out, ensure_ascii=False))
