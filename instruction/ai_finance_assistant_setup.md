# 本地 AI 财经助手搭建教程

## 环境准备

### 硬件与系统
- NVIDIA RTX 4060Ti 显卡（建议 8GB 及以上）
- Windows 10/11 或 Linux
- 建议内存 ≥16GB

### 安装依赖
```bash
# 创建虚拟环境（可选）
conda create -n finance python=3.10
conda activate finance

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装必要库
pip install transformers sentence-transformers chromadb yfinance requests beautifulsoup4 streamlit
```

### 检查 GPU 可用性
```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 正常
```

---

## 模型部署

### 本地运行大模型
- 推荐模型：
  - `Qwen2.5-Coder-7B-Instruct`
  - `Llama3-8B-Instruct`
- 可通过 HuggingFace Transformers 或 Ollama 启动

#### 示例（HuggingFace 加载 Qwen）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
```

---

## 构建知识库（向量检索）

### 步骤：
```python
from sentence_transformers import SentenceTransformer
import chromadb

# 加载嵌入模型（中文）
embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 初始化 Chroma 向量库
client = chromadb.Client()
collection = client.create_collection(name="finance_kb")

# 文档与向量写入
docs = ["股市中长期投资原则：...", "经济学原理-第一章:..."]
embeddings = embed_model.encode(docs, convert_to_numpy=True)
collection.add(documents=docs, embeddings=embeddings, ids=["doc1", "doc2"])
```

---

## 抓取财经数据与新闻

### 使用 yfinance 获取行情数据
```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="5d")
current_price = hist["Close"].iloc[-1]
```

### 获取分析师评级与新闻
```python
rec = ticker.get_recommendations()
news_items = ticker.get_news(count=5)
```

---

## 整合对话与分析逻辑（RAG）

### 示例流程
```python
def generate_answer(user_query):
    # 1. 从向量库检索相关知识
    results = collection.query(query_texts=[user_query], n_results=3)
    context = "\n".join(results['documents'][0])

    # 2. 获取实时行情与新闻
    stock_data = get_stock_data("AAPL")
    news = get_news("AAPL")

    # 3. 构造 Prompt
    prompt = f"""
    你是一位金融投资顾问，请根据以下信息提供分析建议，输出JSON：
    知识：{context}
    行情：{stock_data}
    新闻：{news}
    问题：{user_query}
    """

    # 4. 模型生成
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=500)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## 构建网页交互界面（Streamlit）

### app.py 示例
```python
import streamlit as st
st.title("AI 本地财经助手")
user_query = st.text_input("请输入您的问题：")
if st.button("提交"):
    response = generate_answer(user_query)
    st.json(response)
```

### 启动应用
```bash
streamlit run app.py
```

---

## 调试与优化

- 使用 VS Code 调试 `app.py`、检索逻辑、模型输出格式
- 监控 GPU 使用情况 `nvidia-smi`
- 若显存不足，尝试使用量化模型（8bit / 4bit）
- 输出结构建议使用 JSON Schema 校验

---

## 项目结构建议
```
finance_ai/
├── app.py                  # Streamlit 前端
├── model.py                # 模型加载逻辑
├── retriever.py            # 知识库构建/查询
├── data/                   # 经济学文档目录
├── chroma_db/              # 向量数据库持久化
├── news/                   # 抓取到的财经新闻存档
```

---

## 后续建议

- 支持多市场（美股/港股/ETF）行情
- 接入宏观指标（如 FRED）API
- 增加 JSON 输出字段：风险评级、估值分数、预期回报等
- 日常自动更新数据：使用 APScheduler 定时刷新行情
- 使用 FastAPI + Vue 构建专业前后端界面

---

如需进一步集成自动分析、函数调用（如计算 RSI、MACD）、图表展示，或将助手部署为本地 API，请另行扩展。