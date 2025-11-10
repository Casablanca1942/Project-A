import chromadb
from sentence_transformers import SentenceTransformer
from split_text_chunks import split_text
from pathlib import Path
from tqdm import tqdm

# Step 1: 加载文本块
chunks = split_text("TII.txt", chunk_size=500, chunk_overlap=50)

# Step 2: 初始化嵌入模型（可换成中文模型如 bge-small-zh-v1.5）
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 英文适用

# Step 3: 设置 Chroma 向量数据库，设定持久化目录
CHROMA_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("investor_book")

# Step 4: 生成向量并入库
ids = [f"chunk_{i}" for i in range(len(chunks))]
embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# 每次最多添加 100 条（避免内存过大）
# 修复后的 batch 写入逻辑
batch_size = 100
for i in tqdm(range(0, len(chunks), batch_size)):
    chunk_batch = chunks[i:i+batch_size]
    embedding_batch = embeddings[i:i+batch_size]
    id_batch = ids[i:i+batch_size]
    metadata_batch = [{"source": "investor.docx"}] * len(chunk_batch)

    collection.add(
        documents=chunk_batch,
        embeddings=embedding_batch,
        metadatas=metadata_batch,
        ids=id_batch
    )


print(f"[✔] 向量库已保存至 {CHROMA_DIR}")
