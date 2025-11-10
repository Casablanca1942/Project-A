from pathlib import Path

def split_text(text_path: str, chunk_size=500, chunk_overlap=50):
    text = Path(text_path).read_text(encoding="utf-8")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap
    print(f"[✔] 共生成 {len(chunks)} 个 chunks")
    return chunks
