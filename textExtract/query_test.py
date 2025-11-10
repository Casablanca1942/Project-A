import chromadb
from sentence_transformers import SentenceTransformer

query = "What is margin of safety in investing?"

CHROMA_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("investor_book")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_vec = embedder.encode([query], convert_to_numpy=True)

results = collection.query(query_embeddings=query_vec, n_results=3)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- 结果{i+1} ---\n{doc}")
