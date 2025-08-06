import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

# Yol ayarları
ROOT_DIR = Path("/Users/iremsusavas/Desktop/llm")
INDEX_PATH = ROOT_DIR / "data" / "vector" / "faiss.index"
META_PATH = ROOT_DIR / "data" / "vector" / "chunk_meta.pkl"

# Yüklemeler
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    metadatas = pickle.load(f)

def search(query, top_k=5):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), top_k)

    print(f"\n🔍 Top {top_k} sonuç:")
    for rank, idx in enumerate(I[0]):
        meta = metadatas[idx]
        print(f"\n#{rank+1}")
        print(f"📘 Title: {meta['title']}")
        print(f"📄 Paper ID: {meta['paper_id']}")
        print(f"🔢 Chunk ID: {meta['id']}")

if __name__ == "__main__":
    while True:
        q = input("\n💬 Soru gir (çıkmak için 'q'): ")
        if q.lower() == 'q':
            break
        search(q)
