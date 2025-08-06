import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("../../data/index/faiss_index.bin")
MAPPING_PATH = Path("../../data/index/chunk_mapping.json")

def query_loop():
    index = faiss.read_index(str(INDEX_PATH))
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    while True:
        q = input("\nSoru (çıkmak için 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        q_emb = model.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, k=5)
        for rank, idx in enumerate(I[0]):
            chunk = mapping[idx]
            print(f"\n--- Sonuç {rank+1} ---")
            print(f"Başlık: {chunk['title']}")
            print(f"Chunk idx: {chunk['chunk_index']}")
            print(f"Metin örneği (ilk 300 karakter): {chunk['text'][:300].replace(chr(10), ' ')}")
            print(f"Benzerlik skoru: {D[0][rank]:.4f}")

if __name__ == "__main__":
    query_loop()
