import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNKS_FILE = Path("../../data/processed/chunks.jsonl")
INDEX_DIR = Path("../../data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
MAPPING_PATH = INDEX_DIR / "chunk_mapping.json"

def load_chunks():
    chunks = []
    for line in open(CHUNKS_FILE, encoding="utf-8"):
        chunks.append(json.loads(line))
    return chunks

def build():
    model = SentenceTransformer("all-MiniLM-L6-v2")  # hızlı, güçlü
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize is optional depending on retrieval metric
    # faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine benzeri için inner product, normalize edersen
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    # Mapping sakla
    minimal = [
        {
            "source_id": c["source_id"],
            "title": c["title"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "metadata": c["metadata"],
        }
        for c in chunks
    ]
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(minimal, f, ensure_ascii=False, indent=2)
    print(f"Index oluşturuldu, {len(texts)} vektör eklendi.")

if __name__ == "__main__":
    build()
