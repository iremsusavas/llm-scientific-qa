import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_PATH = ROOT_DIR / "data" / "vector" / "faiss.index"
META_PATH = ROOT_DIR / "data" / "vector" / "metadata.pkl"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read all chunks
chunks = []
metadatas = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        chunks.append(data["text"])  # 🔄 updated from "chunk" → "text"
        metadatas.append({
            "id": data["id"],
            "paper_id": data["paper_id"],
            "title": data["title"]
        })

print(f"[INFO] {len(chunks)} text chunks found. Generating embeddings...")

# Compute embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_PATH))

import pickle
with open(META_PATH, "wb") as f:
    pickle.dump(metadatas, f)

print(f"[DONE] FAISS index and metadata saved: {INDEX_PATH}")
