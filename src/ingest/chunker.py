import json
from pathlib import Path
from PyPDF2 import PdfReader
import re

# Proje kök dizini
ROOT_DIR = Path("/Users/iremsusavas/Desktop/llm")

# Girdi ve çıktı dizinleri
META_PATH = ROOT_DIR / "data" / "raw" / "metadata.json"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000  # karakter
OVERLAP = 200      # karakter

def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        print(f"[HATA] PDF okunamadı: {pdf_path.name}: {e}")
        return ""

def chunk_papers():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        total_chunks = 0
        for paper in metadata:
            pdf_path = Path(paper["pdf_path"])
            text = extract_text_from_pdf(pdf_path)
            if not text:
                continue
            chunks = split_text(text)
            for i, chunk in enumerate(chunks):
                record = {
                "id": f"{paper['id']}_chunk{i}",
                "paper_id": paper["id"],
                "title": paper["title"],
                "text": chunk   # ← burada "chunk" yerine "text" yaz
            }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
            print(f"[OK] {paper['id']} → {len(chunks)} parça")

    print(f"[DONE] Toplam {total_chunks} metin parçası üretildi → {OUTPUT_PATH}")

if __name__ == "__main__":
    chunk_papers()
