import arxiv
import json
import requests
from pathlib import Path

# Proje kök dizinini hesapla (src/ingest/fetch_arxiv.py dosyasına göre 3 üst klasör)
ROOT_DIR = Path("/Users/iremsusavas/Desktop/llm")


# Data dizinleri
DATA_DIR = ROOT_DIR / "data" / "raw"
PDF_DIR = DATA_DIR / "pdfs"
META_FILE = DATA_DIR / "metadata.json"

# ArXiv ayarları
CATEGORIES = ["cs.RO", "cs.AI"]
MAX_RESULTS = 10  # test için 10 tut

# Dizinleri oluştur
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_save():
    all_meta = []
    client = arxiv.Client()

    
    for cat in CATEGORIES:
        print(f"[INFO] {cat} kategorisinden makaleler çekiliyor...")
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        for result in client.results(search):
            meta = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.isoformat(),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            }

            pdf_path = PDF_DIR / f"{meta['id']}.pdf"
            if not pdf_path.exists():
                try:
                    print(f"[INFO] PDF indiriliyor: {result.pdf_url}")
                    r = requests.get(result.pdf_url, timeout=15)
                    r.raise_for_status()
                    with open(pdf_path, "wb") as f:
                        f.write(r.content)
                    print(f"[OK] {pdf_path.name} indirildi.")
                except Exception as e:
                    print(f"[HATA] PDF indirilemedi ({meta['id']}): {e}")
            else:
                print(f"[SKIP] PDF zaten var: {pdf_path.name}")
            
            meta["pdf_path"] = str(pdf_path)
            all_meta.append(meta)

    # Metadata yaz
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    
    print(f"[DONE] Toplam {len(all_meta)} makale metadata olarak kaydedildi → {META_FILE}")

if __name__ == "__main__":
    fetch_and_save()
