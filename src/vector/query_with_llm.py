import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import defaultdict
from ollama import chat
import concurrent.futures

# === Paths ===
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/vector/faiss.index")
EMBED_MODEL = "all-MiniLM-L6-v2"

# === Load Data ===
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

index = faiss.read_index(str(INDEX_PATH))
model = SentenceTransformer(EMBED_MODEL)

def embed(text):
    return model.encode([text])[0].astype("float32")

import re

def extract_used_citations(answer_text):
    return set(re.findall(r'\[(\d+)\]', answer_text))

def search_similar_chunks(query, top_k=8):
    vector = embed(query)
    D, I = index.search(np.array([vector]), top_k)
    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return results

def ask_llm(messages):
    response = chat(model="mistral", messages=messages)
    return response["message"]["content"]

def summarize_single_paper(paper_id, chunks, i):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    title = chunks[0].get("title", "Untitled")
    marker = f"[{i}]"
    prompt = f"""Summarize the following scientific content for the paper titled: '{title}':\n\n{context}"""
    summary = ask_llm([
        {"role": "system", "content": "You are an academic summarizer."},
        {"role": "user", "content": prompt}
    ])
    return paper_id, {"title": title, "summary": summary, "marker": marker}

def summarize_chunks(chunks_by_paper):
    summaries = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, (paper_id, chunks) in enumerate(chunks_by_paper.items(), start=1):
            futures.append(executor.submit(summarize_single_paper, paper_id, chunks, i))

        for future in concurrent.futures.as_completed(futures):
            paper_id, summary_info = future.result()
            summaries[paper_id] = summary_info
    return summaries

def generate_final_answer(query, paper_summaries):
    all_context = "\n\n".join(f"{info['marker']} {info['summary']}" for info in paper_summaries.values())
    prompt = f"""You are a scientific expert. Use the following summaries to answer the question with reasoning and inline citations like [1], [2], etc.

Summaries:
{all_context}

Question: {query}

Follow this format:
### Reasoning:
- Step 1:
- Step 2:
...
### Final Answer:
Make sure to include inline citations in the answer like [1], [2], etc. to show which paper supports which part of your reasoning.
"""
    return ask_llm([
        {"role": "system", "content": "You are a scientific assistant who answers with citations."},
        {"role": "user", "content": prompt}
    ])

def get_final_answer_from_query(query: str):
    top_chunks = search_similar_chunks(query)
    if not top_chunks:
        return "⚠️ No relevant chunks found.", []

    chunks_by_paper = defaultdict(list)
    for chunk in top_chunks:
        chunks_by_paper[chunk["paper_id"]].append(chunk)

    paper_summaries = summarize_chunks(chunks_by_paper)
    final_answer = generate_final_answer(query, paper_summaries)

    used_citation_numbers = extract_used_citations(final_answer)

    citations = []
    for i, (pid, info) in enumerate(paper_summaries.items(), start=1):
        if str(i) in used_citation_numbers:
            citations.append({
                "marker": f"[{i}]",
                "title": info["title"],
                "link": f"https://arxiv.org/abs/{pid}"
            })

    return final_answer.strip(), citations


if __name__ == "__main__":
    print("❗ This module is meant to be used via FastAPI UI. To run it in terminal mode, use the full main() function.")