from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown2 import markdown
import uvicorn

# Markdown destekli özel Jinja2 ortamı
env = Environment(
    loader=FileSystemLoader("src/web/templates"),
    autoescape=select_autoescape(["html", "xml"])
)
env.filters['markdown'] = lambda text: markdown(text)

# Templates sınıfına özel ortamı veriyoruz
templates = Jinja2Templates(directory="src/web/templates")
templates.env = env  # ✅ doğru kullanım

# App tanımı
app = FastAPI()

# Sorgu fonksiyonunu içe aktar
from src.vector.query_with_llm import get_final_answer_from_query

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/", response_class=HTMLResponse)
def handle_query(request: Request, query: str = Form(...)):
    answer, citations = get_final_answer_from_query(query)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": query,
        "answer": answer,
        "citations": citations
    })

if __name__ == "__main__":
    uvicorn.run("src.web.main:app", host="127.0.0.1", port=8000, reload=True)
