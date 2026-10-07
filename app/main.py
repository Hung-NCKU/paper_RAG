import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .rag import Rag, RagError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
pdf_path = Path(os.getenv('PDF_PATH', 'data/pdfs/paper.pdf'))
if not pdf_path.is_absolute():
    pdf_path = ROOT / pdf_path
rag = Rag(pdf_path, ROOT / 'data/index/index.json',
          os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
          os.getenv('LLM_MODEL', 'qwen3:4b'), os.getenv('EMBED_MODEL', 'embeddinggemma'))

app = FastAPI(title='Paper RAG', version='1.0.0')
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=8)


@app.get('/')
def home():
    return FileResponse(ROOT / 'app/static/index.html')


@app.get('/api/status')
def status():
    return rag.status()


@app.post('/api/index')
def index():
    try:
        rag.build()
        return rag.status()
    except RagError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/upload')
async def upload(file: UploadFile):
    try:
        data = await file.read(30 * 1024 * 1024 + 1)
        if len(data) > 30 * 1024 * 1024:
            raise HTTPException(413, 'PDF 不可超過 30 MB。')
        await run_in_threadpool(rag.build, data)
        return rag.status()
    except RagError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await file.close()


@app.post('/api/chat')
def chat(request: Question):
    if not request.question.strip():
        raise HTTPException(422, '問題不可空白。')
    try:
        return rag.ask(request.question.strip(), request.top_k)
    except RagError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get('/api/pdf')
def pdf():
    if not rag.pdf.is_file():
        raise HTTPException(404, '尚未匯入 PDF。')
    return FileResponse(rag.pdf, media_type='application/pdf', headers={'Cache-Control': 'no-store'})
