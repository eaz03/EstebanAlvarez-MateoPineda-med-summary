"""Development API: one endpoint per model/system, plus judge, RAGAS and harness, to try each piece by hand.
LOCAL ONLY: never expose it through a tunnel (POST /liberar unloads every model, /judge and /harness load Flow-Judge).
The deployed system is medsummary.api.main.
Models are created on first use and stay in memory; POST /liberar frees them.

Run from the MedSummary folder:
    uv run uvicorn medsummary.api.desarrollo:app --host 127.0.0.1 --port 8001
Interactive docs: http://127.0.0.1:8001/docs
"""
import threading
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from medsummary.core.data import cargar_eval_set
from medsummary.core.evaluation import Harness, RagasEvaluator
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge
from medsummary.core.rag import AdvancedRAG, BasicRAG, BM25Retriever, Corpus, Reranker, SemanticRetriever, ToolAgent

app = FastAPI(title='MedSummary (desarrollo, solo local)')
lock = threading.Lock()     # one generation at a time: models are not safe to call concurrently (MPS/CUDA)


# ---------- models, created on first use (creating one does not load its weights) ----------
@lru_cache
def baseline() -> BaselineModel:
    return BaselineModel()

@lru_cache
def finetuned() -> FineTunedModel:
    return FineTunedModel()

@lru_cache
def qwen() -> InstructModel:
    return InstructModel()

@lru_cache
def judge() -> Judge:
    return Judge()

@lru_cache
def corpus() -> Corpus:
    return Corpus.desde_pdf()

@lru_cache
def semantic() -> SemanticRetriever:
    return SemanticRetriever(corpus())

@lru_cache
def reranker() -> Reranker:
    return Reranker()

@lru_cache
def basic_rag() -> BasicRAG:
    return BasicRAG(semantic(), qwen())

@lru_cache
def advanced_rag() -> AdvancedRAG:
    return AdvancedRAG(semantic(), BM25Retriever(corpus()), reranker(), qwen())

@lru_cache
def tool_agent() -> ToolAgent:
    return ToolAgent(advanced_rag(), qwen())

MODELOS = {'baseline': baseline, 'finetuned': finetuned, 'qwen': qwen, 'judge': judge, 'reranker': reranker}


# ---------- request bodies ----------
class DialogoRequest(BaseModel):
    dialogo: str

class BorradorRequest(BaseModel):
    borrador: str

class ToolsRequest(BaseModel):
    borrador: str
    dialogo: str | None = None

class JudgeRequest(BaseModel):
    dialogo: str
    nota: str

class RagasRequest(BaseModel):
    borrador: str
    nota: str
    chunks: list[str]

class HarnessRequest(BaseModel):
    sistema: Literal['baseline', 'finetuned'] = 'finetuned'
    n_ejemplos: int = 2


@app.exception_handler(FileNotFoundError)
def archivo_faltante(request, error: FileNotFoundError):
    """Missing adapter or PDF -> 503 with the message, instead of a 500."""
    return JSONResponse(status_code=503, content={'detail': str(error)})


# ---------- health: never runs a model ----------
@app.get('/health')
def health():
    estado = {}
    for nombre, crear_modelo in MODELOS.items():
        modelo = crear_modelo()
        estado[nombre] = {'modelo_id': modelo.modelo_id, 'device': modelo.device, 'cargado': modelo.esta_cargado}
    return {'status': 'ok', 'modelos': estado}


# ---------- summarizers ----------
@app.post('/baseline')
def resumir_baseline(cuerpo: DialogoRequest):
    with lock:
        return {'nota': baseline().resumir(cuerpo.dialogo)}

@app.post('/finetuned')
def resumir_finetuned(cuerpo: DialogoRequest):
    with lock:
        return {'nota': finetuned().resumir(cuerpo.dialogo)}


# ---------- RAG and tools (input: the fine-tuned draft) ----------
@app.post('/rag/basic')
def rag_basico(cuerpo: BorradorRequest):
    with lock:
        nota, chunks = basic_rag().sistema_rag_detallado(cuerpo.borrador)
    return {'nota': nota, 'chunks': chunks}

@app.post('/rag/advanced')
def rag_avanzado(cuerpo: BorradorRequest):
    with lock:
        nota, chunks = advanced_rag().sistema_rag_avanzado_detallado(cuerpo.borrador)
    return {'nota': nota, 'chunks': chunks}

@app.post('/tools')
def tools(cuerpo: ToolsRequest):
    with lock:
        nota, llamadas = tool_agent().responder_con_tools_detallado(cuerpo.borrador, cuerpo.dialogo)
    return {'nota': nota, 'llamadas': llamadas}


# ---------- evaluation ----------
@app.post('/judge')
def juzgar(cuerpo: JudgeRequest):
    with lock:
        final, jueces = judge().juez_completo(cuerpo.dialogo, cuerpo.nota)
    return {'final': final, 'jueces': jueces}

@app.post('/ragas')
def ragas(cuerpo: RagasRequest):
    with lock:
        puntajes, explicaciones = RagasEvaluator(judge()).ragas(cuerpo.borrador, cuerpo.nota, cuerpo.chunks)
    return {'puntajes': puntajes, 'explicaciones': explicaciones}

@app.post('/harness')
def harness(cuerpo: HarnessRequest):
    """Harness on the first n_ejemplos of the eval set. Slow: each example calls the judge 3 times."""
    if cuerpo.sistema == 'baseline':
        sistema = baseline()
    else:
        sistema = finetuned()
    eval_set = cargar_eval_set()[:cuerpo.n_ejemplos]
    with lock:
        return Harness(judge()).harness(eval_set, sistema.resumir)


# ---------- memory ----------
@app.post('/liberar')
def liberar():
    """Frees every loaded model (useful on the Mac before calling the judge)."""
    with lock:
        for crear_modelo in MODELOS.values():
            crear_modelo().liberar()
    return health()
