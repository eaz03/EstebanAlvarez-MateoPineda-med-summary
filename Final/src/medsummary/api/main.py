"""MedSummary in FastAPI: doctor-patient dialogue -> suggested clinical note (ported from Entrega_final/main.py).
Flan-T5 + LoRA writes the draft; an agent with Qwen2.5-1.5B quantized (GGUF Q4_K_M, settings.yaml -> despliegue)
decides whether to use the advanced RAG (medical glossary) and/or schedule the follow-up appointment.

Run from the MedSummary folder (127.0.0.1: only this computer and the tunnel can reach it, not the local network):
    uv run uvicorn medsummary.api.main:app --host 127.0.0.1 --port 8000
Public URL: cloudflared tunnel --url http://localhost:8000
Pages: /  (process)   /registros   /metricas      Docs: /docs      Health check: GET /health (runs no model).

Safety, since the URL is public:
- Only what the system needs is exposed. The per-model endpoints live in api/desarrollo.py, which is never tunneled.
- One request at a time; while one is running, the next one gets 429 instead of queueing (no pile-up of work).
- Dialogues longer than despliegue.max_caracteres are rejected (422).
- Errors never return tracebacks or file paths; the full error is only printed in this computer's console.
- Logs and W&B keep only numbers: never the dialogue, the draft or the note.
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medsummary import observabilidad
from medsummary.api import metricas
from medsummary.core import config
from medsummary.core.pipeline import Pipeline

SETTINGS_DESPLIEGUE = config.settings()['despliegue']
ESTATICOS = Path(__file__).parent / 'static'

pipeline = Pipeline()
lock = threading.Lock()         # one request at a time: the models cannot be used in parallel
carga = {'estado': 'cargando', 'detalle': 'Cargando modelos e indexando el corpus...', 'dispositivo': None, 'segundos': None}


def cargar_en_segundo_plano():
    """Loads the models and builds the index without blocking startup: /health answers meanwhile."""
    inicio = time.perf_counter()
    try:
        with lock:
            carga['dispositivo'] = pipeline.cargar()
        carga.update(estado='listo', detalle='Modelos cargados e índice construido.',
                     segundos=round(time.perf_counter() - inicio, 1))
    except Exception as error:
        # the public message has only the error type: the full one (with paths) stays in the console
        carga.update(estado='error', detalle=f'No se pudieron cargar los modelos ({type(error).__name__}).')
        raise


@asynccontextmanager
async def lifespan(app):
    settings = config.settings()
    observabilidad.configurar_logging()
    observabilidad.iniciar_wandb('produccion', {'seed': settings['seed'], 'modelo_ajustado': settings['models']['summarizer'],
                                                'generador': settings['models']['instruct'],
                                                'cuantizacion': pipeline.variante,
                                                'embedder': settings['models']['embedder'],
                                                'reranker': settings['models']['reranker']})
    threading.Thread(target=cargar_en_segundo_plano, daemon=True).start()
    yield
    observabilidad.cerrar()


app = FastAPI(title='MedSummary', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ESTATICOS), name='static')


@app.middleware('http')
async def cabeceras_de_seguridad(request, call_next):
    """Basic browser protections: no embedding in other sites, no MIME sniffing, no referrer to other sites."""
    respuesta = await call_next(request)
    respuesta.headers['X-Frame-Options'] = 'DENY'
    respuesta.headers['X-Content-Type-Options'] = 'nosniff'
    respuesta.headers['Referrer-Policy'] = 'no-referrer'
    return respuesta


@app.exception_handler(FileNotFoundError)
def archivo_faltante(request, error: FileNotFoundError):
    """e.g. the metrics CSVs are missing: 503 with a message, without the path."""
    print(f'Archivo faltante: {error}')
    return JSONResponse(status_code=503, content={'detail': 'Faltan archivos del sistema (ver la consola del servidor).'})


class DialogoRequest(BaseModel):
    dialogo: str = Field(min_length=1, max_length=SETTINGS_DESPLIEGUE['max_caracteres'],
                         description='Diálogo médico-paciente, en inglés')


# ---------- pages ----------
@app.get('/', include_in_schema=False)
def pagina_procesar():
    return FileResponse(ESTATICOS / 'index.html')

@app.get('/registros', include_in_schema=False)
def pagina_registros():
    return FileResponse(ESTATICOS / 'registros.html')

@app.get('/metricas', include_in_schema=False)
def pagina_metricas():
    return FileResponse(ESTATICOS / 'metricas.html')


# ---------- health check: runs no model ----------
@app.get('/health')
def health():
    """200 if the models loaded and the index answers; 503 while loading or if something failed."""
    salud = pipeline.salud()
    cuerpo = {'status': 'ok' if carga['estado'] == 'listo' else carga['estado'], 'carga': carga,
              'cuantizacion': pipeline.variante, **salud}
    listo = carga['estado'] == 'listo' and all(salud['modelos'].values()) and salud['indice']['responde']
    return JSONResponse(status_code=200 if listo else 503, content=cuerpo)


# ---------- the system end to end ----------
@app.post('/procesar')
def procesar(cuerpo: DialogoRequest):
    if carga['estado'] != 'listo':
        raise HTTPException(status_code=503, detail=f"El sistema no está listo: {carga['detalle']}")
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail='El sistema está procesando otra petición. Intente en unos segundos.')
    try:
        inicio = time.perf_counter()
        resultado = None
        try:
            resultado = pipeline.procesar(cuerpo.dialogo)
        except Exception as error:
            print(f'Error procesando la petición: {type(error).__name__}: {error}')
            raise HTTPException(status_code=500, detail='Error interno al procesar el diálogo.')
        finally:
            if resultado is not None:
                componentes = resultado['componentes']
                valvula = resultado['valvula']
                piezas = {'latencia_borrador_s': resultado['segundos_borrador'],
                          'latencia_agente_s': resultado['segundos_agente']}
            else:
                componentes, valvula, piezas = [], False, None
            observabilidad.registrar_peticion(time.perf_counter() - inicio, componentes, valvula, piezas)
    finally:
        lock.release()
    return {**resultado, 'aviso': SETTINGS_DESPLIEGUE['aviso']}


# ---------- data of the other two views ----------
@app.get('/api/registros')
def registros(n: int = Query(default=200, ge=1, le=1000)):
    return observabilidad.leer_registros(n)

@app.get('/api/metricas')
def tablas_metricas():
    return metricas.tablas()
