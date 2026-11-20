"""The deployed API (medsummary.api.main) with FastAPI's TestClient and fake models (no downloads): background loading,
/health, /procesar (tools, valve, limits, busy -> 429, errors without details), JSON logs, the pages and /api/metricas.
Run: python tests/test_api.py"""
import json, os, sys, tempfile, threading, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
for n in ['AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForCausalLM']: setattr(tr, n, object)
peft = types.ModuleType('peft'); peft.PeftModel = type('PeftModel', (), {})
evaluate = types.ModuleType('evaluate'); evaluate.load = lambda nombre: None
mods = {n: types.ModuleType(n) for n in ['chromadb', 'sentence_transformers', 'pymupdf', 'llama_cpp', 'huggingface_hub']}
mods['sentence_transformers'].SentenceTransformer = mods['sentence_transformers'].CrossEncoder = object
mods['llama_cpp'].Llama = object; mods['huggingface_hub'].hf_hub_download = lambda *a: None
sys.modules.update(torch=torch, transformers=tr, peft=peft, evaluate=evaluate, **mods)
puntos_wandb = []                                            # what the API sends to W&B
wandb = types.ModuleType('wandb'); wandb.Settings = lambda **kw: kw
wandb.init = lambda **kw: types.SimpleNamespace(log=puntos_wandb.append, finish=lambda: None, url=None, job_type=kw['job_type'])
sys.modules['wandb'] = wandb

from fastapi.testclient import TestClient
from medsummary.core import config
from medsummary.core.models import FineTunedModel, InstructModel
from medsummary.core.models.quantized import QuantizedInstructModel
from medsummary.core.rag import Corpus, Reranker, SemanticRetriever

carpeta = tempfile.mkdtemp()
config.settings()['paths']['adapter_dir'] = carpeta                       # the adapter "exists"
config.settings()['paths']['logs'] = carpeta + '/logs/peticiones.jsonl'
cargas = []

# ---------- fake models ----------
FineTunedModel._cargar_tokenizer_y_modelo = lambda self: cargas.append('finetuned') or (None, object())
FineTunedModel.resumir = lambda self, d: ('Roses are red.' if 'Roses' in d else 'The patient reports fainting. Follow-up in three months.')
QuantizedInstructModel._cargar_tokenizer_y_modelo = lambda self: cargas.append(('qwen', self.variante)) or (None, object())
Reranker._cargar_tokenizer_y_modelo = lambda self: cargas.append('reranker') or (None, object())
def generar(self, system, user, max_new_tokens=None):
    if 'You decide whether to use ONE tool' in system:
        if 'fainting' not in user:
            return 'NO_TOOL'
        if '"agendar_cita"' in system:
            return '{"tool": "agendar_cita", "args": {"months": 3}}'
        return '{"tool": "buscar_en_corpus", "args": {}}'
    if 'ONE technical term' in system:
        return 'Syncope'
    if 'search questions' in system:
        return '1. What is the medical term for "fainting"?'
    return 'OK'
InstructModel.generar = generar
Corpus.desde_pdf = classmethod(lambda cls: cls([{'id': 1, 'fuente': 'F', 'texto': 'Syncope (SIN-kuh-pee): fainting, a brief loss of consciousness.'}]))
def indexar(self):
    self.cargar(); cargas.append('indice')
    self.coleccion = types.SimpleNamespace(count=lambda: len(self.corpus.chunks)); return self
SemanticRetriever._cargar_tokenizer_y_modelo = lambda self: (None, object())
SemanticRetriever.indexar = indexar
SemanticRetriever.buscar_semantica = lambda self, consulta, k=3: [(0.9, cid, ch, 'F') for cid, ch in zip(self.corpus.ids, self.corpus.chunks)][:k]
Reranker.rerank = lambda self, consulta, candidatos, k=10: [(1.0, c[1], c[2], c[3]) for c in candidatos][:k]

from medsummary.api import main
fallos = []
def ok(nombre, condicion, detalle=''):
    print(('OK   ' if condicion else 'FAIL ') + nombre + (f'  {detalle}' if detalle else ''))
    if not condicion: fallos.append(nombre)

DIALOGO = 'Doctor: Any fainting?\nPatient: Yes, twice.\nDoctor: We will see you again in three months.'

sin_arrancar = TestClient(main.app)                         # without "with": the models have not been loaded yet
r = sin_arrancar.get('/health')
ok('GET /health while loading -> 503', r.status_code == 503 and r.json()['status'] == 'cargando')
r = sin_arrancar.post('/procesar', json={'dialogo': DIALOGO})
ok('POST /procesar while loading -> 503', r.status_code == 503)

with TestClient(main.app) as cliente:
    for _ in range(100):
        if main.carga['estado'] != 'cargando': break
        time.sleep(0.05)
    r = cliente.get('/health')
    ok('GET /health after loading -> 200, every model + index', r.status_code == 200 and all(r.json()['modelos'].values())
       and r.json()['indice'] == {'responde': True, 'chunks': 1} and r.json()['cuantizacion'] == 'q4_k_m', str(r.json()['modelos']))
    ok('deployed Qwen is GGUF Q4_K_M, loaded once', cargas.count(('qwen', 'q4_k_m')) == 1 and cargas.count('finetuned') == 1)

    r = cliente.post('/procesar', json={'dialogo': DIALOGO}); cuerpo = r.json()
    ok('POST /procesar: RAG + appointment', r.status_code == 200 and cuerpo['componentes'] == ['modelo_ajustado', 'rag', 'agendar_cita']
       and 'syncope' in cuerpo['nota'] and 'Next appointment scheduled' in cuerpo['nota'] and not cuerpo['valvula'], cuerpo['nota'])
    ok('POST /procesar: draft, latencies and warning', cuerpo['borrador'].startswith('The patient reports fainting')
       and cuerpo['segundos'] >= cuerpo['segundos_agente'] and cuerpo['aviso'].startswith('Sugerencia'))
    r = cliente.post('/procesar', json={'dialogo': 'Roses are red, lemons are green.'}); cuerpo = r.json()
    ok('POST /procesar adversarial: valve fires, no tools', r.status_code == 200 and cuerpo['valvula'] and cuerpo['componentes'] == ['modelo_ajustado'])

    r = cliente.post('/procesar', json={'dialogo': 'x' * 6001})
    ok('dialogue longer than max_caracteres -> 422', r.status_code == 422)
    r = cliente.post('/procesar', json={'dialogo': ''})
    ok('empty dialogue -> 422', r.status_code == 422)
    main.lock.acquire()
    r = cliente.post('/procesar', json={'dialogo': DIALOGO})
    main.lock.release()
    ok('busy -> 429 instead of queueing', r.status_code == 429)

    original = main.pipeline.procesar
    main.pipeline.procesar = lambda d: (_ for _ in ()).throw(RuntimeError('/Users/secreto/archivo falló'))
    r = cliente.post('/procesar', json={'dialogo': DIALOGO})
    main.pipeline.procesar = original
    ok('error -> 500 without details or paths', r.status_code == 500 and 'secreto' not in r.text, r.json()['detail'])
    ok('lock released after the error', not main.lock.locked())

    lineas = [json.loads(l) for l in open(config.path('logs'), encoding='utf-8')]
    ok('one JSON log line per processed request (3)', len(lineas) == 3, str(lineas[0]))
    ok('log lines: only time, latency, components, valve', all(set(l) == {'hora', 'latencia_s', 'componentes', 'valvula'} for l in lineas))
    ok('no dialogue or note in the logs', 'fainting' not in open(config.path('logs')).read() and 'Roses' not in open(config.path('logs')).read())
    ok('W&B: one point per request with latency and usage metrics', len(puntos_wandb) == 3
       and all({'latencia_s', 'peticiones', 'usos_rag', 'usos_agendar_cita', 'tasa_valvula'} <= set(p) for p in puntos_wandb)
       and puntos_wandb[-1]['peticiones'] == 3, str(puntos_wandb[1]))
    ok('W&B: only numbers', all(isinstance(v, (int, float)) for p in puntos_wandb for v in p.values()))
    r = cliente.get('/api/registros?n=2')
    ok('GET /api/registros: most recent first', r.status_code == 200 and len(r.json()) == 2 and r.json()[0]['componentes'] == [])
    ok('GET /api/registros?n=5000 -> 422', cliente.get('/api/registros?n=5000').status_code == 422)

    r = cliente.get('/api/metricas')
    if r.status_code == 200:
        tablas = r.json()
        ok('GET /api/metricas: 3 tables', [t['titulo'][:12] for t in tablas] == ['Antes y desp', 'Comparación ', 'Ablación del'])
        ok('deployed Q4_K_M marked in both tables', any('Q4_K_M' in f['version'] and 'desplegada' in f['version'] for f in tablas[0]['filas'])
           and 'Q4_K_M · desplegado' in tablas[1]['filas'][-1]['sistema'], tablas[1]['filas'][-1]['sistema'])
    else:
        ok('GET /api/metricas without CSVs -> 503 without path', r.status_code == 503 and 'metricas' not in r.text)

    for ruta in ['/', '/registros', '/metricas', '/static/comun.js', '/static/estilos.css']:
        r = cliente.get(ruta)
        ok(f'GET {ruta}', r.status_code == 200 and r.headers['x-frame-options'] == 'DENY' and r.headers['x-content-type-options'] == 'nosniff')
    ok('development endpoints are not in the deployed API', cliente.post('/liberar').status_code == 404
       and cliente.post('/judge', json={}).status_code == 404)

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
