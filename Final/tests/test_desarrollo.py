"""Calls every endpoint of the development API with FastAPI's TestClient and fake models (no downloads).
Run: python tests/test_desarrollo.py"""
import os, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
for n in ['AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForCausalLM']: setattr(tr, n, object)
peft = types.ModuleType('peft'); peft.PeftModel = type('PeftModel', (), {})
class FakeMetric:
    def __init__(self, nombre): self.nombre = nombre
    def compute(self, predictions, references, **kw):
        return {'rouge1': 0.4, 'rouge2': 0.2, 'rougeL': 0.3, 'rougeLsum': 0.3} if self.nombre == 'rouge' else {'f1': [0.7] * len(predictions)}
evaluate = types.ModuleType('evaluate'); evaluate.load = FakeMetric
mods = {n: types.ModuleType(n) for n in ['chromadb', 'sentence_transformers', 'pymupdf']}
mods['sentence_transformers'].SentenceTransformer = mods['sentence_transformers'].CrossEncoder = object
sys.modules.update(torch=torch, transformers=tr, peft=peft, evaluate=evaluate, **mods)

from fastapi.testclient import TestClient
from medsummary.core import config
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge
from medsummary.core.rag import Corpus, Reranker, SemanticRetriever

BaselineModel._cargar_tokenizer_y_modelo = lambda self: (None, object())   # fake weights; FineTunedModel still checks the adapter
BaselineModel._generar_notas = lambda self, dialogos: self.cargar() and ['The patient reports fainting.' for _ in dialogos]
InstructModel.generar = lambda self, system, user, max_new_tokens=None: 'Syncope' if 'ONE technical term' in system else ('1. What is the medical term for "fainting"?' if 'search questions' in system else 'NO_TOOL')
Judge.generar_desde_mensajes = lambda self, mensajes, max_new_tokens=None: '<feedback>ok</feedback><score>1</score>'
Corpus.desde_pdf = classmethod(lambda cls: cls([{'id': 1, 'fuente': 'F', 'texto': 'Syncope (SIN-kuh-pee): fainting, a brief loss of consciousness.'}]))
SemanticRetriever.buscar_semantica = lambda self, consulta, k=3: [(0.9, cid, ch, 'F') for cid, ch in zip(self.corpus.ids, self.corpus.chunks)][:k]
Reranker.rerank = lambda self, consulta, candidatos, k=10: [(1.0, c[1], c[2], c[3]) for c in candidatos][:k]

from medsummary.api.desarrollo import app
cliente = TestClient(app)
fallos = []
def ok(nombre, condicion, detalle=''):
    print(('OK   ' if condicion else 'FAIL ') + nombre + (f'  {detalle}' if detalle else ''))
    if not condicion: fallos.append(nombre)

with tempfile.TemporaryDirectory() as carpeta:
    config.settings()['paths']['adapter_dir'] = carpeta + '/no_existe'
    r = cliente.get('/health').json()
    ok('GET /health (no model loaded, nothing generated)', r['status'] == 'ok' and not any(m['cargado'] for m in r['modelos'].values()), str({k: v['device'] for k, v in r['modelos'].items()}))
    r = cliente.post('/baseline', json={'dialogo': 'Doctor: hi\nPatient: I fainted.'}); ok('POST /baseline', r.status_code == 200 and r.json()['nota'], r.json()['nota'])
    r = cliente.post('/finetuned', json={'dialogo': 'x'}); ok('POST /finetuned without adapter -> 503 with message', r.status_code == 503 and 'No existe' in r.json()['detail'])
    config.settings()['paths']['adapter_dir'] = carpeta
    FineTunedModel.resumir = lambda self, d: 'The patient reports fainting.'
    r = cliente.post('/finetuned', json={'dialogo': 'x'}); ok('POST /finetuned', r.status_code == 200, r.json()['nota'])
    r = cliente.post('/rag/basic', json={'borrador': 'The patient reports fainting.'}); ok('POST /rag/basic', r.status_code == 200 and 'chunks' in r.json())
    r = cliente.post('/rag/advanced', json={'borrador': 'The patient reports fainting.'}); ok('POST /rag/advanced', r.status_code == 200 and r.json()['nota'] == 'The patient reports syncope.', r.json()['nota'])
    r = cliente.post('/tools', json={'borrador': 'The patient reports fainting.'}); ok('POST /tools', r.status_code == 200 and 'llamadas' in r.json())
    r = cliente.post('/judge', json={'dialogo': 'd', 'nota': 'n'}); ok('POST /judge', r.status_code == 200 and set(r.json()['jueces']) == {'grounding', 'coverage', 'form'}, f"final={r.json()['final']}")
    r = cliente.post('/ragas', json={'borrador': 'a', 'nota': 'b', 'chunks': ['c1', 'c2']}); ok('POST /ragas', r.status_code == 200 and set(r.json()['puntajes']) == {'context_recall', 'context_precision', 'faithfulness', 'answer_relevancy'})
    r = cliente.post('/harness', json={'sistema': 'baseline', 'n_ejemplos': 2}); ok('POST /harness', r.status_code == 200 and r.json()['total'] == 2, f"aciertos={r.json()['aciertos']}/2")
    r = cliente.post('/harness', json={'sistema': 'otro'}); ok('POST /harness rejects unknown system (422)', r.status_code == 422)
    r = cliente.post('/liberar'); ok('POST /liberar', r.status_code == 200 and not any(m['cargado'] for m in r.json()['modelos'].values()))

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
