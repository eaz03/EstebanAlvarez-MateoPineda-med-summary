"""Runs main.py end to end with fake models (no downloads), with and without the adapter/PDF, to check the wiring.
Run: python tests/test_main.py"""
import contextlib, io, os, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))

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

from medsummary.core import config
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge
from medsummary.core.rag import Corpus, Reranker, SemanticRetriever

# fake behaviour at class level: no weights are loaded
BaselineModel._generar_notas = lambda self, dialogos: ['The patient reports swelling of the legs and fainting.' for _ in dialogos]
FineTunedModel.norma_lora_b = lambda self: 1.234
InstructModel.generar = lambda self, system, user, max_new_tokens=None: 'Syncope' if 'ONE technical term' in system else ('1. What is the medical term for "fainting"?' if 'search questions' in system else 'NO_TOOL')
Judge.generar_desde_mensajes = lambda self, mensajes, max_new_tokens=None: '<feedback>ok</feedback><score>1</score>'
paginas = [{'id': 1, 'fuente': 'F', 'texto': 'Syncope (SIN-kuh-pee): fainting, a brief loss of consciousness.'},
           {'id': 2, 'fuente': 'F', 'texto': 'Edema (eh-DEE-muh): swelling of the legs caused by fluid in the tissues.'}]
Corpus.desde_pdf = classmethod(lambda cls: cls(paginas))
SemanticRetriever.buscar_semantica = lambda self, consulta, k=3: [(0.9, cid, ch, 'F') for cid, ch in zip(self.corpus.ids, self.corpus.chunks)][:k]
Reranker.rerank = lambda self, consulta, candidatos, k=10: [(1.0, c[1], c[2], c[3]) for c in candidatos][:k]

import main
fallos = []
for con_archivos in (True, False):
    with tempfile.TemporaryDirectory() as carpeta:
        adapter, pdf = Path(carpeta) / 'adapter', Path(carpeta) / 'corpus.pdf'
        if con_archivos:
            adapter.mkdir(); pdf.write_text('x')
        config.settings()['paths'].update(adapter_dir=str(adapter), corpus_pdf=str(pdf))
        salida = io.StringIO()
        try:
            with contextlib.redirect_stdout(salida): main.main()
            texto = salida.getvalue()
            esperado = ['Dispositivo: mps', '5 · Harness', 'Aciertos de dominio'] + (
                ['Norma lora_B', 'RAG avanzado: The patient reports swelling of the legs and syncope.', 'RAGAS:'] if con_archivos
                else ['Sin adapter', 'se omite el RAG'])
            faltan = [e for e in esperado if e not in texto]
            print(('OK   ' if not faltan else 'FAIL ') + f'main.py runs end to end (adapter and PDF present={con_archivos})' + (f' missing {faltan}' if faltan else ''))
            if faltan: fallos.append(con_archivos)
        except Exception as error:
            print(f'FAIL main.py (adapter and PDF present={con_archivos}): {type(error).__name__}: {error}'); fallos.append(con_archivos)
        if con_archivos: print('\n--- sample output ---\n' + salida.getvalue()[:1800] + '\n---------------------')
print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED')
sys.exit(1 if fallos else 0)
