"""Runs scripts/generar_tablas.py end to end with fake models (no downloads): checks that every step writes its file,
that the tables have the notebook's headers, and that a second run skips every step. Run: python tests/test_generar_tablas.py"""
import csv, os, runpy, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

# ---------- fakes for the heavy libraries ----------
torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, synchronize=lambda: None)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
torch.mps = types.SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
for n in ['AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForCausalLM']: setattr(tr, n, object)
peft = types.ModuleType('peft'); peft.PeftModel = type('PeftModel', (), {})
class FakeMetric:
    def __init__(self, nombre): self.nombre = nombre
    def compute(self, predictions, references, **kw):
        return {'rouge1': 0.4, 'rouge2': 0.2, 'rougeL': 0.3, 'rougeLsum': 0.3} if self.nombre == 'rouge' else {'f1': [0.5] * len(predictions)}
evaluate = types.ModuleType('evaluate'); evaluate.load = FakeMetric
mods = {n: types.ModuleType(n) for n in ['chromadb', 'sentence_transformers', 'pymupdf', 'llama_cpp', 'huggingface_hub']}
mods['sentence_transformers'].SentenceTransformer = mods['sentence_transformers'].CrossEncoder = object
mods['llama_cpp'].Llama = object; mods['huggingface_hub'].hf_hub_download = lambda *a: None
# fake W&B: records everything that is sent
enviado = {'logs': [], 'artefactos': [], 'cerrado': False, 'init': None}
class FakeRun:
    url = None
    def log(self, datos): enviado['logs'].append(datos)
    def log_artifact(self, artefacto): enviado['artefactos'].append(artefacto)
    def finish(self): enviado['cerrado'] = True
class FakeArtifact:
    def __init__(self, nombre, type): self.archivos = []
    def add_file(self, ruta): self.archivos.append(Path(ruta).name)
wandb = types.ModuleType('wandb')
wandb.init = lambda **kw: enviado.update(init=kw) or FakeRun()
wandb.Settings = lambda **kw: kw
wandb.Table = lambda columns, data: ('tabla', columns, len(data))
wandb.Artifact = FakeArtifact
dotenv = types.ModuleType('dotenv'); dotenv.load_dotenv = lambda *a, **k: None
sys.modules.update(torch=torch, transformers=tr, peft=peft, evaluate=evaluate, wandb=wandb, dotenv=dotenv, **mods)

from medsummary.core import config
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge
from medsummary.core.models.quantized import QuantizedInstructModel
from medsummary.core.rag import Corpus, Reranker, SemanticRetriever

class FakeModulo:                                  # a torch model with 1000 float32 parameters
    def parameters(self): return [types.SimpleNamespace(numel=lambda: 1000, element_size=lambda: 4)]
    def buffers(self): return []
generaciones = []
BaselineModel._cargar_tokenizer_y_modelo = lambda self: (None, FakeModulo())
FineTunedModel._cargar_tokenizer_y_modelo = lambda self: (None, FakeModulo())
BaselineModel._generar_notas = lambda self, dialogos: generaciones.append('t5') or [f'The patient reports fainting {len(d) % 3}.' for d in dialogos]
InstructModel._cargar_tokenizer_y_modelo = lambda self: (None, types.SimpleNamespace(get_memory_footprint=lambda: 3.1e9, dtype='torch.bfloat16'))
def fake_qwen(self, system, user, max_new_tokens=None):
    generaciones.append('qwen')
    if 'ONE technical term' in system: return 'Syncope'
    if 'search questions' in system: return '1. What is the medical term for "fainting"?'
    return 'NO_TOOL'
InstructModel.generar = fake_qwen
QuantizedInstructModel._cargar_tokenizer_y_modelo = lambda self: (None, object())
QuantizedInstructModel.memoria_gb = lambda self: {'q8_0': 1.9, 'q4_k_m': 1.1}[self.variante]
QuantizedInstructModel.liberar = lambda self: None
Judge.generar_desde_mensajes = lambda self, mensajes, max_new_tokens=None: '<feedback>ok</feedback><score>1</score>'
Corpus.desde_pdf = classmethod(lambda cls: cls([{'id': 1, 'fuente': 'F', 'texto': 'Syncope (SIN-kuh-pee): fainting, a brief loss of consciousness.'}]))
SemanticRetriever._cargar_tokenizer_y_modelo = lambda self: (None, FakeModulo())
SemanticRetriever.buscar_semantica = lambda self, consulta, k=3: [(0.9, cid, ch, 'F') for cid, ch in zip(self.corpus.ids, self.corpus.chunks)][:k]
Reranker._cargar_tokenizer_y_modelo = lambda self: (None, types.SimpleNamespace(model=FakeModulo()))
Reranker.rerank = lambda self, consulta, candidatos, k=10: [(1.0, c[1], c[2], c[3]) for c in candidatos][:k]

fallos = []
def ok(nombre, condicion, detalle=''):
    print(('OK   ' if condicion else 'FAIL ') + nombre + (f'  {detalle}' if detalle else ''))
    if not condicion: fallos.append(nombre)

ENCABEZADOS = {
    'scorecard_sistemas.csv': ['metrica', 'baseline', 'finetuned', 'rag_base', 'rag_avanzado'],
    'scorecard_eficiencia.csv': ['metrica', 'baseline', 'finetuned', 'rag_base', 'rag_avanzado', 'avz_mps_sinq', 'avz_mps_q8', 'avz_mps_q4'],
    'tabla_cuantizacion.csv': ['dispositivo', 'version', 'juez_final', 'bert_f1', 'aciertos', 'p50_s', 'p95_s', 'p50_rag_s', 'memoria_qwen_gb', 'vram_extra_gb', 'memoria_total_gb'],
    'tabla_ablacion.csv': ['configuracion', 'juez_final', 'bert_f1', 'acierto', 'context_recall', 'context_precision', 'faithfulness', 'answer_relevancy', 'delta_juez', 'p50_rag_s', 'p95_rag_s', 'notas_distintas'],
}
with tempfile.TemporaryDirectory() as carpeta:
    config.settings()['paths']['outputs'] = carpeta
    sys.argv = ['generar_tablas.py']
    runpy.run_path(str(ROOT / 'scripts' / 'generar_tablas.py'), run_name='__main__')
    salida = Path(carpeta)
    for nombre in ['resultados_sistemas.json', 'resultados_cuantizacion.json', 'resultados_ablacion.json', 'memoria_qwen_referencia.json',
                   'detalle_sistemas.json', 'detalle_eficiencia.json', 'scorecard.csv', 'detalle_baseline.json', 'detalle_finetuned.json']:
        ok(f'{nombre} written', (salida / nombre).is_file())
    for nombre, encabezado in ENCABEZADOS.items():
        ok(f'{nombre}: header as in the notebook', list(csv.reader(open(salida / nombre)))[0] == encabezado)
    cuantizacion = list(csv.DictReader(open(salida / 'tabla_cuantizacion.csv')))
    ok('quantization table: before (bfloat16), Q8_0, Q4_K_M with their memory',
       [f['version'] for f in cuantizacion] == ['Antes: Qwen sin cuantizar (bfloat16)', 'Después: Qwen GGUF Q8_0 (~int8)', 'Después: Qwen GGUF Q4_K_M (~int4)']
       and [f['memoria_qwen_gb'] for f in cuantizacion] == ['3.1', '1.9', '1.1'])
    ablacion = list(csv.DictReader(open(salida / 'tabla_ablacion.csv')))
    ok('ablation table: 4 configurations + reference without RAG', len(ablacion) == 5 and ablacion[0]['delta_juez'] == '0.0')

    # ---------- W&B ----------
    ok('W&B: run started (offline without key) with the config', enviado['init']['job_type'] == 'tablas' and enviado['init']['mode'] == 'offline' and 'modelos' in enviado['init']['config'])
    claves = {k for datos in enviado['logs'] for k in datos}
    ok('W&B: latency per note of every system and Qwen version', {'segundos/finetuned', 'segundos_rag/mps_q4_k_m', 'segundos_rag/sin_reranker'} <= claves)
    ok('W&B: judge scores per evaluated note', {'juez_final/rag_avanzado', 'juez_final/qwen_mps_q8_0', 'juez_final/ablacion_sin_reranker', 'juez_final/qwen_mps_q4_k_m', 'context_recall/qwen_mps_q4_k_m'} <= claves)
    textos = [v for datos in enviado['logs'] for v in datos.values() if isinstance(v, str)]
    ok('W&B: no text is sent (no notes, no dialogues)', textos == [])
    tablas_w = {k for datos in enviado['logs'] for k, v in datos.items() if isinstance(v, tuple) and v[0] == 'tabla'}
    ok('W&B: the CSV tables uploaded as tables and as one artifact', {'tabla_cuantizacion', 'tabla_ablacion', 'scorecard_eficiencia'} <= tablas_w
       and 'tabla_ablacion.csv' in enviado['artefactos'][0].archivos)
    ok('W&B: run closed', enviado['cerrado'])

    generaciones.clear()
    runpy.run_path(str(ROOT / 'scripts' / 'generar_tablas.py'), run_name='__main__')
    ok('second run: every step skipped, nothing generated', generaciones == [])

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
