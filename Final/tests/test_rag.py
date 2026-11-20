"""Runs the notebook's own cells and the new rag/ classes side by side, with the same fake models, and requires
identical outputs and identical LLM calls. No weights are downloaded. Run: python tests/test_rag.py"""
import hashlib, io, json, math, os, re, sys, types, contextlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

# ---------------- fakes for heavy libraries ----------------
torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
torch.mps = types.SimpleNamespace(empty_cache=lambda: None); torch.manual_seed = lambda s: None
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
for n in ['AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForCausalLM']: setattr(tr, n, object)
peft = types.ModuleType('peft'); peft.PeftModel = type('PeftModel', (), {})

class Vector(list):
    def tolist(self): return [list(v) for v in self]
def _embed(texto, dims=64):
    v = [0.0] * dims
    for w in re.findall(r'\w+', texto.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]
class FakeST:
    def __init__(self, name, device=None): self.name = name
    def encode(self, textos, **kw):
        import numpy
        return numpy.array([_embed(t) for t in textos])
class FakeCollection:
    def __init__(self): self.rows = []
    def add(self, ids, documents, metadatas, embeddings): self.rows += list(zip(ids, documents, metadatas, embeddings))
    def count(self): return len(self.rows)
    def query(self, query_embeddings, n_results):
        q = query_embeddings[0]
        scored = sorted(((1 - sum(a * b for a, b in zip(q, e)), i, d, m) for i, d, m, e in self.rows), key=lambda t: (t[0], t[1]))[:n_results]
        return {'distances': [[s[0] for s in scored]], 'ids': [[s[1] for s in scored]],
                'documents': [[s[2] for s in scored]], 'metadatas': [[s[3] for s in scored]]}
class FakeClient:
    _cols = {}
    def delete_collection(self, n): raise ValueError('none')
    def create_collection(self, n, metadata): return FakeCollection()
chromadb = types.ModuleType('chromadb'); chromadb.Client = FakeClient
class FakeCE:
    def __init__(self, name, max_length=None, device=None): pass
    def predict(self, pares):
        out = []
        for q, d in pares:
            qs, ds = set(re.findall(r'\w+', q.lower())), set(re.findall(r'\w+', d.lower()))
            out.append(len(qs & ds) - 0.001 * len(d))
        return out
st_mod = types.ModuleType('sentence_transformers'); st_mod.SentenceTransformer = FakeST; st_mod.CrossEncoder = FakeCE
pymupdf = types.ModuleType('pymupdf')
evaluate = types.ModuleType('evaluate'); evaluate.load = lambda nombre: None   # not used here, imported by evaluation/
sys.modules.update(torch=torch, transformers=tr, peft=peft, chromadb=chromadb, sentence_transformers=st_mod, pymupdf=pymupdf,
                   evaluate=evaluate)

from medsummary.core import config
from medsummary.core.rag import (AdvancedRAG, BasicRAG, BM25Retriever, Corpus, HybridRetriever, KeywordRetriever,
                                 Reranker, SemanticRetriever, ToolAgent)
from medsummary.core.rag.corpus import limpiar_texto
from medsummary.core.evaluation import RagasEvaluator

nb = json.load(open('notebooks/Entrega3_TopicosIA_EAZ_MPA.ipynb'))
cell = lambda i: ''.join(nb['cells'][i]['source'])
fallos = []
def ok(nombre, condicion):
    print(('OK   ' if condicion else 'FAIL ') + nombre)
    if not condicion: fallos.append(nombre)
def run(code, g):
    with contextlib.redirect_stdout(io.StringIO()): exec(code, g)

# ---------------- synthetic PDF pages ----------------
FUENTE = config.settings()['corpus']['fuente']
paginas_crudas = [
    "Chapter 4 Cardiovascular\n\nEdema (eh-DEE-muh): swelling of the legs or other tissues caused by fluid. Syncope (SIN-kuh-pee): fainting, a brief\nloss of consciousness.\n\n12\n",
    "• Hypertension (hy-per-TEN-shun): high blood pressure in the arteries.\n• Dyspnea (DISP-nee-uh): shortness of breath or difficult breathing.\n• short\n",
    "Introduction to the respir-\natory system. The lungs exchange gases with the blood in every breath taken by the patient.\n\nSee https://example.org/ref for more.\n\ncomplications.\n",
    "Pyrexia (pye-REK-see-uh): fever, a body temperature above normal. Polydipsia (pol-ee-DIP-see-uh): excessive thirst over a long time.\n",
]
import numpy as np
g = {'re': re, 'np': np, 'torch': torch}
c43 = cell(43); run(c43[c43.index('def limpiar_texto'):c43.index('corpus = []')], g)
for t in paginas_crudas:
    ok('limpiar_texto == cell 43', limpiar_texto(t) == g['limpiar_texto'](t))
documentos = [{'id': n, 'fuente': FUENTE, 'texto': g['limpiar_texto'](t)} for n, t in enumerate(paginas_crudas, start=1)]
g['corpus'] = documentos
c45 = cell(45); run(c45[:c45.index("print(f'{len(corpus)}")], g)
corpus = Corpus(documentos)
ok(f'chunks/ids/metadatos == cell 45 ({len(corpus.chunks)} chunks)', (corpus.chunks, corpus.ids, corpus.metadatos) == (g['chunks'], g['ids'], g['metadatos']))

# ---------------- retrievers ----------------
run(cell(47), g); run(cell(49).split('def comparar_busquedas')[0], g)
semantic = SemanticRetriever(corpus); keyword = KeywordRetriever(corpus)
consultas = ['She complains of swelling of the legs and fainting.', 'high blood pressure', 'fever and thirst', 'breath']
for q in consultas:
    ok(f'buscar_semantica == cell 49 [{q[:20]}]', semantic.buscar_semantica(q, 5) == g['buscar_semantica'](q, 5))
    ok(f'buscar_keyword == cell 49 [{q[:20]}]', keyword.buscar_keyword(q, 3) == g['buscar_keyword'](q, 3))
run(cell(59), g); run(cell(60), g)
bm25 = BM25Retriever(corpus); hybrid = HybridRetriever(semantic, bm25)
for q in consultas:
    ok(f'buscar_bm25 == cell 59 [{q[:20]}]', bm25.buscar_bm25(q, 4) == g['buscar_bm25'](q, 4))
    ok(f'buscar_hibrida == cell 60 [{q[:20]}]', hybrid.buscar_hibrida(q, k=4, n=6) == g['buscar_hibrida'](q, k=4, n=6))
g['reranker'] = FakeCE('x'); run(cell(64), g)
reranker = Reranker()
cand = g['buscar_hibrida']('fainting', k=6, n=6)
ok('rerank == cell 64', reranker.rerank('What is the medical term for "fainting"?', cand, k=4) == g['rerank']('What is the medical term for "fainting"?', cand, k=4))

# ---------------- fake Qwen shared by notebook and classes ----------------
prompts_rag = config.prompt('rag.yaml')
TERMINOS = {'swelling of the legs': 'Edema', 'fainting': 'Syncope', 'high blood pressure': 'Hypertension',
            'shortness of breath': 'Dyspnea', 'fever': 'Pyrexia', 'thirst': 'NONE'}
def fake_generar(registro):
    def generar(system, user, max_new_tokens=220):
        registro.append((system, user, max_new_tokens))
        if 'You generate search questions' in system:
            return '\n'.join(f'{i}. What is the medical term for "{e}"?' for i, e in enumerate([e for e in TERMINOS if e in user.lower()], 1))
        if 'answer with ONE technical term' in system:
            m = re.search(r'Expression: "([^"]+)"', user); return TERMINOS.get(m.group(1).lower(), 'NONE') + '.'
        if 'You decide whether to use ONE tool' in system:
            if '"agendar_cita"' in system:
                if 'month' not in user.lower(): return 'NO_TOOL'
                return '{"tool": "agendar_cita", "args": {"months": 3}}' if 'Fix the arguments' in user else '{"tool": "agendar_cita", "args": {"months": 30}}'
            return '{"tool": "buscar_en_corpus", "args": {}}' if any(e in user.lower() for e in TERMINOS) else 'NO_TOOL'
        return 'REWRITTEN ' + hashlib.md5((system + user).encode()).hexdigest()[:8]
    return generar
reg_nb, reg_new = [], []
g['generar'] = fake_generar(reg_nb)
run(cell(54), g); run(cell(67), g); run(cell(70), g)
instruct = types.SimpleNamespace(generar=fake_generar(reg_new))
basic = BasicRAG(semantic, instruct); advanced = AdvancedRAG(semantic, bm25, reranker, instruct)

borradores = ['The patient presents with high blood pressure and shortness of breath.',
              'She complains of swelling of the legs and fainting. She also reports fever and thirst for two days.',
              'The cat lies on the couch.']
for b in borradores:
    reg_nb.clear(); reg_new.clear()
    ok(f'sistema_rag == cell 54 [{b[:20]}]', basic.sistema_rag(b) == g['sistema_rag'](b) and reg_new == reg_nb)
    ok(f'sistema_sin_rag == cell 54 [{b[:20]}]', basic.sistema_sin_rag(b) == g['sistema_sin_rag'](b))
    reg_nb.clear(); reg_new.clear()
    ok(f'decomposition == cell 67 [{b[:20]}]', advanced.decomposition(b) == g['decomposition'](b))
    sin_descartada = [c for c in reg_nb if c[2] != 150]
    ok(f'  same LLM calls minus the discarded one ({len(reg_nb) - len(sin_descartada)} removed)', reg_new == sin_descartada)
    reg_nb.clear(); reg_new.clear()
    nota_new, chunks_new = advanced.sistema_rag_avanzado_detallado(b)
    nota_nb = g['sistema_rag_avanzado'](b)
    ctx = g['recuperar_avanzado'](b, k=10)
    chunks_nb = [t for _, ch in ctx for _, _, t, _ in ch]
    ok(f'sistema_rag_avanzado == cell 70 [{b[:20]}] -> {nota_new[:60]!r}', nota_new == nota_nb and chunks_new == chunks_nb)

# ---------------- ablation switches vs recuperar_avanzado_ablacion (final notebook, cell 92) ----------------
nb4 = json.load(open('notebooks/Entrega4_TopicosIA_EAZ_MPA.ipynb'))
c92 = ''.join(nb4['cells'][92]['source'])
run(c92[c92.index('def recuperar_avanzado_ablacion'):c92.index('def correr_rag_avanzado')], g)
for usar_hibrida, usar_reranker in [(True, True), (True, False), (False, True), (False, False)]:
    rag_ablacion = AdvancedRAG(semantic, bm25, reranker, instruct, usar_hibrida=usar_hibrida, usar_reranker=usar_reranker)
    iguales = all(rag_ablacion.recuperar_avanzado(b) == g['recuperar_avanzado_ablacion'](b, k=10, usar_hibrida=usar_hibrida, usar_reranker=usar_reranker)
                  for b in borradores)
    ok(f'ablation hybrid={usar_hibrida} reranker={usar_reranker} == cell 92', iguales)

# ---------------- tools ----------------
g['sistema_rag_avanzado'] = g['sistema_rag_avanzado']
run(cell(74), g); run(cell(75), g); run(cell(76), g); run(cell(77), g)
agent = ToolAgent(advanced, instruct)
ok('ESQUEMAS == cell 75', agent.esquemas == g['ESQUEMAS'])
for nombre in agent.tools:
    ok(f'_system_una_tool({nombre}) == cell 77', agent._system_una_tool(nombre) == g['_system_una_tool'](nombre))
casos = [('The patient reports swelling of the legs. Follow-up in three months.', 'Doctor: see you in three months.'),
         ('The patient presents with high blood pressure.', None), ('Roses are red, lemons are green.', None)]
for resumen, dialogo in casos:
    reg_nb.clear(); reg_new.clear()
    nota_new, llamadas = agent.responder_con_tools_detallado(resumen, dialogo)
    with contextlib.redirect_stdout(io.StringIO()) as log_nb: nota_nb = g['responder_con_tools'](resumen, dialogo)
    ok(f'responder_con_tools == cell 77 [{resumen[:25]}] -> {nota_new[:70]!r} ({len(llamadas)} calls)', nota_new == nota_nb)

# ---------------- RAGAS (Flow, cell 85) ----------------
run(cell(85), g)
def fake_juez(registro):
    def juzgar(entradas, salida, metrica):
        registro.append((json.dumps(entradas, sort_keys=True), salida, metrica))
        return (len(salida) + len(metrica)) % 2, f'exp {metrica}'
    return juzgar
reg_nb, reg_new = [], []
g['juzgar_flow'] = fake_juez(reg_nb)
evaluador = RagasEvaluator(types.SimpleNamespace(juzgar_binario=fake_juez(reg_new)))
for borrador, nota, chunks in [('draft a', 'draft a', ['c1', 'c22']), ('draft a', 'note b', ['c1', 'c22', 'c333']), ('draft', 'other', [])]:
    reg_nb.clear(); reg_new.clear()
    ok(f'ragas == cell 85 (changed={borrador != nota}, chunks={len(chunks)})', evaluador.ragas(borrador, nota, chunks) == g['ragas'](borrador, nota, chunks) and reg_new == reg_nb)

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
