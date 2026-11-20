"""Runs the notebook's evaluation cells (31, 36, 37, 38, 83, 88) and the new evaluation/ classes side by side with the
same fake metrics, judge and systems, and requires identical results. Run: python tests/test_evaluation.py"""
import contextlib, io, json, os, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

# ---------------- fakes ----------------
torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, memory_allocated=lambda: 0)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
for n in ['AutoTokenizer', 'AutoModelForSeq2SeqLM', 'AutoModelForCausalLM']: setattr(tr, n, object)
peft = types.ModuleType('peft'); peft.PeftModel = type('PeftModel', (), {})
llamadas_bert = []
class FakeMetric:
    def __init__(self, nombre): self.nombre = nombre
    def compute(self, predictions, references, **kw):
        if self.nombre == 'rouge':
            def overlap(p, r): a, b = set(p.lower().split()), set(r.lower().split()); return len(a & b) / max(len(b), 1)
            m = sum(overlap(p, r) for p, r in zip(predictions, references)) / len(predictions)
            return {'rouge1': m, 'rouge2': m / 2, 'rougeL': m * 0.9, 'rougeLsum': m * 0.95}
        llamadas_bert.append(kw)
        return {'f1': [min(len(p), len(r)) / max(len(p), len(r), 1) for p, r in zip(predictions, references)]}
evaluate = types.ModuleType('evaluate'); evaluate.load = FakeMetric
others = {n: types.ModuleType(n) for n in ['chromadb', 'sentence_transformers', 'pymupdf', 'rank_bm25']}
others['sentence_transformers'].SentenceTransformer = others['sentence_transformers'].CrossEncoder = object
others['rank_bm25'].BM25Okapi = object
sys.modules.update(torch=torch, transformers=tr, peft=peft, evaluate=evaluate, **others)

from medsummary.core import config
from medsummary.core.data import cargar_eval_set
from medsummary.core.evaluation import ComparacionSistemas, Harness, SanityCheck, RagasEvaluator, classic_metrics, comparar_metricas, metricas_dataset

nb = json.load(open('notebooks/Entrega3_TopicosIA_EAZ_MPA.ipynb'))
cell = lambda i: ''.join(nb['cells'][i]['source'])
fallos = []
def ok(nombre, condicion):
    print(('OK   ' if condicion else 'FAIL ') + nombre)
    if not condicion: fallos.append(nombre)
def run(code, g):
    with contextlib.redirect_stdout(io.StringIO()) as salida: exec(code, g)
    return salida.getvalue()

eval_set = cargar_eval_set()
ok(f'eval set loads ({len(eval_set)} examples, keys input/esperado/criterio)', len(eval_set) == 10 and set(eval_set[0]) == {'input', 'esperado', 'criterio'})

# ---------------- fake judge (same for notebook and classes) ----------------
def fake_juez_completo(dialogo, respuesta, criterio=None):
    g_, c_, f_ = 1 + len(respuesta) % 5, 1 + len(dialogo) % 5, 1 + (len(respuesta) * 7) % 5
    r = {'grounding': {'score': g_, 'explanation': f'g{g_}', 'valido': g_ != 3, 'inconsistente': False},
         'coverage': {'score': c_, 'explanation': f'c{c_}', 'valido': True, 'inconsistente': False},
         'form': {'score': f_, 'explanation': f'f{f_}', 'valido': True, 'inconsistente': False}}
    final = 0.45 * g_ + 0.35 * c_ + 0.20 * f_
    if g_ <= 2: final = min(final, 2.0)
    return round(final, 2), r
judge = types.SimpleNamespace(juez_completo=fake_juez_completo, rubricas={'grounding': 0, 'coverage': 0, 'form': 0},
                              pesos={'grounding': 0.45, 'coverage': 0.35, 'form': 0.20})

# ---------------- cell 31: classic_metrics ----------------
g = {}; run(cell(31), g)
ok('classic_metrics == cell 31', classic_metrics('The patient has fever.', 'Patient reports fever.') == g['classic_metrics']('The patient has fever.', 'Patient reports fever.'))
ok('BERTScore args: lang=en, rescale=True, device=mps (Apple GPU)', llamadas_bert[0] == {'lang': 'en', 'rescale_with_baseline': True, 'device': 'mps'})
preds, refs = ['a b c', 'fever and cough'], ['a b', 'cough fever']
m = metricas_dataset(preds, refs)
rr = FakeMetric('rouge').compute(preds, refs); bb = FakeMetric('bertscore').compute(preds, refs)
ok('metricas_dataset == cells 10/16 (corpus ROUGE + mean BERT F1)', m == {**rr, 'bert_f1': sum(bb['f1']) / len(bb['f1'])})
cmp = comparar_metricas({k: 0.1 for k in m}, {k: 0.3 for k in m})
ok('comparar_metricas delta == cell 18', all(abs(v['delta'] - 0.2) < 1e-12 for v in cmp.values()) and list(cmp) == ["rouge1", "rouge2", "rougeL", "rougeLsum", "bert_f1"])

# ---------------- cell 36: harness ----------------
g.update(juez_completo=fake_juez_completo); run(cell(36), g)
harness = Harness(judge)
sistema = lambda d: 'The patient ' + d[:60].replace('\n', ' ')
nuevo, viejo = harness.harness(eval_set, sistema), g['harness'](eval_set, sistema)
ok('harness scorecard == cell 36 (averages, aciertos, per-example detail)', nuevo == viejo)

# ---------------- cell 38: table ----------------
sb, sf = viejo, g['harness'](eval_set, lambda d: d[:120])
c38 = cell(38); impresion = c38[c38.index('FILAS = '):c38.index("import csv, json")]
salida_nb = run(impresion, {**g, 'sb': sb, 'sf': sf}).rstrip('\n')
ok('Harness.tabla == cell 38 printout', Harness.tabla({'Baseline': sb, 'Finetuned': sf}) == salida_nb)

# ---------------- cell 37: sanity check ----------------
c37 = cell(37); run(c37[:c37.index('# ---------- 1)')], g)
registros_nb = [g['evaluar_nota'](n, t, e, nota) for n, e in enumerate(eval_set) for t, nota in g['crear_notas_de_control'](e).items()]
sanity = SanityCheck(harness)
with tempfile.TemporaryDirectory() as carpeta:
    config.settings()['paths']['outputs'] = carpeta
    registros = sanity.ejecutar(eval_set)
    ok('sanity writes detalle_sanity.json', (Path(carpeta) / 'detalle_sanity.json').is_file())
ok(f'sanity registros == cell 37 ({len(registros)} control notes)', registros == registros_nb)
ok('sanity resumir == cell 37', all(sanity.resumir(registros, t, j) == g['resumir'](registros_nb, t, j) for t in g['EXPECTATIVAS'] for j in g['JUECES']))
ok('sanity reporte builds', 'CADA JUEZ DETECTA SU PROPIO ERROR' in sanity.reporte(registros))

# ---------------- cells 83 + 88: 5-system comparison ----------------
K = 10
def fake_baseline(d): return 'Baseline note: ' + d[:50]
def fake_finetuned(d): return 'The patient reports ' + d[10:70]
def fake_buscar(b, k): return [(0.5, f'id{i}', f'chunk {i} for {b[:15]}', 'F') for i in range(3)]
def fake_sistema_rag(b): return b.replace('reports', 'presents')
def fake_recuperar(b, k=10): return [] if 'cat' in b else [('expr', [(1.0, 'x', 'term chunk ' + b[:10], 'F')])]
def fake_augment(b, ctx): return b + ' [aug]'
def fake_tools(b, d=None): print('  → called x'); return b + ' Next appointment.'
def fake_ragas(borrador, nota, chunks): return ({'context_recall': len(chunks) % 2, 'context_precision': 0.5,
                                                'faithfulness': None if nota == borrador else 1, 'answer_relevancy': None if nota == borrador else 0},
                                               {'context_recall': 'x'})
with tempfile.TemporaryDirectory() as carpeta:
    g83 = {'torch': torch, 'eval_set': eval_set, 'sistema_baseline': fake_baseline, 'sistema_finetuned': fake_finetuned,
           'generar': lambda *a, **k: 'OK', 'sistema_rag': fake_sistema_rag, 'buscar_semantica': fake_buscar, 'K': K,
           'recuperar_avanzado': fake_recuperar, '_augment_generate': fake_augment, 'responder_con_tools': fake_tools}
    run(cell(83).replace('/content/', carpeta + '/'), g83)
    resultados_nb = json.load(open(Path(carpeta) / 'resultados_sistemas.json'))

    config.settings()['paths']['outputs'] = carpeta + '/nuevo'
    noop = lambda: None
    comparacion = ComparacionSistemas(
        baseline=types.SimpleNamespace(resumir=fake_baseline, liberar=noop),
        finetuned=types.SimpleNamespace(resumir=fake_finetuned, liberar=noop),
        basic_rag=types.SimpleNamespace(sistema_rag_detallado=lambda b: (fake_sistema_rag(b), [t for _, _, t, _ in fake_buscar(b, K)]),
                                        instruct=types.SimpleNamespace(generar=lambda *a: 'OK', liberar=noop)),
        advanced_rag=types.SimpleNamespace(
            sistema_rag_avanzado_detallado=lambda b: ((fake_augment(b, fake_recuperar(b)) if fake_recuperar(b) else b),
                                                      [t for _, ch in fake_recuperar(b) for _, _, t, _ in ch]),
            reranker=types.SimpleNamespace(liberar=noop)),
        tool_agent=types.SimpleNamespace(responder_con_tools_detallado=lambda b, d: (fake_tools(b, d), [{'tool': 'x'}])))
    with contextlib.redirect_stdout(io.StringIO()):
        resultados = comparacion.generar(eval_set)
    sin_tiempos = lambda res: {s: [{k: v for k, v in f.items() if k not in ('segundos', 'log')} for f in filas] for s, filas in res.items()}
    ok('generar: notes, drafts and chunks of the 5 systems == cell 83', sin_tiempos(resultados) == sin_tiempos(resultados_nb))
    ok('generar: RAG latency includes the draft time', all(r['segundos'] >= f['segundos'] for r, f in zip(resultados['rag_base'], resultados['finetuned'])))
    ok('generar: resultados_sistemas.json saved and reloadable', comparacion.cargar_resultados() == json.loads(json.dumps(resultados)))

    # cell 88 on the SAME resultados (notebook's ragas there took 4 args: the 4th was unused by the Flow version)
    json.dump(resultados, open(Path(carpeta) / 'resultados_sistemas.json', 'w'))
    import medsummary.core.evaluation.harness as modulo_harness
    g88 = {'classic_metrics': g['classic_metrics'], 'juez_completo': fake_juez_completo, 'juez_tok': 1, 'juez_model': 1,
           'eval_set': eval_set, 'ragas': lambda b, n, c, ref: fake_ragas(b, n, c), 'JUEZ_MODEL': 'x', 'device': 'mps'}
    salida_88 = run(cell(88).replace('/content/', carpeta + '/'), g88)
    ragas_eval = types.SimpleNamespace(ragas=fake_ragas)
    detalle, scorecard = comparacion.evaluar(resultados, eval_set, harness, ragas_eval)
    sin_entradas = {s: [{k: v for k, v in f.items() if k not in ('input', 'esperado')} for f in filas] for s, filas in detalle.items()}
    ok('evaluar: per-example detail == cell 88 (plus input/esperado)', sin_entradas == g88['detalle'])
    ok('evaluar: every row carries its input and expected note',
       all(f['input'] == eval_set[f['ejemplo']]['input'] and f['esperado'] == eval_set[f['ejemplo']]['esperado']
           for filas in detalle.values() for f in filas))
    ok('evaluar: scorecard (means without None) == cell 88', scorecard == g88['scorecard'])
    tabla_nb = salida_88[salida_88.index('=' * 10) - 1:salida_88.index('Guardado:')].strip('\n')
    ok('tabla == cell 88 printout', ComparacionSistemas.tabla(scorecard) == tabla_nb)
    ok('scorecard_sistemas.csv == cell 88', (Path(carpeta) / 'nuevo' / 'scorecard_sistemas.csv').read_text() == (Path(carpeta) / 'scorecard_sistemas.csv').read_text())

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
