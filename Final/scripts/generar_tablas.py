"""Regenerates the final tables on this machine, following the Colab notebook step by step:
  1. Systems (cells 83, 88): baseline, fine-tuned, RAG base, RAG avanzado (+ tools: generated, not evaluated).
  2. Quantization (cells 91 to 93): the advanced RAG with Qwen unquantized, GGUF Q8_0 and GGUF Q4_K_M, same drafts.
  3. Ablation (cell 94): the advanced RAG (unquantized Qwen) without reranker, without hybrid search, without both.
  4. Evaluation of every note with the same harness: classic metrics, 3 Flow judges, RAGAS (cells 88, 95).
  5. Tables (cells 38, 96, 97, 98) in metricas/.
Each step saves its results in metricas/; if that file already exists the step is skipped (delete it to redo it).
Progress, latencies, scores and the final tables go to W&B (only numbers; see medsummary/observabilidad.py).

Run from the MedSummary folder:
    uv run python scripts/generar_tablas.py
"""
import json
import os
import platform
import time

import torch

from medsummary import observabilidad
from medsummary.core import config
from medsummary.core.data import cargar_eval_set
from medsummary.core.evaluation import ComparacionSistemas, Harness, RagasEvaluator, tablas
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge, fijar_semilla
from medsummary.core.models.quantized import QuantizedInstructModel
from medsummary.core.rag import AdvancedRAG, BasicRAG, BM25Retriever, Corpus, Reranker, SemanticRetriever, ToolAgent

CARPETA = config.path('outputs')                    # metricas/
VARIANTES = ['sin_cuantizar', 'q8_0', 'q4_k_m']     # before and after quantizing Qwen (deployed: q4_k_m)
ABLACIONES = {                                      # name: (usar_hibrida, usar_reranker)
    'completo': (True, True),
    'sin_reranker': (True, False),
    'sin_hibrida': (False, True),
    'sin_reranker_ni_hibrida': (False, False),
}


def existe(nombre):
    return (CARPETA / nombre).is_file()

def leer(nombre):
    return json.loads((CARPETA / nombre).read_text(encoding='utf-8'))

def guardar(nombre, contenido):
    CARPETA.mkdir(parents=True, exist_ok=True)
    (CARPETA / nombre).write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding='utf-8')


def sincronizar():
    """The GPU finishes its work before the clock stops (notebook: torch.cuda.synchronize)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

def bytes_modulo_gb(modulo):
    """Size of a torch model's weights: parameters + buffers (cell 91)."""
    total = sum(p.numel() * p.element_size() for p in modulo.parameters())
    total += sum(b.numel() * b.element_size() for b in modulo.buffers())
    return total / 1e9

def vram_usada_gb():
    """GPU memory in use according to the driver (cuda only; on a Mac the memory is shared with the CPU)."""
    libre, total = torch.cuda.mem_get_info()
    return (total - libre) / 1e9


def correr_rag_avanzado(nombre, rag, borradores, segundos_borrador):
    """Advanced RAG on every draft. segundos = draft + RAG, as in the scorecard (cell 92)."""
    filas = []
    for numero, borrador in enumerate(borradores):
        sincronizar()
        inicio = time.perf_counter()
        nota, chunks = rag.sistema_rag_avanzado_detallado(borrador)
        sincronizar()
        segundos_rag = time.perf_counter() - inicio
        filas.append({'nota': nota, 'borrador': borrador, 'chunks': chunks, 'segundos_rag': segundos_rag,
                      'segundos': segundos_borrador[numero] + segundos_rag})
        print(f'  ejemplo {numero}: {segundos_rag:.1f} s')
        observabilidad.registrar({f'segundos_rag/{nombre}': segundos_rag, 'ejemplo': numero})
    return filas

def calentar(rag, borrador):
    """First call of each model: slower, must not count in the latency."""
    rag.instruct.generar('Answer OK.', 'OK', 5)
    rag.recuperar_avanzado(borrador)


# ---------- 1. the systems (cells 83 and 88) ----------
def paso_1_sistemas(eval_set, semantic, bm25, reranker):
    qwen = InstructModel()
    advanced_rag = AdvancedRAG(semantic, bm25, reranker, qwen)
    comparacion = ComparacionSistemas(BaselineModel(), FineTunedModel(), BasicRAG(semantic, qwen), advanced_rag,
                                      ToolAgent(advanced_rag, qwen))
    resultados = comparacion.generar(eval_set)              # saves resultados_sistemas.json and frees the models
    for sistema, filas in resultados.items():
        for numero, fila in enumerate(filas):
            observabilidad.registrar({f'segundos/{sistema}': fila['segundos'], 'ejemplo': numero})


# ---------- 2. before and after quantizing Qwen (cells 91 to 93) ----------
def paso_2_cuantizacion(borradores, segundos_borrador, semantic, bm25, reranker):
    finetuned = FineTunedModel().cargar()      # only to measure the memory of the other pieces
    dispositivo = finetuned.device
    memoria_otras_gb = (bytes_modulo_gb(finetuned.model) + bytes_modulo_gb(semantic.cargar().model)
                        + bytes_modulo_gb(reranker.cargar().model.model))      # T5 + embedder + reranker
    finetuned.liberar()
    maquina = f'{platform.machine()} | {os.cpu_count()} hilos | {dispositivo}'

    resultados = {}
    for variante in VARIANTES:
        print('=' * 70, f'\nQwen {variante} | {dispositivo}')
        vram_antes = vram_usada_gb() if dispositivo == 'cuda' else None
        if variante == 'sin_cuantizar':
            qwen = InstructModel().cargar()
        else:
            qwen = QuantizedInstructModel(variante).cargar()
        rag = AdvancedRAG(semantic, bm25, reranker, qwen)
        calentar(rag, borradores[0])
        filas = correr_rag_avanzado(f'{dispositivo}_{variante}', rag, borradores, segundos_borrador)
        vram_extra_gb = vram_usada_gb() - vram_antes if dispositivo == 'cuda' else None
        resultados[f'{dispositivo}_{variante}'] = {
            'dispositivo': dispositivo, 'variante': variante, 'filas': filas,
            'memoria_qwen_gb': qwen.memoria_gb(), 'vram_extra_gb': vram_extra_gb,
            'memoria_otras_piezas_gb': memoria_otras_gb, 'formato': qwen.formato(),
            'segundos_borrador': segundos_borrador, 'maquina': maquina}
        qwen.liberar()
    guardar('resultados_cuantizacion.json', resultados)


# ---------- 3. ablation of the advanced RAG, unquantized Qwen (cell 94) ----------
def paso_3_ablacion(borradores, segundos_borrador, semantic, bm25, reranker):
    qwen = InstructModel().cargar()
    guardar('memoria_qwen_referencia.json', {'memoria_qwen_gb': qwen.memoria_gb(), 'dispositivo': qwen.device})
    resultados = {}
    for nombre, (usar_hibrida, usar_reranker) in ABLACIONES.items():
        print('=' * 70, f'\n{nombre} | hybrid: {usar_hibrida} | reranker: {usar_reranker}')
        rag = AdvancedRAG(semantic, bm25, reranker, qwen, usar_hibrida=usar_hibrida, usar_reranker=usar_reranker)
        if nombre == 'completo':
            calentar(rag, borradores[0])
        resultados[nombre] = correr_rag_avanzado(nombre, rag, borradores, segundos_borrador)
    qwen.liberar()
    guardar('resultados_ablacion.json', resultados)


# ---------- 4. evaluation with the same harness (cells 88 and 95) ----------
def paso_4_evaluar(eval_set, harness, ragas_evaluator):
    def registrar_fila(nombre, fila):
        """One W&B point per evaluated note: only its numbers (scores, seconds)."""
        numeros = observabilidad.solo_numeros(fila)
        observabilidad.registrar({f'{clave}/{nombre}': valor for clave, valor in numeros.items()})

    def evaluar(nombre, filas):
        return ComparacionSistemas.evaluar_filas(filas, eval_set, harness, ragas_evaluator,
                                                 al_evaluar=lambda fila: registrar_fila(nombre, fila))

    if not existe('detalle_sistemas.json'):
        resultados = leer('resultados_sistemas.json')
        sistemas = config.settings()['comparacion']['sistemas_evaluar']      # tools is not evaluated, as in cell 88
        detalle = {sistema: evaluar(sistema, resultados[sistema]) for sistema in sistemas}
        guardar('detalle_sistemas.json', detalle)
        tablas.escribir_scorecard_sistemas(tablas.scorecard_por_sistema(detalle), CARPETA / 'scorecard_sistemas.csv')

    if not existe('detalle_eficiencia.json'):
        ya_evaluadas = {}       # two lists with exactly the same notes are evaluated once (the judge is deterministic)

        def evaluar_una_vez(nombre, filas):
            clave = json.dumps([f['nota'] for f in filas], ensure_ascii=False)
            if clave in ya_evaluadas:
                print(f'{nombre}: mismas notas que {ya_evaluadas[clave][0]}, se reutiliza su evaluación')
                reutilizadas = [dict(evaluada, segundos=f['segundos'], segundos_rag=f['segundos_rag'])
                                for evaluada, f in zip(ya_evaluadas[clave][1], filas)]
                for fila in reutilizadas:
                    registrar_fila(nombre, fila)
                return reutilizadas
            print(f'Evaluando {nombre} ...')
            evaluadas = evaluar(nombre, filas)
            ya_evaluadas[clave] = (nombre, evaluadas)
            return evaluadas

        detalle_eficiencia = {'cuantizacion': {}, 'ablacion': {}}
        for clave, datos in leer('resultados_cuantizacion.json').items():
            detalle_eficiencia['cuantizacion'][clave] = evaluar_una_vez(f'qwen_{clave}', datos['filas'])
        for nombre, filas in leer('resultados_ablacion.json').items():
            detalle_eficiencia['ablacion'][nombre] = evaluar_una_vez(f'ablacion_{nombre}', filas)
        guardar('detalle_eficiencia.json', detalle_eficiencia)


# ---------- 5. tables (cells 38, 96, 97, 98) ----------
def paso_5_tablas(eval_set, harness):
    detalle_sistemas = leer('detalle_sistemas.json')
    detalle_eficiencia = leer('detalle_eficiencia.json')
    resultados_cuantizacion = leer('resultados_cuantizacion.json')

    scorecard, dispositivos = tablas.scorecard_eficiencia(detalle_sistemas, detalle_eficiencia['cuantizacion'],
                                                          resultados_cuantizacion, leer('memoria_qwen_referencia.json'))
    tablas.escribir_scorecard_eficiencia(scorecard, dispositivos, CARPETA / 'scorecard_eficiencia.csv')
    tablas.escribir_tabla_cuantizacion(tablas.tabla_cuantizacion(detalle_eficiencia['cuantizacion'], resultados_cuantizacion),
                                       CARPETA / 'tabla_cuantizacion.csv')
    tablas.escribir_tabla_ablacion(tablas.tabla_ablacion(detalle_eficiencia['ablacion'], detalle_sistemas['finetuned']),
                                   CARPETA / 'tabla_ablacion.csv')
    print(ComparacionSistemas.tabla(tablas.scorecard_por_sistema(detalle_sistemas)))

    if not existe('scorecard.csv'):            # Module 2 harness: baseline vs fine-tuned (cell 38)
        baseline = BaselineModel()
        scorecard_baseline = harness.harness(eval_set, baseline.resumir)
        baseline.liberar()
        finetuned = FineTunedModel()
        scorecard_finetuned = harness.harness(eval_set, finetuned.resumir)
        finetuned.liberar()
        scorecards = {'baseline': scorecard_baseline, 'finetuned': scorecard_finetuned}
        harness.guardar(scorecards)
        print(Harness.tabla(scorecards))
    observabilidad.registrar_tablas(sorted(CARPETA.glob('*.csv')))
    print(f'Tablas en {CARPETA}')


def main():
    settings = config.settings()
    fijar_semilla(settings['seed'])
    observabilidad.iniciar_wandb('tablas', {'seed': settings['seed'], 'modelos': settings['models'],
                                            'variantes': VARIANTES, 'ablaciones': list(ABLACIONES),
                                            'metricas': settings['metricas'], 'harness': settings['harness']})
    eval_set = cargar_eval_set()
    pasos_de_generacion = ['resultados_sistemas.json', 'resultados_cuantizacion.json', 'resultados_ablacion.json']

    if not all(existe(nombre) for nombre in pasos_de_generacion):
        corpus = Corpus.desde_pdf()
        semantic, bm25, reranker = SemanticRetriever(corpus), BM25Retriever(corpus), Reranker()
        if not existe('resultados_sistemas.json'):
            paso_1_sistemas(eval_set, semantic, bm25, reranker)
        resultados_sistemas = leer('resultados_sistemas.json')
        # Same drafts (fine-tuned note) as the RAG systems, and how long each one took (cell 91)
        borradores = [fila['borrador'] for fila in resultados_sistemas['rag_avanzado']]
        segundos_borrador = [fila['segundos'] for fila in resultados_sistemas['finetuned']]
        if not existe('resultados_cuantizacion.json'):
            paso_2_cuantizacion(borradores, segundos_borrador, semantic, bm25, reranker)
        if not existe('resultados_ablacion.json'):
            paso_3_ablacion(borradores, segundos_borrador, semantic, bm25, reranker)
            # Control (cell 94): with everything on, the ablation must give the same notes as the advanced RAG
            iguales = sum(a['nota'] == b['nota'] for a, b in zip(resultados_sistemas['rag_avanzado'],
                                                                 leer('resultados_ablacion.json')['completo']))
            print(f'Control: "completo" coincide con rag_avanzado en {iguales}/{len(eval_set)} notas.')
        semantic.liberar()
        reranker.liberar()

    judge = Judge()                            # loaded only now, when the generators are already freed
    harness = Harness(judge)
    paso_4_evaluar(eval_set, harness, RagasEvaluator(judge))
    paso_5_tablas(eval_set, harness)
    observabilidad.cerrar()


if __name__ == '__main__':
    main()
