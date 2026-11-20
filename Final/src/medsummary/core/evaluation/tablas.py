"""Tables of the final report, built from evaluated rows (one dict per example, as in detalle_*.json) and written
as CSV in the exact formats of the notebook (cells 88, 96, 97, 98), which the metrics page reads.
Pure functions: they do not run any model."""
import csv

import numpy as np

# Per-system metrics (notebook cell 88)
METRICAS = [('ROUGE-L', 'rougeL'),
            ('BERTScore F1', 'bert_f1'),
            ('Juez grounding (1-5)', 'juez_grounding'),
            ('Juez coverage (1-5)', 'juez_coverage'),
            ('Juez form (1-5)', 'juez_form'),
            ('Juez final ponderado (1-5)', 'juez_final'),
            ('RAGAS context recall', 'context_recall'),
            ('RAGAS context precision', 'context_precision'),
            ('RAGAS faithfulness *', 'faithfulness'),
            ('RAGAS answer relevancy *', 'answer_relevancy'),
            ('Latencia promedio (s)', 'segundos'),
            ('Aciertos de dominio (fracción)', 'acierto')]
# Extra rows of the efficiency scorecard (cell 97)
FILAS_EXTRA = [('Latencia mediana p50 (s)', 'latencia_p50'), ('Latencia p95 (s)', 'latencia_p95'),
               ('Memoria de Qwen (GB)', 'memoria_qwen_gb')]
ABREVIATURA = {'sin_cuantizar': 'sinq', 'q8_0': 'q8', 'q4_k_m': 'q4'}
NOMBRES_VERSION = {'sin_cuantizar': 'Antes: Qwen sin cuantizar', 'q8_0': 'Después: Qwen GGUF Q8_0 (~int8)',
                   'q4_k_m': 'Después: Qwen GGUF Q4_K_M (~int4)'}
NOMBRES_ABLACION = {'completo': 'Sistema completo (hybrid + reranker)', 'sin_reranker': 'Sin reranker (solo hybrid)',
                    'sin_hibrida': 'Sin hybrid (semántica + reranker)',
                    'sin_reranker_ni_hibrida': 'Sin reranker ni hybrid (solo semántica)'}
COLUMNAS_ABLACION = ['juez_final', 'bert_f1', 'acierto', 'context_recall', 'context_precision', 'faithfulness',
                     'answer_relevancy']


def promedio(filas: list[dict], clave: str) -> float | None:
    """Mean of a metric, without the rows where it does not apply (None)."""
    valores = [fila[clave] for fila in filas if fila.get(clave) is not None]
    return sum(valores) / len(valores) if valores else None


def percentiles(valores: list[float]) -> dict:
    return {'p50': float(np.percentile(valores, 50)), 'p95': float(np.percentile(valores, 95))}


# ---------- systems scorecard (cell 88) ----------
def scorecard_por_sistema(detalle: dict[str, list[dict]]) -> dict:
    return {sistema: {clave: promedio(filas, clave) for _, clave in METRICAS} for sistema, filas in detalle.items()}


def escribir_scorecard_sistemas(scorecard: dict, ruta) -> None:
    with open(ruta, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(['metrica'] + list(scorecard))
        for _, clave in METRICAS:
            escritor.writerow([clave] + [scorecard[sistema][clave] for sistema in scorecard])


# ---------- efficiency scorecard: systems + quantized columns, with p50/p95 and memory (cell 97) ----------
def scorecard_eficiencia(detalle_sistemas: dict, cuantizacion_evaluada: dict, resultados_cuantizacion: dict,
                         memoria_referencia: dict) -> tuple[dict, dict]:
    """Returns (scorecard, device of each column). Columns: baseline, finetuned, rag_base, rag_avanzado, avz_<device>_<version>."""
    sistemas = {nombre: detalle_sistemas[nombre] for nombre in ['baseline', 'finetuned', 'rag_base', 'rag_avanzado']}
    clave_de_columna = {}
    for clave, datos in resultados_cuantizacion.items():
        columna = f'avz_{datos["dispositivo"]}_{ABREVIATURA[datos["variante"]]}'
        clave_de_columna[columna] = clave
        sistemas[columna] = cuantizacion_evaluada[clave]

    dispositivo_columna = {}
    scorecard = {}
    for sistema, filas in sistemas.items():
        if sistema in clave_de_columna:
            dispositivo_columna[sistema] = resultados_cuantizacion[clave_de_columna[sistema]]['dispositivo']
        else:
            dispositivo_columna[sistema] = memoria_referencia['dispositivo']
        scorecard[sistema] = {clave: promedio(filas, clave) for _, clave in METRICAS}
        latencias = percentiles([fila['segundos'] for fila in filas])
        scorecard[sistema]['latencia_p50'] = latencias['p50']
        scorecard[sistema]['latencia_p95'] = latencias['p95']
        if sistema in ('rag_base', 'rag_avanzado'):           # they use the unquantized Qwen
            scorecard[sistema]['memoria_qwen_gb'] = memoria_referencia['memoria_qwen_gb']
        elif sistema in clave_de_columna:
            scorecard[sistema]['memoria_qwen_gb'] = resultados_cuantizacion[clave_de_columna[sistema]]['memoria_qwen_gb']
        else:
            scorecard[sistema]['memoria_qwen_gb'] = None
    return scorecard, dispositivo_columna


def escribir_scorecard_eficiencia(scorecard: dict, dispositivo_columna: dict, ruta) -> None:
    with open(ruta, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(['metrica'] + list(scorecard))
        escritor.writerow(['dispositivo_latencia'] + [dispositivo_columna[s] for s in scorecard])
        for _, clave in METRICAS + FILAS_EXTRA:
            escritor.writerow([clave] + [scorecard[s][clave] for s in scorecard])


# ---------- before/after quantization (cell 96) ----------
def tabla_cuantizacion(cuantizacion_evaluada: dict, resultados_cuantizacion: dict) -> list[dict]:
    filas_tabla = []
    for clave, datos in resultados_cuantizacion.items():
        evaluadas = cuantizacion_evaluada[clave]
        latencias = percentiles([fila['segundos'] for fila in evaluadas])
        latencias_rag = percentiles([fila['segundos_rag'] for fila in evaluadas])
        version = NOMBRES_VERSION[datos['variante']]
        if datos['variante'] == 'sin_cuantizar':
            version += f' ({datos["formato"].replace("torch.", "")})'
        filas_tabla.append({'dispositivo': datos['dispositivo'], 'version': version,
                            'juez_final': promedio(evaluadas, 'juez_final'), 'bert_f1': promedio(evaluadas, 'bert_f1'),
                            'aciertos': promedio(evaluadas, 'acierto'),
                            'p50_s': latencias['p50'], 'p95_s': latencias['p95'], 'p50_rag_s': latencias_rag['p50'],
                            'memoria_qwen_gb': datos['memoria_qwen_gb'], 'vram_extra_gb': datos['vram_extra_gb'],
                            'memoria_total_gb': datos['memoria_qwen_gb'] + datos['memoria_otras_piezas_gb']})
    return filas_tabla


def escribir_tabla_cuantizacion(filas_tabla: list[dict], ruta) -> None:
    columnas = ['dispositivo', 'version', 'juez_final', 'bert_f1', 'aciertos', 'p50_s', 'p95_s', 'p50_rag_s',
                'memoria_qwen_gb', 'vram_extra_gb', 'memoria_total_gb']
    with open(ruta, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(columnas)
        for fila in filas_tabla:
            escritor.writerow([fila[c] for c in columnas])


# ---------- ablation of the advanced RAG (cell 98) ----------
def tabla_ablacion(ablacion_evaluada: dict, finetuned_evaluada: list[dict]) -> list[dict]:
    completo = ablacion_evaluada['completo']
    juez_completo = promedio(completo, 'juez_final')
    filas_tabla = []
    for nombre, evaluadas in ablacion_evaluada.items():
        latencias = percentiles([fila['segundos_rag'] for fila in evaluadas])
        fila = {'configuracion': NOMBRES_ABLACION[nombre], **{c: promedio(evaluadas, c) for c in COLUMNAS_ABLACION},
                'p50_rag_s': latencias['p50'], 'p95_rag_s': latencias['p95'],
                'notas_distintas': sum(a['nota'] != b['nota'] for a, b in zip(evaluadas, completo))}
        fila['delta_juez'] = fila['juez_final'] - juez_completo
        filas_tabla.append(fila)
    # Reference: the fine-tuned draft without any RAG
    filas_tabla.append({'configuracion': 'Referencia: sin RAG (fine-tuned)',
                        **{c: promedio(finetuned_evaluada, c) for c in COLUMNAS_ABLACION},
                        'p50_rag_s': None, 'p95_rag_s': None,
                        'notas_distintas': sum(a['nota'] != b['nota'] for a, b in zip(finetuned_evaluada, completo)),
                        'delta_juez': promedio(finetuned_evaluada, 'juez_final') - juez_completo})
    return filas_tabla


def escribir_tabla_ablacion(filas_tabla: list[dict], ruta) -> None:
    columnas = ['configuracion'] + COLUMNAS_ABLACION + ['delta_juez', 'p50_rag_s', 'p95_rag_s', 'notas_distintas']
    with open(ruta, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(columnas)
        for fila in filas_tabla:
            escritor.writerow([fila[c] for c in columnas])
