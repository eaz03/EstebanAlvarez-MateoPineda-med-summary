"""The three comparison tables, read from the CSVs already computed by scripts/generar_tablas.py (metricas/).
Nothing is measured here: they are only read and ordered (ported from Entrega_final/medsummary/metricas.py).
- scorecard_eficiencia.csv: the full harness + p50/p95 latency of every system and every version of Qwen.
- tabla_cuantizacion.csv:  the memory of every version of Qwen (weights, extra VRAM and system total).
- tabla_ablacion.csv:      the advanced RAG without reranker, without hybrid and without both."""
import csv

from medsummary.core import config

# Harness dimensions (Module 2): classic metrics, the three judges + the weighted one and the domain hits.
HARNESS = [('rougeL', 'ROUGE-L', 3), ('bert_f1', 'BERTScore F1', 3), ('juez_grounding', 'Juez grounding (1-5)', 2),
           ('juez_coverage', 'Juez coverage (1-5)', 2), ('juez_form', 'Juez form (1-5)', 2),
           ('juez_final', 'Juez final ponderado (1-5)', 2), ('acierto', 'Aciertos de dominio', 2)]
ABREVIATURA = {'sin cuantizar': 'sinq', 'Q8_0': 'q8', 'Q4_K_M': 'q4'}      # version in tabla_cuantizacion -> scorecard column
ABREVIATURA_VARIANTE = {'sin_cuantizar': 'sinq', 'q8_0': 'q8', 'q4_k_m': 'q4'}  # settings.yaml -> scorecard column


def variante_desplegada() -> str:
    return config.settings()['despliegue']['variante']


def _numero(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _leer_csv(nombre):
    with open(config.path('outputs') / nombre, encoding='utf-8') as archivo:
        return list(csv.reader(archivo))


def _leer_scorecard():
    """scorecard_eficiencia.csv has one row per metric and one column per system: flipped to {system: {metric: value}}."""
    filas = _leer_csv('scorecard_eficiencia.csv')
    sistemas = filas[0][1:]
    tabla = {sistema: {} for sistema in sistemas}
    for fila in filas[1:]:
        for sistema, valor in zip(sistemas, fila[1:]):
            if fila[0] == 'dispositivo_latencia':
                tabla[sistema][fila[0]] = valor
            else:
                tabla[sistema][fila[0]] = _numero(valor)
    return tabla


def _leer_csv_dict(nombre):
    filas = _leer_csv(nombre)
    return [dict(zip(filas[0], fila)) for fila in filas[1:]]


def tabla_cuantizacion(scorecard):
    desplegada = ABREVIATURA_VARIANTE[variante_desplegada()]
    columnas = [('version', 'Versión', None), ('dispositivo', 'Dispositivo', None), *HARNESS,
                ('latencia_p50', 'Latencia p50 (s)', 2), ('latencia_p95', 'Latencia p95 (s)', 2),
                ('memoria_qwen_gb', 'Memoria Qwen (GB)', 2), ('vram_extra_gb', 'VRAM extra (GB)', 2),
                ('memoria_total_gb', 'Memoria total del sistema (GB)', 2)]
    filas = []
    for fila in _leer_csv_dict('tabla_cuantizacion.csv'):
        abreviatura = next(a for texto, a in ABREVIATURA.items() if texto in fila['version'])
        sistema = scorecard[f"avz_{fila['dispositivo']}_{abreviatura}"]
        version = fila['version']
        if abreviatura == desplegada:
            version += ' · desplegada'
        filas.append({'version': version,
                      'dispositivo': fila['dispositivo'],
                      **{clave: sistema[clave] for clave, _, _ in HARNESS},
                      'latencia_p50': sistema['latencia_p50'], 'latencia_p95': sistema['latencia_p95'],
                      'memoria_qwen_gb': _numero(fila['memoria_qwen_gb']), 'vram_extra_gb': _numero(fila['vram_extra_gb']),
                      'memoria_total_gb': _numero(fila['memoria_total_gb'])})
    return {'titulo': 'Antes y después de cuantizar Qwen (sistema completo: borrador + RAG avanzado)',
            'nota': 'Mismo eval set (10 ejemplos) y mismo harness, misma máquina. Latencia = borrador del modelo ajustado + '
                    'RAG avanzado. Con 10 ejemplos el p95 es casi el máximo. Memoria Qwen = pesos (GGUF: tamaño del archivo).',
            'columnas': columnas, 'filas': filas}


def tabla_sistemas(scorecard):
    variante = variante_desplegada()
    dispositivo = next(s['dispositivo_latencia'] for nombre, s in scorecard.items() if nombre.startswith('avz_'))
    sistemas = [('finetuned', 'Modelo ajustado (Flan-T5 + LoRA)'),
                ('rag_base', 'RAG ingenuo (Qwen sin cuantizar)'),
                ('rag_avanzado', 'RAG avanzado (Qwen sin cuantizar)'),
                (f'avz_{dispositivo}_{ABREVIATURA_VARIANTE[variante]}',
                 f'RAG avanzado con Qwen {variante.upper()} · desplegado')]
    columnas = [('sistema', 'Sistema', None), ('dispositivo_latencia', 'Dispositivo', None), *HARNESS,
                ('latencia_p50', 'Latencia mediana p50 (s)', 2)]
    filas = [{'sistema': nombre, **{clave: scorecard[s][clave] for clave, _, _ in columnas[1:]}} for s, nombre in sistemas]
    return {'titulo': 'Comparación de sistemas: modelo ajustado, RAG ingenuo, RAG avanzado y cuantizado',
            'nota': 'Mismo eval set y mismo harness. Los RAG reciben el borrador del modelo ajustado; su latencia incluye la del borrador.',
            'columnas': columnas, 'filas': filas}


def tabla_ablacion():
    columnas = [('configuracion', 'Configuración', None), ('juez_final', 'Juez final (1-5)', 2), ('bert_f1', 'BERTScore F1', 3),
                ('acierto', 'Aciertos de dominio', 2), ('context_recall', 'RAGAS context recall', 2),
                ('context_precision', 'RAGAS context precision', 2), ('faithfulness', 'RAGAS faithfulness *', 2),
                ('answer_relevancy', 'RAGAS answer relevancy *', 2), ('delta_juez', 'Δ juez vs. completo', 3),
                ('p50_rag_s', 'p50 RAG (s)', 2), ('p95_rag_s', 'p95 RAG (s)', 2), ('notas_distintas', 'Notas distintas (de 10)', 0)]
    filas = []
    for fila in _leer_csv_dict('tabla_ablacion.csv'):
        filas.append({clave: (fila[clave] if clave == 'configuracion' else _numero(fila[clave])) for clave, _, _ in columnas})
    return {'titulo': 'Ablación del RAG avanzado (Qwen sin cuantizar)',
            'nota': 'Δ juez = juez final menos el del sistema completo. p50/p95 RAG = solo recuperar y reescribir (sin el borrador). '
                    'Notas distintas = cuántas notas cambian frente al sistema completo. * Solo cuando el RAG cambió la nota. '
                    'Sin reranker se toman los 3 primeros del retriever por expresión.',
            'columnas': columnas, 'filas': filas}


def tablas() -> list[dict]:
    scorecard = _leer_scorecard()
    resultado = []
    for tabla in (tabla_cuantizacion(scorecard), tabla_sistemas(scorecard), tabla_ablacion()):
        tabla['columnas'] = [{'clave': c, 'nombre': n, 'decimales': d} for c, n, d in tabla['columnas']]
        resultado.append(tabla)
    return resultado
