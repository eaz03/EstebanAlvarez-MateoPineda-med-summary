"""ComparacionSistemas: generates the notes of the 5 systems on the eval set and evaluates them with classic metrics,
the 3 judges, RAGAS (RAG systems only) and latency (notebook cells 83 and 88)."""
import json
import time

from medsummary.core import config
from medsummary.core.evaluation.harness import Harness
from medsummary.core.evaluation.ragas import RagasEvaluator
from medsummary.core.evaluation.tablas import METRICAS, escribir_scorecard_sistemas, scorecard_por_sistema
from medsummary.core.models.baseline import BaselineModel
from medsummary.core.models.finetuned import FineTunedModel
from medsummary.core.rag.advanced_rag import AdvancedRAG
from medsummary.core.rag.basic_rag import BasicRAG
from medsummary.core.rag.tools import ToolAgent

def _medir(funcion, *argumentos):
    """Runs funcion(*argumentos) and returns (result, seconds)."""
    inicio = time.perf_counter()
    resultado = funcion(*argumentos)
    return resultado, time.perf_counter() - inicio


class ComparacionSistemas:

    def __init__(self, baseline: BaselineModel, finetuned: FineTunedModel, basic_rag: BasicRAG,
                 advanced_rag: AdvancedRAG, tool_agent: ToolAgent):
        self.baseline = baseline
        self.finetuned = finetuned
        self.basic_rag = basic_rag
        self.advanced_rag = advanced_rag
        self.tool_agent = tool_agent
        self.sistemas_evaluar = config.settings()['comparacion']['sistemas_evaluar']   # notebook: SISTEMAS_EVALUAR
        self.carpeta = config.path('outputs')

    def generar(self, eval_set: list[dict]) -> dict:
        """Notes of the 5 systems + latency; saves resultados_sistemas.json.
        NOTE vs notebook: the baseline runs first for every example and is freed before the others, because it is now
        a separate model (the notebook shared one model with the fine-tuned one). The notes are the same."""
        resultados = {'baseline': [], 'finetuned': [], 'rag_base': [], 'rag_avanzado': [], 'tools': []}

        # 1) Baseline: Flan-T5 without adapter. Warm-up first: the first call is slower and must not count.
        self.baseline.resumir(eval_set[0]['input'])
        for ejemplo in eval_set:
            nota, segundos = _medir(self.baseline.resumir, ejemplo['input'])
            resultados['baseline'].append({'nota': nota, 'segundos': segundos, 'chunks': None})
        self.baseline.liberar()

        self.finetuned.resumir(eval_set[0]['input'])                       # warm-up
        self.basic_rag.instruct.generar('Answer OK.', 'OK', 5)               # warm-up
        for ejemplo in eval_set:
            dialogo = ejemplo['input']

            # 2) Fine-tuned: its note is the draft that the next three systems receive
            borrador, segundos_borrador = _medir(self.finetuned.resumir, dialogo)
            resultados['finetuned'].append({'nota': borrador, 'segundos': segundos_borrador, 'chunks': None})

            # 3) RAG base: semantic search + Qwen rewrites the draft
            (nota, chunks), segundos = _medir(self.basic_rag.sistema_rag_detallado, borrador)
            resultados['rag_base'].append({'nota': nota, 'segundos': segundos_borrador + segundos,
                                           'chunks': chunks, 'borrador': borrador})

            # 4) RAG avanzado: decomposition -> hybrid search -> rerank -> term substitution
            (nota, chunks), segundos = _medir(self.advanced_rag.sistema_rag_avanzado_detallado, borrador)
            resultados['rag_avanzado'].append({'nota': nota, 'segundos': segundos_borrador + segundos,
                                               'chunks': chunks, 'borrador': borrador})

            # 5) Tools: the list of tool calls is kept in 'log'
            (nota, llamadas), segundos = _medir(self.tool_agent.responder_con_tools_detallado, borrador, dialogo)
            resultados['tools'].append({'nota': nota, 'segundos': segundos_borrador + segundos,
                                        'chunks': None, 'log': llamadas})

        self.carpeta.mkdir(parents=True, exist_ok=True)
        with open(self.carpeta / 'resultados_sistemas.json', 'w', encoding='utf-8') as archivo:
            json.dump(resultados, archivo, ensure_ascii=False, indent=2)

        # The GPU must be free for the judge: T5, Qwen and the reranker are released.
        self.finetuned.liberar()
        self.basic_rag.instruct.liberar()
        self.advanced_rag.reranker.liberar()
        return resultados

    def cargar_resultados(self) -> dict:
        """Reads resultados_sistemas.json from disk (useful after restarting, as in the notebook)."""
        with open(self.carpeta / 'resultados_sistemas.json', encoding='utf-8') as archivo:
            return json.load(archivo)

    @staticmethod
    def evaluar_filas(filas: list[dict], eval_set: list[dict], harness: Harness,
                      ragas_evaluator: RagasEvaluator, al_evaluar=None) -> list[dict]:
        """Classic metrics + 3 judges for every note, and RAGAS for the RAG notes (cells 88 and 95).
        al_evaluar: optional function called with each evaluated row (e.g. to log it to W&B)."""
        evaluadas = []
        for numero, resultado in enumerate(filas):
            ejemplo = eval_set[numero]
            valores, _, extras = harness.evaluar_nota(ejemplo, resultado['nota'])
            fila = {'ejemplo': numero, 'input': ejemplo['input'], 'esperado': ejemplo['esperado'],
                    'nota': resultado['nota'], 'segundos': resultado['segundos']}
            if 'segundos_rag' in resultado:                                  # quantization and ablation runs
                fila['segundos_rag'] = resultado['segundos_rag']
            fila['rougeL'] = valores['rougeL']
            fila['bert_f1'] = valores['bert_f1']
            for juez in harness.judge.rubricas:
                fila[f'juez_{juez}'] = valores[f'juez_{juez}']
            fila['juez_final'] = valores['juez_final']
            for juez in harness.judge.rubricas:
                fila[f'explicacion_{juez}'] = extras[f'explicacion_{juez}']
            fila['acierto'] = harness.es_acierto(valores['bert_f1'], valores['juez_final'])

            if resultado['chunks'] is not None:                              # only the RAG systems
                puntajes, explicaciones = ragas_evaluator.ragas(resultado['borrador'], resultado['nota'],
                                                                resultado['chunks'])
                fila.update(puntajes)
                fila['explicaciones_ragas'] = explicaciones
            evaluadas.append(fila)
            if al_evaluar is not None:
                al_evaluar(fila)
        return evaluadas

    def evaluar(self, resultados: dict, eval_set: list[dict], harness: Harness,
                ragas_evaluator: RagasEvaluator) -> tuple[dict, dict]:
        """Returns (detalle, scorecard); saves detalle_sistemas.json and scorecard_sistemas.csv."""
        detalle = {}
        for sistema in self.sistemas_evaluar:
            detalle[sistema] = self.evaluar_filas(resultados[sistema], eval_set, harness, ragas_evaluator)

        scorecard = scorecard_por_sistema(detalle)     # mean of each metric, without the ones that do not apply
        self.carpeta.mkdir(parents=True, exist_ok=True)
        with open(self.carpeta / 'detalle_sistemas.json', 'w', encoding='utf-8') as archivo:
            json.dump(detalle, archivo, ensure_ascii=False, indent=2)
        escribir_scorecard_sistemas(scorecard, self.carpeta / 'scorecard_sistemas.csv')
        return detalle, scorecard

    @staticmethod
    def tabla(scorecard: dict) -> str:
        encabezado = f'{"Métrica":<34}' + ''.join(f'{sistema:>14}' for sistema in scorecard)
        lineas = ['=' * len(encabezado), encabezado, '-' * len(encabezado)]
        for nombre, clave in METRICAS:
            linea = f'{nombre:<34}'
            for sistema in scorecard:
                valor = scorecard[sistema][clave]
                linea += f'{"-":>14}' if valor is None else f'{valor:>14.2f}'
            lineas.append(linea)
        lineas += ['=' * len(encabezado), '"-" = no aplica.   * Solo en los casos en que el RAG cambió la nota.']
        return '\n'.join(lineas)
