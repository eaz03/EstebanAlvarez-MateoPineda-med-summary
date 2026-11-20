"""Harness: evaluates any system (a function dialogue -> note) on the domain eval set in 3 dimensions
(notebook cells 36 and 38): 1) ROUGE and BERTScore, 2) the three Flow judges + weighted average, 3) domain hit."""
import csv
import json

from medsummary.core import config
from medsummary.core.evaluation.metrics import classic_metrics
from medsummary.core.models.judge import Judge

FILAS = [('1 · ROUGE-1 promedio', 'rouge1_promedio', '.3f'),
         ('1 · ROUGE-2 promedio', 'rouge2_promedio', '.3f'),
         ('1 · ROUGE-L promedio', 'rougeL_promedio', '.3f'),
         ('1 · BERTScore F1 promedio', 'bert_f1_promedio', '.3f'),
         ('2 · Juez grounding (1-5)', 'juez_grounding_promedio', '.2f'),
         ('2 · Juez coverage (1-5)', 'juez_coverage_promedio', '.2f'),
         ('2 · Juez form (1-5)', 'juez_form_promedio', '.2f'),
         ('2 · Juez final ponderado (1-5)', 'juez_final_promedio', '.2f')]


class Harness:

    def __init__(self, judge: Judge):
        settings_harness = config.settings()['harness']
        self.judge = judge
        self.umbral_bert = settings_harness['umbral_bert']     # notebook: UMBRAL_BERT
        self.umbral_juez = settings_harness['umbral_juez']     # notebook: UMBRAL_JUEZ

    def es_acierto(self, bert_f1: float, juez_final: float) -> bool:
        """Dimension 3: a domain hit if the note is semantically close to the reference OR the judges approve it."""
        return bert_f1 >= self.umbral_bert or juez_final >= self.umbral_juez

    def evaluar_nota(self, ejemplo: dict, nota: str) -> tuple[dict, dict, dict]:
        """Scores one note. Returns (metric values, judge results, explanations + review flags)."""
        rouge, bert = classic_metrics(nota, ejemplo['esperado'])                    # dimension 1
        juez_final, resultados_jueces = self.judge.juez_completo(ejemplo['input'], nota)   # dimension 2
        fila = {'rouge1': rouge['rouge1'], 'rouge2': rouge['rouge2'], 'rougeL': rouge['rougeL'],
                'bert_f1': bert['f1'][0]}
        for juez in resultados_jueces:
            fila[f'juez_{juez}'] = resultados_jueces[juez]['score']
        fila['juez_final'] = juez_final

        extras = {}
        for juez in resultados_jueces:
            extras[f'explicacion_{juez}'] = resultados_jueces[juez]['explanation']
        # judges whose answer could not be parsed (their score defaulted to 3)
        extras['revisar'] = [juez for juez in resultados_jueces if not resultados_jueces[juez]['valido']]
        return fila, resultados_jueces, extras

    def harness(self, eval_set: list[dict], sistema) -> dict:
        """sistema: any function dialogue -> note, e.g. baseline.resumir or finetuned.resumir."""
        detalle, filas = [], []
        for e in eval_set:
            resp = sistema(e['input'])
            fila, _, extras = self.evaluar_nota(e, resp)
            acierto = self.es_acierto(fila['bert_f1'], fila['juez_final'])      # dimension 3
            filas.append(fila)
            detalle.append({'input': e['input'], 'respuesta': resp,
                            **{k: round(v, 3) for k, v in fila.items()},
                            **extras,
                            'acierto': acierto})
        n = len(eval_set)
        promedios = {f'{k}_promedio': sum(f[k] for f in filas) / n for k in filas[0]}
        return {**promedios,
                'aciertos': sum(d['acierto'] for d in detalle),
                'total': n,
                'detalle': detalle}

    @staticmethod
    def tabla(scorecards: dict[str, dict]) -> str:
        """Side-by-side table of several scorecards, e.g. {'Baseline': sb, 'Finetuned': sf} (cell 38)."""
        ancho = 44 + 15 * len(scorecards)
        lineas = ['=' * ancho, f'{"Métrica":<40}' + ''.join(f'{nombre:>15}' for nombre in scorecards), '-' * ancho]
        for etiqueta, clave, formato in FILAS:
            lineas.append(f'{etiqueta:<40}' + ''.join(f'{s[clave]:>15{formato}}' for s in scorecards.values()))
        aciertos = ''.join(f'{str(s["aciertos"]) + "/" + str(s["total"]):>15}' for s in scorecards.values())
        lineas += [f'{"3 · Aciertos de dominio":<40}' + aciertos, '=' * ancho]
        return '\n'.join(lineas)

    def guardar(self, scorecards: dict[str, dict]) -> None:
        """scorecard.csv (summary) and detalle_<sistema>.json (per example, with the judges' explanations)."""
        carpeta = config.path('outputs')
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / 'scorecard.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['dimension'] + [f'puntaje_{nombre}' for nombre in scorecards])
            for _, clave, _ in FILAS:
                w.writerow([clave] + [round(s[clave], 3) for s in scorecards.values()])
            w.writerow(['aciertos_dominio'] + [f"{s['aciertos']}/{s['total']}" for s in scorecards.values()])
            w.writerow(['pesos_juez', json.dumps(self.judge.pesos)])
        for nombre, scorecard in scorecards.items():
            with open(carpeta / f'detalle_{nombre}.json', 'w', encoding='utf-8') as fh:
                json.dump(scorecard['detalle'], fh, ensure_ascii=False, indent=2)
