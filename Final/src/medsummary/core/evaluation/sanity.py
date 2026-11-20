"""SanityCheck: tests the judges themselves with control notes whose expected score is known (notebook cell 37).
For each eval-set example it builds: the correct note, the dialogue copied as a note, the note + an invented fact,
and the note cut to one sentence; then checks that each judge reacts to its own error and only to it."""
import json
import re

from medsummary.core import config
from medsummary.core.evaluation.harness import Harness


def dividir_en_oraciones(texto: str) -> list[str]:
    """Splits a text into sentences (cuts after '.', '?' or '!')."""
    return re.split(r'(?<=[.!?])\s+', texto.strip())


def cumple(puntaje: float, direccion: str, umbral: float) -> bool:
    """'alto': the score must be >= umbral. 'bajo': the score must be <= umbral."""
    if direccion == 'alto':
        return puntaje >= umbral
    return puntaje <= umbral


def describir_expectativa(direccion: str, umbral: float) -> str:
    if direccion == 'alto':
        return f'{umbral} o más'
    return f'{umbral} o menos'


class SanityCheck:

    def __init__(self, harness: Harness):
        settings_sanity = config.settings()['sanity']
        self.harness = harness
        self.jueces = list(harness.judge.rubricas)                                  # notebook: JUECES
        self.oracion_inventada = settings_sanity['oracion_inventada']             # notebook: ORACION_INVENTADA
        self.expectativas = settings_sanity['expectativas']                       # notebook: EXPECTATIVAS
        self.juez_que_debe_detectarlo = settings_sanity['juez_que_debe_detectarlo']

    def crear_notas_de_control(self, ejemplo: dict) -> dict:
        nota_correcta = ejemplo['esperado']
        oraciones = dividir_en_oraciones(nota_correcta)
        notas = {'referencia': nota_correcta,
                 'dialogo': ejemplo['input'],
                 'inventada': nota_correcta + self.oracion_inventada}
        if len(oraciones) >= 3:
            notas['recortada'] = oraciones[0]
        return notas

    def evaluar_nota(self, numero_ejemplo: int, tipo_nota: str, ejemplo: dict, nota: str) -> dict:
        fila, _, extras = self.harness.evaluar_nota(ejemplo, nota)
        registro = {'ejemplo': numero_ejemplo, 'tipo_nota': tipo_nota, 'input': ejemplo['input'], 'respuesta': nota,
                    'rouge1': round(fila['rouge1'], 3), 'rouge2': round(fila['rouge2'], 3),
                    'rougeL': round(fila['rougeL'], 3), 'bert_f1': round(fila['bert_f1'], 3)}
        for juez in self.jueces:
            registro[f'juez_{juez}'] = fila[f'juez_{juez}']
        registro['juez_final'] = fila['juez_final']
        registro.update(extras)
        registro['acierto'] = self.harness.es_acierto(registro['bert_f1'], registro['juez_final'])

        # what was expected of each judge and whether it delivered
        registro['esperado_jueces'] = {}
        registro['cumple_jueces'] = {}
        for juez in self.jueces:
            direccion, umbral = self.expectativas[tipo_nota][juez]
            registro['esperado_jueces'][juez] = describir_expectativa(direccion, umbral)
            registro['cumple_jueces'][juez] = cumple(registro[f'juez_{juez}'], direccion, umbral)
        return registro

    def ejecutar(self, eval_set: list[dict]) -> list[dict]:
        """Evaluates every control note and saves detalle_sanity.json."""
        registros = []
        for numero_ejemplo, ejemplo in enumerate(eval_set):
            for tipo_nota, nota in self.crear_notas_de_control(ejemplo).items():
                registros.append(self.evaluar_nota(numero_ejemplo, tipo_nota, ejemplo, nota))
        carpeta = config.path('outputs')
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / 'detalle_sanity.json', 'w', encoding='utf-8') as archivo:
            json.dump(registros, archivo, ensure_ascii=False, indent=2)
        return registros

    @staticmethod
    def resumir(registros: list[dict], tipo_nota: str, juez: str) -> tuple[int, int, float]:
        """How many notes of one type met the expectation for one judge, and their mean score."""
        total, aciertos, suma = 0, 0, 0
        for registro in registros:
            if registro['tipo_nota'] != tipo_nota:
                continue
            total += 1
            suma += registro[f'juez_{juez}']
            if registro['cumple_jueces'][juez]:
                aciertos += 1
        promedio = suma / total if total else float('nan')
        return aciertos, total, promedio

    def _linea_resumen(self, registros: list[dict], tipo_nota: str, juez: str) -> str:
        aciertos, total, promedio = self.resumir(registros, tipo_nota, juez)
        if total == 0:
            return f'  {juez:<10} sin notas de tipo {tipo_nota}'
        direccion, umbral = self.expectativas[tipo_nota][juez]
        return (f'  {juez:<10} con nota {tipo_nota:<12} {aciertos:>2} de {total:<2} '
                f'dieron {describir_expectativa(direccion, umbral):<10} ({aciertos / total:>4.0%})   promedio {promedio:.1f}')

    def reporte(self, registros: list[dict]) -> str:
        lineas = ['1. PROMEDIO DE CADA JUEZ POR TIPO DE NOTA   (promedio observado [lo que debería dar])',
                  f'   {"tipo de nota":<14}' + ''.join(f'{juez:>22}' for juez in self.jueces)]
        for tipo_nota in self.expectativas:
            celdas = ''
            for juez in self.jueces:
                _, _, promedio = self.resumir(registros, tipo_nota, juez)
                direccion, umbral = self.expectativas[tipo_nota][juez]
                celdas += f'{promedio:>9.1f} [{describir_expectativa(direccion, umbral):>10}]'
            lineas.append(f'   {tipo_nota:<14}{celdas}')

        lineas.append('\n2. ¿RECONOCEN UNA NOTA CORRECTA?  (todos los jueces deberían dar 4 o más)')
        lineas += [self._linea_resumen(registros, 'referencia', juez) for juez in self.jueces]

        lineas.append('\n3. ¿CADA JUEZ DETECTA SU PROPIO ERROR?  (debería bajar el puntaje)')
        for tipo_nota, juez in self.juez_que_debe_detectarlo.items():
            lineas.append(self._linea_resumen(registros, tipo_nota, juez))

        lineas.append('\n4. ¿SE MANTIENEN EN SU DIMENSIÓN?  (un error que no es suyo no debería bajarles el puntaje)')
        for tipo_nota, juez_responsable in self.juez_que_debe_detectarlo.items():
            for juez in self.jueces:
                if juez != juez_responsable:
                    lineas.append(self._linea_resumen(registros, tipo_nota, juez))

        lineas.append('\nCómo leerlo: un % bajo en la sección 3 significa que el juez no ve ese error;'
                      '\nun % bajo en las secciones 2 o 4 significa que el juez castiga algo que no debe.')
        return '\n'.join(lineas)
