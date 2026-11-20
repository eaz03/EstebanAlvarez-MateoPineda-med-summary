"""Judge: Flow-Judge-v0.1. Three rubrics scored 1-5 (notebook cell 33) and the RAGAS 0/1 verdict (cell 85)."""
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from medsummary.core import config
from medsummary.core.models.instruct import InstructModel

# logging: like print, but each message has a level (INFO, WARNING, ...) and main.py decides in one line
# whether to show it, hide it or send it elsewhere (a file, W&B) without editing this file.
log = logging.getLogger(__name__)


class Veredicto(BaseModel):
    explanation: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)


class VeredictoBinario(BaseModel):
    explanation: str = Field(min_length=1)
    score: Literal[0, 1]


class Judge(InstructModel):
    seccion_settings = 'judge'   # reads models.judge and the judge: block of settings.yaml

    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.max_new_tokens_ragas = config.settings()[self.seccion_settings]['max_new_tokens_ragas']
        self.plantilla_flow = config.prompt('judge/flow_template.txt')               # notebook: PLANTILLA_FLOW
        self.plantilla_seccion_input = config.prompt('judge/flow_input_section.txt')  # notebook: SECCION_INPUT

        rubricas_juez = config.prompt('judge/rubricas.yaml')
        self.rubricas = rubricas_juez['rubricas']                                     # notebook: RUBRICAS
        self.pesos = rubricas_juez['pesos']                                           # notebook: PESOS
        # If grounding scores this or less, the final score cannot exceed puntaje_final_maximo_si_falla_grounding.
        self.umbral_grounding = rubricas_juez['grounding_cap']['si_grounding_menor_o_igual']
        self.puntaje_final_maximo_si_falla_grounding = rubricas_juez['grounding_cap']['final_maximo']

        prompts_ragas = config.prompt('judge/ragas.yaml')
        self.rubricas_ragas = prompts_ragas['rubricas']                               # notebook: RUBRICAS_RAGAS
        self.mensaje_reintento_ragas = prompts_ragas['reintento']

    def _argumentos_de_carga(self) -> dict:
        return {'torch_dtype': self.dtype, 'low_cpu_mem_usage': True, 'attn_implementation': 'sdpa'}

    def construir_prompt_flow(self, entradas: dict | None, salida: str, criterio: str, rubrica: str) -> str:
        """Official Flow-Judge template. entradas={'tag': text} become <tag>text</tag> blocks; None = no INPUT section."""
        seccion_input = ''
        if entradas:
            bloques = ''
            for etiqueta, texto in entradas.items():
                bloques += f'<{etiqueta}>\n{texto}\n</{etiqueta}>\n'
            seccion_input = config.rellenar_plantilla(self.plantilla_seccion_input, entradas=bloques)
        return config.rellenar_plantilla(self.plantilla_flow, seccion_input=seccion_input, output=salida,
                                         criterio=criterio, rubrica=rubrica)

    @staticmethod
    def parsear_flow(texto: str) -> dict:
        """'<feedback>...</feedback> <score>N</score>' -> {'score', 'explanation', 'valido'}."""
        feedback = re.search(r'<feedback>\s*(.*?)\s*</feedback>', texto, re.S)
        puntaje = re.search(r'<score>\s*([1-5])\s*</score>', texto)
        valido = bool(feedback and puntaje)

        if feedback:
            explicacion = feedback.group(1)
        else:
            explicacion = texto.strip() or '(empty)'
        if puntaje:
            score = int(puntaje.group(1))
        else:
            score = 3   # NOTE: an unparseable score defaults to 3 (notebook behaviour); 'valido' flags it for review

        veredicto = Veredicto(explanation=explicacion, score=score)
        return {'score': veredicto.score, 'explanation': veredicto.explanation, 'valido': valido}

    def juez_puntua(self, dialogo: str, respuesta: str, rubrica: dict) -> dict:
        if rubrica['usa_dialogo']:
            entradas = {'conversation': dialogo}
        else:
            entradas = None
        prompt = self.construir_prompt_flow(entradas, respuesta, rubrica['criterio'], rubrica['rubrica'])
        mensajes = [{'role': 'user', 'content': prompt}]
        texto = self.generar_desde_mensajes(mensajes, self.max_new_tokens)
        return self.parsear_flow(texto)

    def juez_completo(self, dialogo: str, respuesta: str) -> tuple[float, dict]:
        """Three judges + weighted average. A factual error (low grounding) is not compensated by good form."""
        resultados = {}
        for juez, rubrica in self.rubricas.items():
            resultados[juez] = self.juez_puntua(dialogo, respuesta, rubrica)

        final = 0
        for juez, peso in self.pesos.items():
            final += peso * resultados[juez]['score']
        if resultados['grounding']['score'] <= self.umbral_grounding:
            final = min(final, self.puntaje_final_maximo_si_falla_grounding)
        return round(final, 2), resultados

    def juzgar_binario(self, entradas: dict, salida: str, metrica: str) -> tuple[int | None, str]:
        """One RAGAS verdict, 0 or 1 (notebook cell 85: juzgar_flow). If the answer is malformed it retries once
        showing the model its error; if it is still invalid it returns (None, raw answer)."""
        rubrica = self.rubricas_ragas[metrica]
        prompt = self.construir_prompt_flow(entradas, salida, rubrica['criterio'], rubrica['rubrica'])
        mensajes = [{'role': 'user', 'content': prompt}]

        for intento in range(2):
            respuesta = self.generar_desde_mensajes(mensajes, self.max_new_tokens_ragas)
            feedback = re.search(r'<feedback>\s*(.*?)\s*</feedback>', respuesta, re.S)
            puntaje = re.search(r'<score>\s*(\d+)\s*</score>', respuesta)
            # FIX vs notebook: the regex returns text ('1'); pydantic rejects text for Literal[0, 1], so in cell 85
            # every verdict was invalid. It is converted to int before validating.
            explicacion = feedback.group(1) if feedback else ''
            score = int(puntaje.group(1)) if puntaje else None
            try:
                veredicto = VeredictoBinario(explanation=explicacion, score=score)
                return veredicto.score, veredicto.explanation
            except ValidationError as error:
                # Greedy decoding gives the same answer to the same prompt, so the error is shown to the model.
                mensaje_error = config.rellenar_plantilla(self.mensaje_reintento_ragas, error=error.errors()[0]['msg'])
                mensajes.append({'role': 'assistant', 'content': respuesta})
                mensajes.append({'role': 'user', 'content': mensaje_error})

        log.warning('Veredicto inválido en %s; no se cuenta en el promedio', metrica)
        return None, respuesta
