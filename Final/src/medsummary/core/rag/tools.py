"""ToolAgent: Qwen decides, one tool at a time, whether to schedule the next appointment or to rewrite the draft
with the AdvancedRAG glossary (notebook cells 74 to 77). Arguments are validated with pydantic."""
import json
import logging
import re
from datetime import date
from typing import Optional

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, ValidationError

from medsummary.core import config
from medsummary.core.models.instruct import InstructModel
from medsummary.core.rag.advanced_rag import AdvancedRAG

log = logging.getLogger(__name__)


def agendar_cita(weeks=None, months=None) -> str:
    try:
        if not weeks and not months:
            return 'error: give weeks or months'
        fecha_cita = date.today() + relativedelta(weeks=int(weeks or 0), months=int(months or 0))
        return f"Next appointment scheduled for {fecha_cita.strftime('%d/%m/%Y')}."
    except Exception as e:
        return f"error: {e}"


def _se_parece(borrador: str, salida: str, minimo: float = 0.5) -> bool:
    """Share of the OUTPUT words that were already in the draft. Stops the RAG from replacing the whole note with chunks."""
    en_borrador = set(re.findall(r'\w+', borrador.lower()))
    sal = re.findall(r'\w+', salida.lower())
    return len([w for w in sal if w in en_borrador]) / max(len(sal), 1) >= minimo


def _extraer_json(t: str):
    i, j = t.find('{'), t.rfind('}')
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


class AgendarCitaArgs(BaseModel):
    weeks: Optional[int] = Field(default=None, ge=1, le=104, description='Weeks until the next appointment')
    months: Optional[int] = Field(default=None, ge=1, le=24, description='Months until the next appointment')


class BuscarEnCorpusArgs(BaseModel):
    consulta: str = Field(min_length=3, description='Clinical draft with colloquial expressions to look up in the medical glossary')


class ToolAgent:

    def __init__(self, advanced_rag: AdvancedRAG, instruct: InstructModel):
        settings = config.settings()
        prompts_tools = config.prompt('tools.yaml')
        self.advanced_rag = advanced_rag
        self.instruct = instruct
        self.similitud_minima = settings['tools']['similitud_minima']
        self.max_intentos = settings['tools']['max_intentos']
        self.max_new_tokens_tool = settings['instruct']['max_new_tokens_tool']
        self.plantilla_system_una_tool = prompts_tools['system_una_tool']
        self.plantilla_user_tool = prompts_tools['user_tool']
        self.plantilla_reintento_tool = prompts_tools['reintento_tool']
        descripciones = prompts_tools['descripciones']

        self.tools = {                                                      # notebook: TOOLS
            'agendar_cita': {'fn': agendar_cita, 'args': AgendarCitaArgs,
                             'description': descripciones['agendar_cita'],
                             # optional safety check: the text must mention weeks or months
                             'validar': lambda texto: bool(re.search(r'\b(week|month)s?\b', texto, re.I))},
            'buscar_en_corpus': {'fn': self.buscar_en_corpus, 'args': BuscarEnCorpusArgs,
                                 'description': descripciones['buscar_en_corpus']},
        }
        self.esquemas = []                                                  # notebook: ESQUEMAS
        for nombre, tool in self.tools.items():
            if nombre == 'buscar_en_corpus':
                args = {}                                                   # the draft is sent automatically
            else:
                args = tool['args'].model_json_schema()['properties']
            self.esquemas.append({'name': nombre, 'description': tool['description'], 'args': args})

    def buscar_en_corpus(self, consulta: str) -> str:
        salida = self.advanced_rag.sistema_rag_avanzado(consulta, k=self.advanced_rag.k)
        if _se_parece(consulta, salida, self.similitud_minima):
            return salida
        return consulta                                                     # not the draft anymore -> keep the draft

    def ejecutar_tool(self, nombre: str, args: dict, resumen: str | None = None) -> str:
        tool = self.tools.get(nombre)
        if tool is None:
            return f'error: the tool {nombre} does not exist.'
        try:
            if nombre == 'buscar_en_corpus':
                args = {'consulta': resumen}
            datos = tool['args'](**args)                                    # pydantic checks names, types and ranges
            return str(tool['fn'](**datos.model_dump()))
        except ValidationError as e:
            return f'error: invalid arguments for {nombre}: {e.errors()[0]["msg"]}'
        except Exception as e:
            return f'error: {e}'

    def _system_una_tool(self, nombre: str) -> str:
        esquema = next(e for e in self.esquemas if e['name'] == nombre)
        return config.rellenar_plantilla(self.plantilla_system_una_tool,
                                         esquema=json.dumps(esquema, ensure_ascii=False), nombre=nombre)

    def responder_con_tools_detallado(self, resumen: str, dialogo: str | None = None) -> tuple[str, list[dict]]:
        """Returns the final note and the list of tool calls made (the notebook printed them to a log)."""
        texto = dialogo or resumen
        nota, citas, llamadas = resumen, [], []

        for nombre, tool in self.tools.items():
            mensaje = config.rellenar_plantilla(self.plantilla_user_tool, resumen=resumen)
            for _ in range(self.max_intentos):
                respuesta = self.instruct.generar(self._system_una_tool(nombre), mensaje, self.max_new_tokens_tool)
                pedido = _extraer_json(respuesta)
                if not pedido or pedido.get('tool') != nombre:
                    break
                validar = tool.get('validar')
                if validar and not validar(texto):
                    break

                obs = self.ejecutar_tool(nombre, pedido.get('args', {}), resumen=resumen)
                llamadas.append({'tool': nombre, 'resultado': obs})
                log.info('called %s = %r', nombre, obs[:70])

                if obs.startswith('error'):
                    mensaje += config.rellenar_plantilla(self.plantilla_reintento_tool, nombre=nombre, obs=obs)
                    continue
                if nombre == 'buscar_en_corpus':
                    nota = obs
                else:
                    citas.append(obs)
                break

        nota_final = nota + ''.join(f' {c}' for c in citas)
        return nota_final, llamadas

    def responder_con_tools(self, resumen: str, dialogo: str | None = None) -> str:
        nota_final, _ = self.responder_con_tools_detallado(resumen, dialogo)
        return nota_final
