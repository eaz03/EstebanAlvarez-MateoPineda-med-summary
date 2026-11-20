"""Pipeline: the whole system, end to end (ported from Entrega_final/medsummary/pipeline.py).
dialogue -> draft of the fine-tuned model (Flan-T5 + LoRA) -> agent with tools (ToolAgent): Qwen decides whether to
rewrite the draft with the advanced RAG (medical glossary) and/or schedule the follow-up appointment.
The agent's Qwen is the one in settings.yaml -> despliegue.variante (deployed: q4_k_m, GGUF)."""
import time

from medsummary.core import config
from medsummary.core.models import FineTunedModel, InstructModel, fijar_semilla
from medsummary.core.rag import AdvancedRAG, BM25Retriever, Corpus, Reranker, SemanticRetriever, ToolAgent


class Pipeline:

    def __init__(self, variante: str | None = None):
        """variante: 'sin_cuantizar', 'q8_0' or 'q4_k_m'. None = the deployed one (settings.yaml -> despliegue)."""
        if variante is None:
            variante = config.settings()['despliegue']['variante']
        self.variante = variante
        self.finetuned = None
        self.semantic = None
        self.bm25 = None
        self.reranker = None
        self.qwen = None
        self.agente = None

    def cargar(self) -> str:
        """Loads every model, builds the index and warms up the models. Returns the device used."""
        fijar_semilla(config.settings()['seed'])
        self.finetuned = FineTunedModel().cargar()
        corpus = Corpus.desde_pdf()
        self.semantic = SemanticRetriever(corpus).indexar()
        self.bm25 = BM25Retriever(corpus)
        self.reranker = Reranker().cargar()
        if self.variante == 'sin_cuantizar':
            self.qwen = InstructModel().cargar()
        else:
            # imported here so the rest of the project does not need llama.cpp installed
            from medsummary.core.models.quantized import QuantizedInstructModel
            self.qwen = QuantizedInstructModel(self.variante).cargar()
        self.agente = ToolAgent(AdvancedRAG(self.semantic, self.bm25, self.reranker, self.qwen), self.qwen)

        # Warm-up: the first call to each model is slower and must not count in the latency
        self.finetuned.resumir('Doctor: How are you? Patient: Fine.')
        self.qwen.generar('Answer OK.', 'OK', 5)
        return self.finetuned.device

    def procesar(self, dialogo: str) -> dict:
        inicio = time.perf_counter()
        borrador = self.finetuned.resumir(dialogo)
        segundos_borrador = time.perf_counter() - inicio

        inicio_agente = time.perf_counter()
        nota, llamadas = self.agente.responder_con_tools_detallado(borrador, dialogo)
        segundos_agente = time.perf_counter() - inicio_agente

        herramientas = []                               # tools that answered without error, in order
        for llamada in llamadas:
            if not llamada['resultado'].startswith('error') and llamada['tool'] not in herramientas:
                herramientas.append(llamada['tool'])
        componentes = ['modelo_ajustado']
        if 'buscar_en_corpus' in herramientas:
            componentes.append('rag')
        if 'agendar_cita' in herramientas:
            componentes.append('agendar_cita')

        return {'borrador': borrador, 'nota': nota, 'herramientas': herramientas, 'componentes': componentes,
                'valvula': nota.strip() == borrador.strip(),        # the agent returned the draft unchanged
                'segundos_borrador': segundos_borrador, 'segundos_agente': segundos_agente,
                'segundos': segundos_borrador + segundos_agente}

    def salud(self) -> dict:
        """What is loaded and whether the index answers, without running any model."""
        modelos = {
            'modelo_ajustado': self.finetuned is not None and self.finetuned.esta_cargado,
            'embedder': self.semantic is not None and self.semantic.esta_cargado,
            'reranker': self.reranker is not None and self.reranker.esta_cargado,
            'bm25': self.bm25 is not None,
            'qwen': self.qwen is not None and self.qwen.esta_cargado,
        }
        if self.semantic is None or self.semantic.coleccion is None:
            indice = {'responde': False}
        else:
            try:
                indice = {'responde': True, 'chunks': self.semantic.coleccion.count()}
            except Exception as error:
                indice = {'responde': False, 'error': str(error)}
        return {'modelos': modelos, 'indice': indice}
