"""AdvancedRAG: BasicRAG + query decomposition -> hybrid search -> rerank -> term substitution (notebook cells 67, 70)."""
import re

from medsummary.core import config
from medsummary.core.models.instruct import InstructModel
from medsummary.core.rag.basic_rag import BasicRAG
from medsummary.core.rag.retrievers import BM25Retriever, HybridRetriever, Reranker, SemanticRetriever


class AdvancedRAG(BasicRAG):

    def __init__(self, semantic: SemanticRetriever, bm25: BM25Retriever, reranker: Reranker, instruct: InstructModel,
                 usar_hibrida: bool = True, usar_reranker: bool = True):
        """usar_hibrida / usar_reranker = False switch a piece off for the ablation (notebook cell 92)."""
        super().__init__(semantic, instruct)
        self.usar_hibrida = usar_hibrida
        self.usar_reranker = usar_reranker
        settings = config.settings()
        prompts_rag = config.prompt('rag.yaml')
        self.hybrid = HybridRetriever(semantic, bm25)
        self.reranker = reranker

        settings_retrieval = settings['retrieval']
        self.n_hybrid = settings_retrieval['n_hybrid']
        self.k_por_expresion = settings_retrieval['k_por_expresion']     # notebook: K_POR_EXPRESION
        self.max_expresiones = settings_retrieval['max_expresiones']     # notebook: MAX_EXPRESIONES
        self.min_score = settings_retrieval['min_score']                 # notebook: MIN_SCORE

        settings_instruct = settings['instruct']
        self.max_new_tokens_decomposition = settings_instruct['max_new_tokens_decomposition']
        self.max_new_tokens_elegir_termino = settings_instruct['max_new_tokens_elegir_termino']

        self.system_decomposition = prompts_rag['system_decomposition']
        self.plantilla_user_decomposition = prompts_rag['user_decomposition']
        self.system_elegir_termino = prompts_rag['system_elegir_termino']
        self.plantilla_user_elegir_termino = prompts_rag['user_elegir_termino']
        self.plantilla_consulta_rerank = prompts_rag['consulta_rerank']

    def decomposition(self, consulta: str) -> list[str]:
        """One search question per everyday expression: 'What is the medical term for "<expression>"?'.
        NOTE vs notebook: cell 67 also called generar() once on the whole note and discarded the answer; that call
        is removed (same output, one less LLM call, so latency is lower than the notebook's measurements)."""
        preguntas = []
        # one call per sentence: a small model finds more when it reads less
        for frase in re.split(r'(?<=[.!?])\s+', consulta.strip()):
            if len(frase) < 10:
                continue
            user = config.rellenar_plantilla(self.plantilla_user_decomposition, frase=frase)
            texto = self.instruct.generar(self.system_decomposition, user, self.max_new_tokens_decomposition)
            for linea in texto.splitlines():
                linea = re.sub(r"^\s*(?:\d+[\.\)]|[-•])\s*", "", linea).strip()
                if linea and "?" in linea and linea not in preguntas:
                    preguntas.append(linea)
        return preguntas

    @staticmethod
    def _expresion(pregunta: str) -> str | None:
        """Keeps only the quoted expression of a decomposition question, which is what gets searched."""
        m = re.search(r'"([^"]+)"', pregunta)
        return m.group(1).strip() if m else None

    def recuperar_avanzado(self, consulta: str, k: int | None = None, n_hybrid: int | None = None) -> list[tuple]:
        """Returns [(expression, chosen chunks)]: decomposition -> hybrid search per expression -> rerank."""
        if k is None:
            k = self.k
        if n_hybrid is None:
            n_hybrid = self.n_hybrid

        # 1) Decomposition: keep only expressions that really appear in the draft
        expresiones = []
        for pregunta in self.decomposition(consulta):
            expresion = self._expresion(pregunta)
            if expresion and expresion.lower() in consulta.lower() and expresion not in expresiones:
                expresiones.append(expresion)
        expresiones = expresiones[:self.max_expresiones]

        # 2) Per expression: hybrid search (bare expression) -> rerank (as a question)
        final = []
        for expresion in expresiones:
            if self.usar_hibrida:
                candidatos = self.hybrid.buscar_hibrida(expresion, k=k, n=n_hybrid)      # semantic + BM25 + RRF
            else:
                candidatos = self.semantic.buscar_semantica(expresion, k)                 # semantic only
            if self.usar_reranker:
                consulta_rerank = config.rellenar_plantilla(self.plantilla_consulta_rerank, expresion=expresion)
                ordenados = self.reranker.rerank(consulta_rerank, candidatos, k=len(candidatos))
                elegidos = [r for r in ordenados if r[0] >= self.min_score][:self.k_por_expresion]
            else:
                elegidos = candidatos[:self.k_por_expresion]     # retriever order; there is no score to filter
            if not elegidos:
                continue
            final.append((expresion, elegidos))
        return final

    def _elegir_termino(self, expresion: str, chunks_expresion: list[tuple]) -> str | None:
        """Asks Qwen for ONE technical term from the chunks; None if there is none or it fails the safety checks."""
        contexto = '\n\n'.join(doc for _, _, doc, _ in chunks_expresion)
        user = config.rellenar_plantilla(self.plantilla_user_elegir_termino, contexto=contexto, expresion=expresion)
        respuesta = self.instruct.generar(self.system_elegir_termino, user, self.max_new_tokens_elegir_termino)
        termino = respuesta.strip().strip('."\'').split('\n')[0]

        # safety: the term must really be in the chunks, and not be the expression itself
        if not termino or termino.upper() == 'NONE' or termino.lower() == expresion.lower():
            return None
        if termino.lower() not in contexto.lower():
            return None
        return termino

    def _augment_generate(self, resumen: str, resultados: list[tuple]) -> str:
        nota = resumen
        for expresion, chunks_expresion in resultados:
            termino = self._elegir_termino(expresion, chunks_expresion)
            if termino:
                if termino[0].isupper() and not termino[:2].isupper():   # not an acronym
                    termino = termino[0].lower() + termino[1:]           # glossary terms start in lowercase in the note
                nota = re.sub(re.escape(expresion), termino, nota, count=1, flags=re.I)
        return nota

    def sistema_rag_avanzado_detallado(self, pregunta: str, k: int | None = None,
                                       n_hybrid: int | None = None) -> tuple[str, list[str]]:
        """Returns the rewritten note and the texts of every chunk chosen for every expression."""
        ctx = self.recuperar_avanzado(pregunta, k, n_hybrid)
        if ctx:
            nota = self._augment_generate(pregunta, ctx)
        else:
            nota = pregunta
        chunks_recuperados = []
        for _, chunks_expresion in ctx:
            for _, _, doc, _ in chunks_expresion:
                chunks_recuperados.append(doc)
        return nota, chunks_recuperados

    def sistema_rag_avanzado(self, pregunta: str, k: int | None = None, n_hybrid: int | None = None) -> str:
        nota, _ = self.sistema_rag_avanzado_detallado(pregunta, k, n_hybrid)
        return nota
