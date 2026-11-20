"""BasicRAG: semantic retrieval + Qwen rewrites the fine-tuned draft with glossary terms (notebook cell 54)."""
from medsummary.core import config
from medsummary.core.models.instruct import InstructModel
from medsummary.core.rag.retrievers import SemanticRetriever


class BasicRAG:

    def __init__(self, semantic: SemanticRetriever, instruct: InstructModel):
        settings = config.settings()
        prompts_rag = config.prompt('rag.yaml')
        self.semantic = semantic
        self.instruct = instruct                                         # notebook: generar() with gen_model
        self.k = settings['retrieval']['k']                              # notebook: K
        self.system_rag = prompts_rag['system_rag']                      # notebook: SYSTEM_RAG
        self.plantilla_user_rag = prompts_rag['user_rag']
        self.plantilla_chunk_contexto = prompts_rag['chunk_contexto']
        self.system_sin_rag = prompts_rag['system_sin_rag']

    def sistema_rag_detallado(self, resumen: str, k: int | None = None) -> tuple[str, list[str]]:
        """Returns the rewritten note and the texts of the retrieved chunks (RAGAS judges those chunks)."""
        if k is None:
            k = self.k
        resultados = self.semantic.buscar_semantica(resumen, k)                      # RETRIEVE

        bloques_contexto = []                                                         # AUGMENT
        for _, _, doc, fuente in resultados:
            bloques_contexto.append(config.rellenar_plantilla(self.plantilla_chunk_contexto, fuente=fuente, doc=doc))
        contexto = '\n\n'.join(bloques_contexto)
        user = config.rellenar_plantilla(self.plantilla_user_rag, contexto=contexto, resumen=resumen)

        nota = self.instruct.generar(self.system_rag, user)                          # GENERATE
        chunks_recuperados = [doc for _, _, doc, _ in resultados]
        return nota, chunks_recuperados

    def sistema_rag(self, resumen: str, k: int | None = None) -> str:
        nota, _ = self.sistema_rag_detallado(resumen, k)
        return nota

    def sistema_sin_rag(self, resumen: str) -> str:
        """Same rewrite task with no retrieved context (ablation of the retrieval step)."""
        return self.instruct.generar(self.system_sin_rag, resumen)
