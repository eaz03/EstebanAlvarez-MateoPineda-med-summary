"""Local smoke test: runs every class once, in pipeline order, on the first example of the eval set.
It only checks that everything loads and runs on this machine; the full evaluation is ComparacionSistemas.

Run from the MedSummary folder:
    uv sync
    uv run python main.py

Needs (not in git): artifacts/m1_lora_adapter/ (the M1 adapter) and data/Medical_terminology.pdf (the RAG corpus).
Without the adapter the baseline note is used as the draft; without the PDF the RAG steps are skipped.
"""
import logging
import time

from medsummary.core import config
from medsummary.core.data import cargar_eval_set
from medsummary.core.evaluation import Harness, RagasEvaluator
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge, fijar_semilla
from medsummary.core.rag import AdvancedRAG, BasicRAG, BM25Retriever, Corpus, Reranker, SemanticRetriever, ToolAgent

EJEMPLOS_HARNESS = 2      # how many eval-set examples the harness evaluates at the end

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')   # shows the classes' messages


def titulo(texto: str) -> None:
    print('\n' + '=' * 80 + f'\n{texto}\n' + '=' * 80)


def main() -> None:
    inicio = time.perf_counter()
    fijar_semilla(config.settings()['seed'])
    eval_set = cargar_eval_set()
    dialogo = eval_set[0]['input']
    hay_adapter = config.path('adapter_dir').is_dir()
    hay_corpus = config.path('corpus_pdf').is_file()

    titulo('1 · BaselineModel (Flan-T5 sin fine-tuning)')
    baseline = BaselineModel()
    print('Dispositivo:', baseline.device)
    nota_baseline = baseline.resumir(dialogo)
    print(nota_baseline)
    baseline.liberar()

    titulo('2 · FineTunedModel (Flan-T5 + adapter LoRA)')
    if hay_adapter:
        finetuned = FineTunedModel()
        borrador = finetuned.resumir(dialogo)
        print('Norma lora_B (0 = adapter sin entrenar):', round(finetuned.norma_lora_b(), 3))
        print(borrador)
        finetuned.liberar()
        sistema_harness = finetuned.resumir
    else:
        print(f'Sin adapter en {config.path("adapter_dir")}: se usa la nota del baseline como borrador.')
        borrador = nota_baseline
        sistema_harness = baseline.resumir

    nota_rag_avanzado, chunks = borrador, []
    if hay_corpus:
        titulo('3 · RAG (Corpus, retrievers, BasicRAG, AdvancedRAG, ToolAgent)')
        corpus = Corpus.desde_pdf()
        print(f'{len(corpus.documentos)} páginas -> {len(corpus.chunks)} chunks')
        semantic = SemanticRetriever(corpus)
        qwen = InstructModel()
        basic_rag = BasicRAG(semantic, qwen)
        advanced_rag = AdvancedRAG(semantic, BM25Retriever(corpus), Reranker(), qwen)
        tool_agent = ToolAgent(advanced_rag, qwen)

        print('\nRAG básico  :', basic_rag.sistema_rag(borrador))
        nota_rag_avanzado, chunks = advanced_rag.sistema_rag_avanzado_detallado(borrador)
        print('RAG avanzado:', nota_rag_avanzado)
        nota_tools, llamadas = tool_agent.responder_con_tools_detallado(borrador, dialogo)
        print('Con tools   :', nota_tools)
        print('Tools usadas:', [llamada['tool'] for llamada in llamadas])

        qwen.liberar()                         # frees memory for the judge
        advanced_rag.reranker.liberar()
        semantic.liberar()
    else:
        titulo('3 · RAG')
        print(f'Sin corpus en {config.path("corpus_pdf")}: se omite el RAG.')

    titulo('4 · Judge (Flow-Judge) + RAGAS')
    judge = Judge()
    final, jueces = judge.juez_completo(dialogo, borrador)
    for juez, resultado in jueces.items():
        print(f'{juez:<10} {resultado["score"]}  valido={resultado["valido"]}  {resultado["explanation"][:100]}')
    print('Final ponderado:', final)
    if chunks:
        puntajes_ragas, _ = RagasEvaluator(judge).ragas(borrador, nota_rag_avanzado, chunks)
        print('RAGAS:', puntajes_ragas)

    titulo(f'5 · Harness sobre {EJEMPLOS_HARNESS} ejemplos del eval set')
    harness = Harness(judge)
    scorecard = harness.harness(eval_set[:EJEMPLOS_HARNESS], sistema_harness)
    nombre_sistema = 'Finetuned' if hay_adapter else 'Baseline'
    print(Harness.tabla({nombre_sistema: scorecard}))

    print(f'\nListo en {time.perf_counter() - inicio:.0f} s.')


if __name__ == '__main__':
    main()
