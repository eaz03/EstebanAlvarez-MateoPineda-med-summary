"""RagasEvaluator: the 4 RAGAS metrics (0 to 1) of a RAG note, each verdict 0/1 by Flow-Judge (notebook cells 84-85).
Uses the Flow version (cell 85). Cell 86 (Qwen + JSON) is dropped: it existed only because the judge was Qwen."""
from medsummary.core.models.judge import Judge


class RagasEvaluator:

    def __init__(self, judge: Judge):
        self.judge = judge

    def ragas(self, borrador: str, nota: str, chunks: list[str]) -> tuple[dict, dict]:
        """None = not applicable or no valid verdict."""
        if chunks:
            pasajes = '\n\n'.join(chunks)
        else:
            pasajes = '(no passages were retrieved)'
        puntajes = {}
        explicaciones = {}

        # Recall: do the chunks give a term for every everyday expression of the draft? (1 verdict)
        puntajes['context_recall'], explicaciones['context_recall'] = self.judge.juzgar_binario(
            {'clinical_draft': borrador}, pasajes, 'context_recall')

        # Precision: fraction of useful chunks (1 verdict per chunk)
        utiles = 0
        juzgados = 0
        explicaciones['context_precision'] = []
        for chunk in chunks:
            puntaje, explicacion = self.judge.juzgar_binario({'clinical_draft': borrador}, chunk, 'context_precision')
            explicaciones['context_precision'].append(explicacion)
            if puntaje is not None:
                utiles += puntaje
                juzgados += 1
        if juzgados > 0:
            puntajes['context_precision'] = utiles / juzgados
        else:
            puntajes['context_precision'] = None

        # Faithfulness and term correctness: only apply if the RAG changed the note (1 verdict each)
        if nota.strip() == borrador.strip():
            puntajes['faithfulness'] = None
            puntajes['answer_relevancy'] = None
        else:
            puntajes['faithfulness'], explicaciones['faithfulness'] = self.judge.juzgar_binario(
                {'original_note': borrador, 'retrieved_passages': pasajes}, nota, 'faithfulness')
            puntajes['answer_relevancy'], explicaciones['answer_relevancy'] = self.judge.juzgar_binario(
                {'original_note': borrador}, nota, 'answer_relevancy')

        return puntajes, explicaciones
