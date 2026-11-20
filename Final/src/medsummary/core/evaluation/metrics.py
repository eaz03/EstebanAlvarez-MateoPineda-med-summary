"""Classic metrics: ROUGE (word overlap) and BERTScore (semantic similarity), per example (M2 harness, cell 31) and
per dataset (M1, cells 10, 16, 18, 20)."""
from functools import lru_cache

import evaluate

from medsummary.core import config
from medsummary.core.models.hardware import detectar_dispositivo

METRICAS_M1 = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bert_f1"]


@lru_cache
def _rouge():
    return evaluate.load("rouge")


@lru_cache
def _bertscore():
    return evaluate.load("bertscore")


def _argumentos_bertscore() -> dict:
    settings = config.settings()
    # NOTE vs notebook: the device is passed explicitly so that BERTScore also runs on the Apple GPU (mps);
    # the notebook let bert_score choose (cuda or cpu).
    return {'lang': settings['metricas']['bertscore_lang'],
            'rescale_with_baseline': settings['metricas']['rescale_with_baseline'],
            'device': detectar_dispositivo(settings['device'])}


def classic_metrics(output_llm: str, expected: str) -> tuple[dict, dict]:
    """One note against its reference."""
    rouge_result = _rouge().compute(predictions=[output_llm], references=[expected])
    bert_result = _bertscore().compute(predictions=[output_llm], references=[expected], **_argumentos_bertscore())
    return rouge_result, bert_result


def metricas_dataset(predicciones: list[str], referencias: list[str]) -> dict:
    """Corpus ROUGE + mean BERTScore F1 over a whole split."""
    rouge_results = _rouge().compute(predictions=predicciones, references=referencias)
    bert_results = _bertscore().compute(predictions=predicciones, references=referencias, **_argumentos_bertscore())
    bert_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
    return {**rouge_results, "bert_f1": bert_f1}


def evaluar_en_dataset(modelo, df) -> tuple[dict, list[str]]:
    """modelo: BaselineModel or FineTunedModel. Returns (metrics, predictions) for a DataFrame like val_df or test_df."""
    predicciones = modelo.resumir_lote(df["dialogue"].tolist())
    metricas = metricas_dataset(predicciones, df["section_text"].tolist())
    return metricas, predicciones


def comparar_metricas(metricas_baseline: dict, metricas_finetuned: dict) -> dict:
    """Delta that the fine-tuning added, metric by metric (cell 18)."""
    comparacion = {}
    for metrica in METRICAS_M1:
        base = metricas_baseline[metrica]
        fine = metricas_finetuned[metrica]
        comparacion[metrica] = {'baseline': base, 'finetuned': fine, 'delta': fine - base}
    return comparacion
