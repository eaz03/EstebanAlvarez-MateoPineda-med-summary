"""Datasets: MTS-Dialog for training and M1 evaluation (notebook cell 5) and the hand-curated domain eval set (M2)."""
import json

import pandas as pd

from medsummary.core import config


def cargar_mts_dialog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns train_df, val_df, test_df with the columns dialogue and section_text. Test = test 1 + test 2."""
    settings_dataset = config.settings()['dataset']
    cols = settings_dataset['columnas']
    urls = settings_dataset['urls']
    train_df = pd.read_csv(urls['train'])[cols].dropna().reset_index(drop=True)
    val_df = pd.read_csv(urls['validation'])[cols].dropna().reset_index(drop=True)
    test_df = pd.concat([pd.read_csv(urls['test1'])[cols], pd.read_csv(urls['test2'])[cols]], ignore_index=True).dropna()
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def cargar_eval_set() -> list[dict]:
    """Domain eval set: [{'input': dialogue, 'esperado': reference note, 'criterio': what a good note must do}]."""
    return json.loads(config.path('eval_set').read_text(encoding='utf-8'))
