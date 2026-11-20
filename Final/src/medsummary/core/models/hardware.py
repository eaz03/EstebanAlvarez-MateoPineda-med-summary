"""Device detection (NVIDIA, Apple GPU or CPU), memory release and seed. Used by every model class."""
import gc
import os

# Must be set before importing torch: operations that MPS (Apple GPU) does not implement run on the CPU instead of failing.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
from transformers import set_seed

DTYPES = {'auto': 'auto', 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}


def detectar_dispositivo(preferencia: str = 'auto') -> str:
    """'auto' picks cuda (NVIDIA), then mps (Apple GPU), then cpu. Any other value (e.g. 'cpu') is used as given."""
    if preferencia != 'auto':
        return preferencia
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def liberar_memoria() -> None:
    """Returns the GPU memory PyTorch had reserved (notebook: gc.collect() + torch.cuda.empty_cache())."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def fijar_semilla(seed: int) -> None:
    """Seeds random, numpy and torch (notebook: set_seed(SEED))."""
    set_seed(seed)
