"""Installs llama-cpp-python (GGUF) with GPU support on a machine with an NVIDIA GPU (copied from Entrega_final).
With GPU: prebuilt wheel for torch's CUDA version; if it does not work, it compiles with CUDA (slow). Without GPU: nothing.
Not needed on a Mac: `uv sync` compiles llama.cpp from source with Metal (Apple GPU) already.
The environment is managed by uv (no pip), so it installs with `uv pip install`.

Usage, from the MedSummary folder and only on a machine with an NVIDIA GPU:
    uv run python scripts/instalar_llama_cpp.py
Afterwards start with `uv run ...` (not `uv sync`, which reinstalls llama-cpp-python from pyproject.toml without CUDA).
"""
import os
import subprocess
import sys

import torch


def pip_instalar(*argumentos, entorno=None):
    subprocess.check_call(['uv', 'pip', 'install', '-q', '--python', sys.executable, *argumentos], env=entorno)

def llama_soporta_gpu():
    """Se pregunta en un proceso nuevo, para no depender de lo que ya haya importado esta sesión."""
    r = subprocess.run([sys.executable, '-c', 'import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'],
                       capture_output=True, text=True)
    return r.stdout.strip().endswith('True')

def etiqueta_cuda(version_cuda):
    """Ruedas publicadas: cu118, cu121, cu122, cu123, cu124, cu125, cu130, cu132. Se usa la más alta que no pase de la versión de torch."""
    publicadas = [(11, 8), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (13, 0), (13, 2)]
    mayor, menor = (int(x) for x in version_cuda.split('.')[:2])
    validas = [v for v in publicadas if v <= (mayor, menor)]
    return 'cu%d%d' % validas[-1] if validas else None

INDICE = 'https://abetlen.github.io/llama-cpp-python/whl/'
if torch.cuda.is_available() and torch.version.cuda:
    etiqueta = etiqueta_cuda(torch.version.cuda)
    print(f'GPU detectada (CUDA {torch.version.cuda}): rueda {etiqueta}')
    if etiqueta:
        pip_instalar('llama-cpp-python', '--reinstall', '--extra-index-url', INDICE + etiqueta, '--index-strategy', 'unsafe-best-match')
    if not llama_soporta_gpu():
        print('La rueda no trae soporte de GPU para este entorno: se compila con CUDA (varios minutos).')
        pip_instalar('llama-cpp-python', '--reinstall', '--no-cache', '--no-binary', 'llama-cpp-python',
                     entorno={**os.environ, 'CMAKE_ARGS': '-DGGML_CUDA=on'})
    print('llama.cpp con soporte de GPU:', llama_soporta_gpu())
else:
    print('Sin GPU NVIDIA (o torch sin CUDA): se queda el llama-cpp-python de pyproject.toml (en Mac, con Metal).')
    if not torch.cuda.is_available() and sys.platform == 'win32':
        print('En Windows, torch de PyPI es solo CPU: si tiene GPU NVIDIA, instale primero torch con CUDA '
              '(https://pytorch.org/get-started/locally/) y vuelva a correr este script.')
