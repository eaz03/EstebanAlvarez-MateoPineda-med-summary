"""How many notes Flow has already scored in the current run of generar_tablas.py (step 4).
Reads the local W&B file of the latest run; it does not touch the run. Run in another terminal:
    uv run python scripts/progreso.py
"""
import collections
import glob
import re

archivos = sorted(glob.glob('wandb/run-*/run-*.wandb'))
if not archivos:
    raise SystemExit('No hay ninguna corrida de W&B en wandb/.')
datos = open(archivos[-1], 'rb').read()
cuenta = collections.Counter(m.decode() for m in re.findall(rb'juez_final/([A-Za-z0-9_]+)', datos))
orden = ['baseline', 'finetuned', 'rag_base', 'rag_avanzado', 'qwen_mps_sin_cuantizar', 'qwen_mps_q8_0', 'qwen_mps_q4_k_m',
         'ablacion_completo', 'ablacion_sin_reranker', 'ablacion_sin_hibrida', 'ablacion_sin_reranker_ni_hibrida']
hechas = 0
for nombre in orden:
    n = min(cuenta.get(nombre, 0) // 2, 10)     # each W&B log writes the key twice (history record + summary update)
    hechas += n
    print(f'{nombre:<34} {n:>2}/10 {"#" * n}')
print(f'TOTAL {hechas}/{len(orden) * 10} notas evaluadas')
