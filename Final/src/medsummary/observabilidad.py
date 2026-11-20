"""Observability for the whole project (ported from Entrega_final/medsummary/observabilidad.py).
- W&B: generar_tablas.py logs latencies, scores and tables; the deployed API logs each request's latency and usage
  metrics. Only numbers and tables go to W&B: never dialogues or notes. With WANDB_API_KEY (in .env or the
  environment) it uploads online; without it, it saves offline in wandb/ (upload later with
  `uv run wandb sync wandb/offline-run-*`). If W&B fails, the program keeps going.
- Structured logging (API): each request leaves ONE JSON line with the time, the latency, the components used and
  whether the valve fired. Nothing else: not the dialogue, the draft or the note (no personal data)."""
import csv
import json
import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv

from medsummary.core import config

run = None
log_peticiones = logging.getLogger('medsummary.peticiones')
usos = {'peticiones': 0, 'usos_rag': 0, 'usos_agendar_cita': 0, 'valvulas': 0}


def iniciar_wandb(job_type: str, configuracion: dict) -> None:
    global run
    load_dotenv(config.ROOT / '.env')
    try:
        import wandb
        modo = os.environ.get('WANDB_MODE') or ('online' if os.environ.get('WANDB_API_KEY') else 'offline')
        run = wandb.init(project=os.environ.get('WANDB_PROJECT', config.settings()['wandb']['proyecto']),
                         job_type=job_type, mode=modo, config=configuracion, dir=str(config.ROOT),
                         settings=wandb.Settings(console='off'))     # do not copy the console: it could contain text
        print(f'W&B: modo {modo}' + (f' | {run.url}' if getattr(run, 'url', None) else ''))
    except Exception as error:
        print(f'W&B desactivado: {error}')
        run = None


def solo_numeros(fila: dict) -> dict:
    """Keeps the numeric values of a row (bool -> 0/1); drops text such as notes and explanations."""
    return {clave: int(valor) if isinstance(valor, bool) else valor
            for clave, valor in fila.items() if isinstance(valor, (int, float))}


def registrar(datos: dict) -> None:
    if run is not None:
        try:
            run.log(datos)
        except Exception as error:
            print(f'W&B: no se pudo registrar: {error}')


def registrar_tablas(rutas_csv: list) -> None:
    """Each CSV as a W&B table, and all of them as one artifact."""
    if run is None:
        return
    try:
        import wandb
        artefacto = wandb.Artifact('tablas', type='tablas')
        for ruta in rutas_csv:
            with open(ruta, encoding='utf-8') as archivo:
                filas = list(csv.reader(archivo))
            run.log({ruta.stem: wandb.Table(columns=filas[0], data=filas[1:])})
            artefacto.add_file(str(ruta))
        run.log_artifact(artefacto)
    except Exception as error:
        print(f'W&B: no se pudieron subir las tablas: {error}')


def cerrar() -> None:
    if run is not None:
        run.finish()


# ---------- structured logging of the deployed API ----------
class FormatoJson(logging.Formatter):
    """Writes only the request's dictionary, as JSON on one line."""
    def format(self, record):
        return json.dumps(record.datos, ensure_ascii=False)


def configurar_logging() -> None:
    if log_peticiones.handlers:
        return
    ruta_logs = config.path('logs')
    ruta_logs.parent.mkdir(parents=True, exist_ok=True)
    for manejador in (logging.FileHandler(ruta_logs, encoding='utf-8'), logging.StreamHandler(sys.stdout)):
        manejador.setFormatter(FormatoJson())
        log_peticiones.addHandler(manejador)
    log_peticiones.setLevel(logging.INFO)
    log_peticiones.propagate = False


def registrar_peticion(latencia_s: float, componentes: list[str], valvula: bool,
                       latencias_por_pieza: dict | None = None) -> None:
    """One JSON line per request (what the assignment asks for) and one point in W&B."""
    datos = {'hora': datetime.now(timezone.utc).isoformat(timespec='seconds'),
             'latencia_s': round(latencia_s, 3),
             'componentes': componentes,
             'valvula': valvula}
    log_peticiones.info('peticion', extra={'datos': datos})

    usos['peticiones'] += 1
    usos['usos_rag'] += 'rag' in componentes
    usos['usos_agendar_cita'] += 'agendar_cita' in componentes
    usos['valvulas'] += valvula
    if latencias_por_pieza is None:
        latencias_por_pieza = {}
    registrar({'latencia_s': latencia_s, 'valvula': int(valvula), **latencias_por_pieza, **usos,
               'tasa_valvula': usos['valvulas'] / usos['peticiones']})


def leer_registros(n: int = 200) -> list[dict]:
    """The last n requests of the log file, most recent first."""
    ruta_logs = config.path('logs')
    if not ruta_logs.is_file():
        return []
    with open(ruta_logs, encoding='utf-8') as archivo:
        ultimas = deque(archivo, maxlen=n)
    registros = []
    for linea in reversed(ultimas):
        try:
            registros.append(json.loads(linea))
        except json.JSONDecodeError:
            continue
    return registros
