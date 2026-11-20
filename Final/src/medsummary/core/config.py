"""The only file that reads config/settings.yaml and prompts/. Import it as a module: `from medsummary.core import config`,
then call config.settings(), config.prompt(...), config.path(...), config.rellenar_plantilla(...)."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]   # the MedSummary/ folder


@lru_cache
def settings() -> dict:
    """Contents of config/settings.yaml (read once, then cached)."""
    return yaml.safe_load((ROOT / 'config' / 'settings.yaml').read_text(encoding='utf-8'))


@lru_cache
def prompt(nombre_archivo: str):
    """prompts/<nombre_archivo>: a .txt returns its text, a .yaml returns its content as a dict."""
    ruta = ROOT / 'prompts' / nombre_archivo
    texto = ruta.read_text(encoding='utf-8')
    if ruta.suffix == '.yaml':
        return yaml.safe_load(texto)
    return texto


def path(clave: str) -> Path:
    """Absolute path of an entry in settings.yaml -> paths (e.g. path('adapter_dir'))."""
    return ROOT / settings()['paths'][clave]


def rellenar_plantilla(plantilla: str, **valores) -> str:
    """Replaces each [CLAVE] placeholder, in the given order, with .replace() (same method as the notebook's Flow-Judge code)."""
    for clave, valor in valores.items():
        plantilla = plantilla.replace(f'[{clave.upper()}]', str(valor))
    return plantilla
