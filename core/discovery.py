"""
Descubrimiento de modelos en disco (sin cambios frente al Lab 2).

Estructura esperada (BASE_MODELS en config.py):
    modelos/<tarea>/<familia>/OPENVINOFP16/<nombre>_openvino_model/
El nombre del modelo es la carpeta SIN "_openvino_model" (ej. yolo11n-pose).
"""
from pathlib import Path

from core.engine import ModelInfo

OPENVINO_SUFFIX = "_openvino_model"


def discover_openvino_models(base: Path, fmt: str) -> dict[str, ModelInfo]:
    """Retorna {nombre_modelo: ModelInfo} con todos los modelos OpenVINO encontrados."""
    found: dict[str, ModelInfo] = {}
    if not base.exists():
        return found
    for fmt_dir in sorted(base.glob(f"*/*/{fmt}")):
        task, family = fmt_dir.parent.parent.name, fmt_dir.parent.name
        for folder in sorted(p for p in fmt_dir.iterdir() if p.is_dir()):
            if not any(folder.glob("*.xml")):
                continue
            name = folder.name.removesuffix(OPENVINO_SUFFIX)
            found[name] = ModelInfo(name=name, family=family, task=task,
                                    format=fmt, path=folder)
    return found


def resolve_combination(names: list[str], available: dict[str, ModelInfo]) -> list[ModelInfo]:
    """Convierte los nombres de una combinacion en ModelInfo (lista TODOS los faltantes)."""
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"Modelos no encontrados en disco: {missing}. "
                         f"Disponibles: {sorted(available)}")
    return [available[n] for n in names]
