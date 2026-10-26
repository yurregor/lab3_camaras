"""
Fabrica de camaras: convierte la configuracion en una lista de camaras.

Por que existe (GRASP Creator + SRP): decidir CUANTAS camaras hay, como se
llaman y de donde sale cada una es una responsabilidad distinta a LEER video
(CameraStream) y a COORDINAR la corrida (Orchestrator). Asi, agregar un modo
nuevo (por ejemplo, camaras desde un archivo YAML) solo toca este archivo.

Dos modos (config.CAMERA_MODE):
  "replicar" -> la camara base (CAMERA_SOURCE_TYPE + RTSP_URL / VIDEO_FILE_PATH)
               se abre CAMERA_REPLICAS veces, cada una con su propia conexion
               (cam1, cam2, ...). Simula N camaras reales con una sola.
  "lista"    -> cada elemento de CAMERAS es una camara distinta.

Las funciones que construyen la lista solo usan la libreria estandar: main.py
las llama ANTES de importar cv2 (para contar camaras en el presupuesto de hilos).
"""
import os
from dataclasses import dataclass
from pathlib import Path

SOURCE_TYPES = ("rtsp", "file", "webcam")


@dataclass(frozen=True)
class CameraSpec:
    """Una camara ya resuelta: nombre, tipo y fuente concreta para OpenCV."""
    name: str
    source_type: str          # rtsp | file | webcam
    source: object            # URL (str), ruta (str) o indice (int)
    replica_de: str | None = None   # en modo "replicar": de que fuente es copia

    def describe(self) -> dict:
        """Datos para eventos/logs SIN credenciales (la URL RTSP no se registra)."""
        shown = "***" if self.source_type == "rtsp" else str(self.source)
        return {"camara": self.name, "tipo": self.source_type, "fuente": shown,
                "replica": self.replica_de is not None}


def build_camera_specs(cfg) -> list[CameraSpec]:
    """Lista de camaras segun CAMERA_MODE. No valida: ver camera_spec_errors()."""
    if cfg.CAMERA_MODE == "replicar":
        return _replicated(cfg)
    return [_from_entry(entry, i) for i, entry in enumerate(cfg.CAMERAS, start=1)]


def camera_spec_errors(cfg) -> list[str]:
    """Revisa la configuracion de camaras y devuelve TODOS los errores (vacia = OK)."""
    errors = []
    if cfg.CAMERA_MODE not in ("replicar", "lista"):
        return [f"CAMERA_MODE debe ser 'replicar' o 'lista' (es {cfg.CAMERA_MODE!r})"]
    if cfg.CAMERA_MODE == "replicar":
        errors += _replica_errors(cfg)
    else:
        errors += _list_errors(cfg)
    if errors:
        return errors
    specs = build_camera_specs(cfg)
    if not 1 <= len(specs) <= cfg.MAX_CAMERAS:
        errors.append(f"Hay {len(specs)} camaras; deben estar entre 1 y MAX_CAMERAS={cfg.MAX_CAMERAS}")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        errors.append(f"Nombres de camara repetidos: {names}")
    return errors


# ---------------- modo replicar ----------------
def _replicated(cfg) -> list[CameraSpec]:
    source = cfg.camera_source()
    return [CameraSpec(f"cam{i}", cfg.CAMERA_SOURCE_TYPE, source, replica_de="base")
            for i in range(1, cfg.CAMERA_REPLICAS + 1)]


def _replica_errors(cfg) -> list[str]:
    errors = []
    if not isinstance(cfg.CAMERA_REPLICAS, int) or cfg.CAMERA_REPLICAS < 1:
        errors.append("CAMERA_REPLICAS debe ser un entero >= 1")
    if cfg.CAMERA_SOURCE_TYPE not in SOURCE_TYPES:
        errors.append(f"CAMERA_SOURCE_TYPE invalido: {cfg.CAMERA_SOURCE_TYPE!r}")
    elif cfg.CAMERA_SOURCE_TYPE == "rtsp" and not cfg.RTSP_URL:
        errors.append("Falta la variable de entorno LAB2_RTSP_URL (ver seccion 4 de config.py)")
    elif cfg.CAMERA_SOURCE_TYPE == "file" and not Path(cfg.VIDEO_FILE_PATH).is_file():
        errors.append(f"VIDEO_FILE_PATH no existe: {cfg.VIDEO_FILE_PATH!r}")
    elif cfg.CAMERA_SOURCE_TYPE == "webcam" and cfg.CAMERA_REPLICAS > 1:
        errors.append("Una webcam no se puede abrir varias veces: usa CAMERA_REPLICAS = 1")
    return errors


# ---------------- modo lista ----------------
def _from_entry(entry: dict, index: int) -> CameraSpec:
    kind = entry.get("type")
    name = entry.get("name") or f"cam{index}"
    if kind == "rtsp":
        source = os.environ.get(entry.get("url_env", ""), "")
    elif kind == "file":
        source = entry.get("path", "")
    else:
        source = int(entry.get("index", 0))
    return CameraSpec(name, kind, source)


def _list_errors(cfg) -> list[str]:
    if not cfg.CAMERAS:
        return ["CAMERA_MODE='lista' pero CAMERAS esta vacia"]
    errors = []
    for i, entry in enumerate(cfg.CAMERAS, start=1):
        label = entry.get("name") or f"cam{i}"
        kind = entry.get("type")
        if kind not in SOURCE_TYPES:
            errors.append(f"[{label}] type invalido: {kind!r}. Opciones: {list(SOURCE_TYPES)}")
        elif kind == "rtsp" and not os.environ.get(entry.get("url_env", ""), ""):
            errors.append(f"[{label}] falta la variable de entorno {entry.get('url_env')!r}")
        elif kind == "file" and not Path(entry.get("path", "")).is_file():
            errors.append(f"[{label}] el video no existe: {entry.get('path')!r}")
    return errors
