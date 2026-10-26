"""
status.json (para reanudar) y heartbeat.json (esta vivo?).

Cambio multicamara: el heartbeat informa el estado de CADA camara y los FPS
por par camara-modelo. status.json no cambia: el tiempo es por combinacion.

Escritura ATOMICA: se escribe a un .tmp y luego os.replace().
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, data: dict, retries: int = 5) -> bool:
    """Escribe JSON de forma atomica. Retorna False si no pudo (no lanza)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.2)
    logger.warning("No se pudo actualizar %s (archivo bloqueado por otro programa)", path.name)
    return False


class StatusManager:
    """Avance del run: por cada combinacion, su estado y los SEGUNDOS que ya corrio."""

    def __init__(self, results_dir: Path, run_id: str, combos: list[str],
                 duracion_s: float, schema: str, camaras: list[str] | None = None):
        self.path = results_dir / f"{run_id}_status.json"
        self.data = {
            "run_id": run_id, "status": "running",
            "started_at": datetime.now().isoformat(),
            "updated_at": None, "current_model_index": 0, "current_model": "N/A",
            "models_completed": [], "models_remaining": list(combos),
            "last_heartbeat": None, "schema_version": schema,
            "duracion_combinacion_s": duracion_s, "reinicios": 0,
            "camaras": list(camaras or []),
            "combos": {c: {"estado": "pending", "inicio": None, "fin": None,
                           "segundos": 0.0} for c in combos},
        }

    @classmethod
    def load(cls, path: Path) -> "StatusManager":
        obj = cls.__new__(cls)
        obj.path = path
        obj.data = json.loads(path.read_text(encoding="utf-8"))
        return obj

    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    def pending(self) -> list[str]:
        return list(self.data["models_remaining"])

    def seconds_done(self, combo: str) -> float:
        return float(self.data["combos"][combo]["segundos"])

    def start_combo(self, combo: str) -> None:
        c = self.data["combos"][combo]
        c["estado"] = "running"
        c["inicio"] = c["inicio"] or datetime.now().isoformat()
        names = list(self.data["combos"])
        self.data.update(current_model_index=names.index(combo), current_model=combo)
        self.save("running")

    def add_seconds(self, combo: str, seconds: float) -> None:
        c = self.data["combos"][combo]
        c["segundos"] = round(c["segundos"] + seconds, 1)

    def finish_combo(self, combo: str) -> None:
        c = self.data["combos"][combo]
        c["estado"], c["fin"] = "completed", datetime.now().isoformat()
        self.data["models_completed"].append(combo)
        self.data["models_remaining"].remove(combo)
        self.save("running")

    def mark_resumed(self) -> None:
        self.data["reinicios"] = self.data.get("reinicios", 0) + 1

    def save(self, status: str | None = None) -> None:
        if status:
            self.data["status"] = status
        now = datetime.now().isoformat()
        self.data["updated_at"] = now
        self.data["last_heartbeat"] = now
        write_json_atomic(self.path, self.data)


def find_resumable(results_dir: Path) -> Path | None:
    """status.json mas reciente que quedo 'running' (caida) o 'interrupted' (Ctrl+C)."""
    if not results_dir.exists():
        return None
    for path in sorted(results_dir.glob("run_*_status.json"), reverse=True):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") in ("running", "interrupted"):
                return path
        except (OSError, ValueError):
            continue
    return None


class Heartbeat:
    """heartbeat.json: estado general + estado de cada camara."""

    def __init__(self, results_dir: Path):
        self.path = results_dir / "heartbeat.json"

    def write(self, status: str, combinacion: str | None = None, modelos: str = "",
              camaras: dict | None = None, fps_actual: dict | None = None,
              minutos_restantes: float | None = None,
              disco_libre_gb: float | None = None) -> None:
        """camaras = {"cam1": {"ok": True, "frame_id": 123, "reconexiones": 0}, ...}
        fps_actual = {"cam1/yolo11n": 29.8, ...}
        """
        camaras = camaras or {}
        write_json_atomic(self.path, {
            "status": status, "modelo": modelos, "pid": os.getpid(),
            "last_update": datetime.now().isoformat(), "combinacion": combinacion,
            "camaras_ok": all(c.get("ok") for c in camaras.values()) if camaras else None,
            "camaras": camaras, "fps_actual": fps_actual or {},
            "minutos_restantes_combo": None if minutos_restantes is None else round(minutos_restantes, 1),
            "disco_libre_gb": disco_libre_gb,
        })
