"""
Carga e inferencia de modelos OpenVINO exportados por Ultralytics.

Sin cambios frente al Lab 2 (Open/Closed): multicamara solo crea MAS
instancias, una por par camara-modelo.

- El dispositivo sale de config (DEVICE = "cpu" | "gpu").
- Se devuelven los resultados del modelo y los tiempos por etapa.
- En CPU se limita INFERENCE_NUM_THREADS (ver core/threads.py).
"""
import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Datos de un modelo en disco (resultado del descubrimiento)."""
    name: str
    family: str
    task: str          # deteccion | segmentacion | pose
    format: str        # OPENVINOFP16
    path: Path         # carpeta *_openvino_model/


@dataclass
class InferenceResult:
    """Salida de una inferencia sobre un frame."""
    result: object     # ultralytics.engine.results.Results
    total_ms: float
    pre_ms: float
    inf_ms: float
    post_ms: float


class OpenVinoModel:
    """Envuelve un modelo YOLO OpenVINO y sus parametros de prediccion.

    Cada worker crea SU PROPIA instancia: Ultralytics no garantiza que un
    mismo objeto YOLO sea seguro entre hilos.
    """

    def __init__(self, info: ModelInfo, cfg, threads_per_model: int):
        self.info = info
        self.cfg = cfg
        self.threads_per_model = threads_per_model
        self.device = cfg.openvino_device()
        self.model = None
        self.load_time_s = 0.0
        self.threads_applied: bool | None = None

    def load(self) -> float:
        """Carga el modelo. Retorna el tiempo de carga en segundos."""
        from ultralytics import YOLO
        t0 = time.perf_counter()
        self.model = YOLO(str(self.info.path))
        self.load_time_s = time.perf_counter() - t0
        return self.load_time_s

    def predict(self, frame) -> InferenceResult:
        """Ejecuta una inferencia. Los parametros son los mismos del Lab 1."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame, verbose=False, device=self.device, conf=self.cfg.CONF_THRESHOLD,
            imgsz=self.cfg.IMGSZ, classes=self.cfg.CLASSES)
        total_ms = (time.perf_counter() - t0) * 1000
        res = results[0]
        speed = getattr(res, "speed", {}) or {}
        return InferenceResult(
            result=res, total_ms=total_ms,
            pre_ms=float(speed.get("preprocess") or 0.0),
            inf_ms=float(speed.get("inference") or 0.0),
            post_ms=float(speed.get("postprocess") or 0.0))

    def limit_cpu_threads(self) -> bool:
        """Solo en CPU: recompila el modelo con INFERENCE_NUM_THREADS."""
        if self.cfg.DEVICE != "cpu" or not self.cfg.APPLY_OV_CPU_THREADS:
            return False
        backend = self._backend()
        if backend is None or not hasattr(backend, "ov_compiled_model"):
            self.threads_applied = False
            return False
        import openvino as ov
        core = ov.Core()
        xml = next(Path(self.info.path).glob("*.xml"))
        compiled = core.compile_model(core.read_model(str(xml)), "CPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": self.threads_per_model})
        backend.ov_compiled_model = compiled
        self.threads_applied = True
        return True

    def effective_device(self) -> str:
        """Dispositivo donde OpenVINO REALMENTE ejecuta (ej. 'GPU.0')."""
        backend = self._backend()
        compiled = getattr(backend, "ov_compiled_model", None)
        if compiled is None:
            return self.device
        try:
            return ",".join(compiled.get_property("EXECUTION_DEVICES"))
        except Exception:  # la propiedad no existe en todas las versiones
            return self.device

    def release(self) -> None:
        self.model = None
        gc.collect()

    def _backend(self):
        """AutoBackend interno de Ultralytics (existe tras la 1a prediccion)."""
        predictor = getattr(self.model, "predictor", None)
        return getattr(predictor, "model", None)


def available_openvino_devices() -> list[str]:
    """Dispositivos que OpenVINO ve en este equipo (ej. ['CPU', 'GPU'])."""
    import openvino as ov
    return list(ov.Core().available_devices)
