"""
Acumulador de estadisticas por ventana de tiempo (summary.jsonl). Sin cambios.

Cada worker (par camara-modelo) tiene su propio WindowStats (sin locks).
"""
from dataclasses import dataclass, field

import numpy as np


def percentile(values: list, p: float) -> float | None:
    """Percentil redondeado a 2 decimales; None si no hay datos."""
    if not values:
        return None
    return round(float(np.percentile(values, p)), 2)


def _mean(values: list, decimals: int = 2) -> float | None:
    return round(float(np.mean(values)), decimals) if values else None


@dataclass
class WindowStats:
    """Listas de mediciones de UNA ventana de un worker."""
    fps: list = field(default_factory=list)
    lat_ms: list = field(default_factory=list)
    pre_ms: list = field(default_factory=list)
    inf_ms: list = field(default_factory=list)
    post_ms: list = field(default_factory=list)
    edad_ms: list = field(default_factory=list)
    n_det: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    saltados: int = 0
    errores: int = 0

    def add(self, fps: float, lat: float, pre: float, inf: float, post: float,
            edad: float, n_det: int, conf_mean: float | None, saltados: int) -> None:
        self.fps.append(fps)
        self.lat_ms.append(lat)
        self.pre_ms.append(pre)
        self.inf_ms.append(inf)
        self.post_ms.append(post)
        self.edad_ms.append(edad)
        self.n_det.append(n_det)
        if conf_mean is not None:
            self.conf.append(conf_mean)
        self.saltados += saltados

    @property
    def total_frames(self) -> int:
        return len(self.lat_ms)

    def summarize(self, duracion_s: float) -> dict:
        """fps_mean = inferencias / duracion de la ventana (FPS reales sobre frames unicos)."""
        n = self.total_frames
        procesados_y_saltados = n + self.saltados
        return {
            "fps_mean": round(n / duracion_s, 2) if duracion_s > 0 else None,
            "fps_min": round(float(np.min(self.fps)), 2) if self.fps else None,
            "fps_max": round(float(np.max(self.fps)), 2) if self.fps else None,
            "fps_p50": percentile(self.fps, 50),
            "fps_p5": percentile(self.fps, 5),
            "latency_p50_ms": percentile(self.lat_ms, 50),
            "latency_p95_ms": percentile(self.lat_ms, 95),
            "latency_p99_ms": percentile(self.lat_ms, 99),
            "pre_ms_p50": percentile(self.pre_ms, 50),
            "inf_ms_p50": percentile(self.inf_ms, 50),
            "post_ms_p50": percentile(self.post_ms, 50),
            "edad_frame_p50_ms": percentile(self.edad_ms, 50),
            "edad_frame_p95_ms": percentile(self.edad_ms, 95),
            "n_detections_mean": _mean(self.n_det),
            "n_detections_max": int(max(self.n_det)) if self.n_det else None,
            "conf_mean": _mean(self.conf, 4),
            "total_frames": n,
            "frames_saltados_total": self.saltados,
            "cobertura_pct": round(100 * n / procesados_y_saltados, 2) if procesados_y_saltados else None,
            "errores_inferencia_ventana": self.errores,
        }
