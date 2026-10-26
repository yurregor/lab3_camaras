"""
InferenceWorker: UN hilo por par (camara, modelo).

Cambios multicamara frente al Lab 2:
- El hilo se llama "worker-<camara>-<modelo>".
- Los archivos incluyen la camara: {run}_{combo}_frames_{camara}_{modelo}_<periodo>.jsonl
- Eventos y detecciones llevan el campo "camara".
- summary.jsonl registra n_camaras_config real y las camaras activas.

Ciclo de vida: cargar modelo -> warmup -> medir hasta stop_event -> resumen final -> liberar.
"""
import logging
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.result_parser import frame_counts, parse_detections
from core.stats import WindowStats
from core.writers import Compressor, JsonlWriter

logger = logging.getLogger(__name__)


@dataclass
class WorkerContext:
    """Lo que el worker necesita saber de la corrida (lo arma el orquestador)."""
    run_id: str
    run_dir: Path
    combinacion: str
    modelos_concurrentes: list[str]
    threads_per_model: int
    camaras: list[str] = field(default_factory=list)


class InferenceWorker(threading.Thread):
    """Ejecuta un modelo sobre UNA camara y registra todo."""

    def __init__(self, model, camera, hw, events, summary_writer: JsonlWriter,
                 compressor: Compressor, ctx: WorkerContext, cfg, stop_event: threading.Event):
        super().__init__(name=f"worker-{camera.name}-{model.info.name}", daemon=True)
        self.model = model
        self.info = model.info
        self.camera = camera
        self.hw = hw
        self.events = events
        self.summary_writer = summary_writer
        self.ctx = ctx
        self.cfg = cfg
        self.stop_event = stop_event
        self.failed = False
        self.current_fps = 0.0                 # lo lee el heartbeat
        self.warmup_time_s = 0.0
        self.device_efectivo = cfg.openvino_device()
        prefix = f"{ctx.run_id}_{ctx.combinacion}"
        tag = f"{camera.name}_{self.info.name}"
        self.frames_writer = JsonlWriter(ctx.run_dir, f"{prefix}_frames_{tag}",
                                         cfg.BATCH_WRITE_FRAMES, cfg.ROTATE_EVERY_S, compressor)
        self.det_writer = JsonlWriter(ctx.run_dir, f"{prefix}_detections_{tag}",
                                      cfg.BATCH_WRITE_FRAMES, cfg.ROTATE_EVERY_S, compressor)
        self._fps_times: deque = deque()
        self._last_saved_frame = 0.0

    @property
    def key(self) -> str:
        """Identificador del worker para heartbeat: 'cam1/yolo11n'."""
        return f"{self.camera.name}/{self.info.name}"

    # ---------------- ciclo de vida ----------------
    def run(self) -> None:
        native_id = threading.get_native_id()
        self.hw.register_thread(self.name, native_id)
        try:
            self._load()
            self._warmup()
            self._measure()
        except Exception as exc:
            self.failed = True
            logger.exception("Worker %s detenido por error", self.name)
            self._emit("worker_error", error=str(exc)[:300])
        finally:
            self.frames_writer.close()
            self.det_writer.close()
            self.model.release()
            self.hw.unregister_thread(native_id)
            self._emit("model_stopped")

    def _emit(self, tipo: str, **datos) -> None:
        """Todo evento del worker lleva camara y modelo."""
        self.events.emit(tipo, camara=self.camera.name, modelo=self.info.name, **datos)

    def _load(self) -> None:
        load_s = self.model.load()
        self._emit("model_loaded", tarea=self.info.task, load_time_s=round(load_s, 2),
                   device_config=self.model.device)

    def _warmup(self) -> None:
        """Frames sin metricas para estabilizar (compilacion en GPU, caches)."""
        t0 = time.perf_counter()
        last_id, done = self.camera.frame_id, 0
        while done < self.cfg.WARMUP_FRAMES and not self.stop_event.is_set():
            last_id, frame, _ = self.camera.wait_new_frame(last_id, self.cfg.FRAME_WAIT_TIMEOUT_S)
            if frame is None:
                continue
            self.model.predict(frame)
            if done == 0:
                self._after_first_prediction()
            done += 1
        self.warmup_time_s = round(time.perf_counter() - t0, 2)
        self._emit("warmup_done", warmup_time_s=self.warmup_time_s,
                   device_efectivo=self.device_efectivo)

    def _after_first_prediction(self) -> None:
        """Lo que solo se puede hacer cuando Ultralytics ya creo su backend."""
        if self.cfg.DEVICE == "cpu" and self.cfg.APPLY_OV_CPU_THREADS:
            applied = self.model.limit_cpu_threads()
            self._emit("ov_cpu_threads", aplicado=applied, hilos=self.ctx.threads_per_model)
        self.device_efectivo = self.model.effective_device()

    # ---------------- medicion ----------------
    def _measure(self) -> None:
        t0 = time.perf_counter()
        last_id = self.camera.frame_id
        window = self._new_window()
        consecutive_errors = 0
        while not self.stop_event.is_set():
            t_wait = time.perf_counter()
            frame_id, frame, capture_ts = self.camera.wait_new_frame(
                last_id, self.cfg.FRAME_WAIT_TIMEOUT_S)
            if frame is not None:
                saltados = frame_id - last_id - 1
                last_id = frame_id
                ok = self._process_frame(frame, frame_id, capture_ts, t_wait, t0, saltados, window)
                consecutive_errors = 0 if ok else consecutive_errors + 1
                if consecutive_errors >= self.cfg.MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(f"{consecutive_errors} errores de inferencia seguidos")
            if time.time() - window["wall_start"] >= self.cfg.SUMMARY_WINDOW_S:
                self._write_summary(window, "ventana")
                window = self._new_window(window["idx"] + 1)
        self._write_summary(window, "final")

    def _process_frame(self, frame, frame_id: int, capture_ts: float, t_wait: float,
                       t0: float, saltados: int, window: dict) -> bool:
        """Infiere un frame y registra frames, detecciones y estadisticas. True = OK."""
        espera_ms = (time.perf_counter() - t_wait) * 1000
        try:
            res = self.model.predict(frame)
        except Exception as exc:
            window["stats"].errores += 1
            self._emit("inference_error", frame_id=frame_id, error=str(exc)[:300])
            return False
        now = time.perf_counter()
        fps = self._instant_fps(now, t0)
        detections = parse_detections(res.result, self.cfg)
        counts = frame_counts(detections)
        edad_ms = (now - capture_ts) * 1000
        self.frames_writer.write(self._frame_line(res, frame_id, now - t0, fps, saltados,
                                                  espera_ms, edad_ms, counts))
        self._write_detections(detections, frame_id, frame)
        window["stats"].add(fps, res.total_ms, res.pre_ms, res.inf_ms, res.post_ms,
                            edad_ms, counts["n_detections"], counts["conf_mean"], saltados)
        self._maybe_save_frame(res)
        return True

    def _instant_fps(self, now: float, t0: float) -> float:
        """Inferencias completadas en el ultimo segundo (sobre frames unicos)."""
        self._fps_times.append(now)
        while self._fps_times and self._fps_times[0] < now - 1.0:
            self._fps_times.popleft()
        self.current_fps = len(self._fps_times) / min(1.0, max(now - t0, 1e-3))
        return round(self.current_fps, 1)

    def _frame_line(self, res, frame_id, t, fps, saltados, espera_ms, edad_ms, counts) -> dict:
        """Linea de frames.jsonl: campos del Lab 1 + nuevos (camara ya existia)."""
        hw = self.hw.latest()
        return {"schema_version": self.cfg.SCHEMA_VERSION, "run_id": self.ctx.run_id,
                "combinacion": self.ctx.combinacion, "modelo": self.info.name,
                "formato": self.info.format, "tarea": self.info.task,
                "camara": self.camera.name, "t": round(t, 2), "fps": fps,
                "lat_ms": round(res.total_ms, 2), "cpu": hw.get("cpu_total"),
                "ram_mb": hw.get("proc_rss_mb"), "temp_cpu": hw.get("temp_cpu"),
                "ts": datetime.now().isoformat(), "frame_id": frame_id,
                "frames_saltados": saltados, "pre_ms": round(res.pre_ms, 2),
                "inf_ms": round(res.inf_ms, 2), "post_ms": round(res.post_ms, 2),
                "espera_frame_ms": round(espera_ms, 2), "edad_frame_ms": round(edad_ms, 2),
                **counts}

    def _write_detections(self, detections: list, frame_id: int, frame) -> None:
        if not self.cfg.SAVE_DETECTIONS or self.hw.low_disk:
            return
        if not detections and not self.cfg.WRITE_EMPTY_DETECTIONS:
            return
        self.det_writer.write({
            "schema_version": self.cfg.SCHEMA_VERSION, "run_id": self.ctx.run_id,
            "combinacion": self.ctx.combinacion, "camara": self.camera.name,
            "modelo": self.info.name, "tarea": self.info.task, "frame_id": frame_id,
            "ts": datetime.now().isoformat(),
            "img_w": int(frame.shape[1]), "img_h": int(frame.shape[0]),
            "detecciones": detections})

    def _maybe_save_frame(self, res) -> None:
        """Evidencia visual opcional (SAVE_FRAMES). Apagada por defecto: dibujar cuesta CPU."""
        if not self.cfg.SAVE_FRAMES:
            return
        now = time.time()
        if now - self._last_saved_frame < self.cfg.SAVE_FRAMES_EVERY_S:
            return
        import cv2
        self._last_saved_frame = now
        folder = self.ctx.run_dir / "frames_guardados" / self.camera.name / self.info.name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), res.result.plot())
        self._emit("frame_saved", archivo=str(path))

    # ---------------- resumenes ----------------
    def _new_window(self, idx: int = 0) -> dict:
        return {"idx": idx, "stats": WindowStats(), "wall_start": time.time(),
                "cam_id_start": self.camera.frame_id,
                "reconnect_start": self.camera.reconnections,
                "dropped_start": self.camera.frames_dropped}

    def _write_summary(self, window: dict, tipo: str) -> None:
        """Una linea de summary.jsonl con los campos del Lab 1 + nuevos."""
        wall_end = time.time()
        duracion = wall_end - window["wall_start"]
        if window["stats"].total_frames == 0 and tipo == "ventana":
            return
        cam = self.camera.get_stats()
        line = {
            "schema_version": self.cfg.SCHEMA_VERSION, "run_id": self.ctx.run_id,
            "modo": "multicamara", "combinacion": self.ctx.combinacion,
            "n_camaras_config": len(self.ctx.camaras),
            "n_modelos_config": len(self.ctx.modelos_concurrentes),
            "modelo": self.info.name, "familia": self.info.family, "formato": self.info.format,
            "tarea": self.info.task, "camara": self.camera.name, "device": self.device_efectivo,
            "device_config": self.cfg.DEVICE, "os": platform.system(), "show_video": False,
            "input_resolution": cam["input_resolution"],
            "model_resolution": f"{self.cfg.IMGSZ}x{self.cfg.IMGSZ}",
            "model_load_time_s": round(self.model.load_time_s, 2),
            "warmup_time_s": self.warmup_time_s,
            **window["stats"].summarize(duracion),
            **self.hw.window_means(window["wall_start"], wall_end),
            "frames_dropped": self.camera.frames_dropped - window["dropped_start"],
            "duracion_s": round(duracion, 1),
            "timestamp_inicio": datetime.fromtimestamp(window["wall_start"]).isoformat(),
            "timestamp_fin": datetime.fromtimestamp(wall_end).isoformat(),
            "tipo_resumen": tipo, "ventana_idx": window["idx"],
            "modelos_concurrentes": self.ctx.modelos_concurrentes,
            "camaras_activas": self.ctx.camaras,
            "frames_unicos_camara": self.camera.frame_id - window["cam_id_start"],
            "reconexiones_ventana": self.camera.reconnections - window["reconnect_start"],
            "hilos_por_modelo": self.ctx.threads_per_model,
        }
        self.summary_writer.write(line)
        self.summary_writer.flush()
