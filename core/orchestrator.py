"""
Orquestador (GRASP Controller): corre las combinaciones UNA DETRAS DE OTRA, sin
pausa, sobre TODAS las camaras.

Cambios multicamara frente al Lab 2:
- Recibe la lista de camaras ya resuelta (CameraSpec) y abre UNA CameraStream
  por camara (cada una con su hilo de captura). Quedan abiertas todo el laboratorio.
- Por cada combinacion crea un worker por par (camara, modelo).
- El heartbeat informa el estado de cada camara.
- Corrige un error del Lab 2: los manejadores de senales quedaban DENTRO del
  propio manejador y nunca se instalaban.

Hilos de Python durante una combinacion:
    MainThread + hw-monitor + 1 cam-capture por camara + 1 worker por (camara, modelo).
"""
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

from core.camera import CameraStream
from core.camera_factory import CameraSpec
from core.discovery import discover_openvino_models, resolve_combination
from core.engine import OpenVinoModel
from core.events import EventLog
from core.hardware import HardwareMonitor
from core.status import Heartbeat, StatusManager, find_resumable
from core.threads import ThreadBudget
from core.worker import InferenceWorker, WorkerContext
from core.writers import Compressor, JsonlWriter, compress_orphans

logger = logging.getLogger(__name__)

COMPLETED, INTERRUPTED = "completed", "interrupted"


class Orchestrator:
    """Abre lo compartido (camaras, monitor) una vez y recorre las combinaciones."""

    def __init__(self, cfg, budget: ThreadBudget, camera_specs: list[CameraSpec],
                 force_resume: bool = False, force_new: bool = False, model_factory=None):
        self.cfg = cfg
        self.budget = budget
        self.camera_specs = camera_specs
        self.force_resume = force_resume     # lo pide el supervisor al relanzar
        self.force_new = force_new           # el supervisor ya pregunto y se eligio "n"
        # Permite inyectar un modelo falso en las pruebas (sin OpenVINO).
        self.model_factory = model_factory or (lambda info: OpenVinoModel(
            info, cfg, budget.hilos_por_modelo))
        self.stop_event = threading.Event()      # Ctrl+C / SIGTERM / SIGBREAK
        self.results_dir = Path(cfg.RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- punto de entrada ----------------
    def run(self) -> str:
        """Corre todas las combinaciones pendientes. Retorna COMPLETED o INTERRUPTED."""
        status, resumed = self._load_or_create_status()
        run_dir = self.results_dir / status.run_id
        compressor = Compressor(self.cfg.COMPRESS_CLOSED_FILES)
        events = EventLog(JsonlWriter(run_dir, f"{status.run_id}_events", 1, None),
                          status.run_id, self.cfg.SCHEMA_VERSION)
        compressor.on_done = lambda d: events.emit("file_compressed", **d)
        self._install_signal_handlers(events)
        events.emit("run_resume" if resumed else "run_start", device=self.cfg.DEVICE,
                    duracion_combinacion_min=self.cfg.DURACION_COMBINACION_MIN,
                    pendientes=status.pending(), reinicios=status.data["reinicios"],
                    modo_camaras=self.cfg.CAMERA_MODE,
                    camaras=[s.describe() for s in self.camera_specs])
        events.emit("thread_budget", **self.budget.as_dict())
        compress_orphans(run_dir, compressor, self.cfg.ROTATE_EVERY_S)

        heartbeat = Heartbeat(self.results_dir)
        hw = HardwareMonitor(self.cfg, run_dir, JsonlWriter(
            run_dir, f"{status.run_id}_hardware", self.cfg.BATCH_WRITE_FRAMES,
            self.cfg.ROTATE_EVERY_S, compressor), compressor, events.emit, status.run_id)
        cameras = [CameraStream(s.name, s.source, s.source_type, self.cfg, events.emit)
                   for s in self.camera_specs]
        result = INTERRUPTED
        try:
            heartbeat.write("starting")
            hw.probe()
            hw.start()
            self._start_cameras(cameras, hw, events)
            available = discover_openvino_models(Path(self.cfg.BASE_MODELS), self.cfg.MODEL_FORMAT)
            result = self._run_all(status, run_dir, available, cameras, hw, events, compressor, heartbeat)
        except Exception as exc:
            logger.exception("Error no controlado en el orquestador")
            events.emit("run_error", error=str(exc)[:500])
            heartbeat.write("crashed")
            raise
        finally:
            for camera in cameras:
                camera.stop()
            hw.stop()
            compressor.wait_all()
            status.save("completed" if result == COMPLETED else "interrupted")
            events.emit("run_end", resultado=result)
            events.close()
        heartbeat.write("finished" if result == COMPLETED else "interrupted")
        return result

    @staticmethod
    def _start_cameras(cameras: list[CameraStream], hw, events) -> None:
        """Abre cada camara de forma independiente: si una falla, las demas siguen."""
        for camera in cameras:
            if not camera.start():
                events.emit("camera_initial_connect_failed", camara=camera.name,
                            nota="el hilo de captura seguira reintentando")
            hw.register_thread(f"cam-capture-{camera.name}", camera.native_thread_id)

    # ---------------- combinaciones ----------------
    def _run_all(self, status, run_dir, available, cameras, hw, events, compressor, heartbeat) -> str:
        """Corre las combinaciones pendientes una tras otra, sin pausa entre ellas."""
        total_s = self.cfg.duracion_combinacion_s()
        for combo in status.pending():
            remaining_s = total_s - status.seconds_done(combo)
            models = resolve_combination(self.cfg.COMBINATIONS[combo], available)
            status.start_combo(combo)
            self._run_combination(combo, models, remaining_s, status, run_dir, cameras,
                                  hw, events, compressor, heartbeat)
            if self.stop_event.is_set():
                return INTERRUPTED
            status.finish_combo(combo)
        return COMPLETED

    def _run_combination(self, combo, models, remaining_s, status, run_dir, cameras,
                         hw, events, compressor, heartbeat) -> None:
        """Arranca un worker por (camara, modelo), espera `remaining_s` y los detiene."""
        names = [m.name for m in models]
        cam_names = [c.name for c in cameras]
        hw.combinacion = combo
        events.emit("combo_start", combinacion=combo, modelos=names, camaras=cam_names,
                    workers=len(names) * len(cam_names),
                    minutos_por_correr=round(remaining_s / 60, 2))
        combo_stop = threading.Event()
        summary = JsonlWriter(run_dir, f"{status.run_id}_{combo}_summary", 1,
                              self.cfg.ROTATE_EVERY_S, compressor)
        ctx = WorkerContext(status.run_id, run_dir, combo, names,
                            self.budget.hilos_por_modelo, cam_names)
        workers = [InferenceWorker(self.model_factory(info), camera, hw, events, summary,
                                   compressor, ctx, self.cfg, combo_stop)
                   for camera in cameras for info in models]
        for worker in workers:
            worker.start()
        try:
            self._wait(combo, remaining_s, workers, status, cameras, hw, heartbeat)
        finally:
            # Detener ESTA combinacion: avisar a sus hilos, esperar a que terminen y cerrar.
            combo_stop.set()
            for worker in workers:
                worker.join(timeout=60)
            summary.close()
            events.emit("combo_end", combinacion=combo,
                        workers_con_error=[w.key for w in workers if w.failed],
                        hilos_sin_terminar=[w.name for w in workers if w.is_alive()])

    def _wait(self, combo, remaining_s, workers, status, cameras, hw, heartbeat) -> None:
        """Bucle del hilo principal: cuenta segundos, escribe heartbeat y status."""
        cfg = self.cfg
        elapsed = 0.0
        t_last = time.perf_counter()
        last_hb = last_status = 0.0
        while elapsed < remaining_s and not self.stop_event.is_set():
            if all(w.failed for w in workers):
                break                                # nada que medir: pasar a la siguiente
            self.stop_event.wait(min(cfg.MAIN_LOOP_TICK_S, remaining_s - elapsed))
            now = time.perf_counter()
            status.add_seconds(combo, now - t_last)
            elapsed += now - t_last
            t_last = now
            if now - last_hb >= cfg.HEARTBEAT_INTERVAL_S:
                self._beat(heartbeat, combo, workers, cameras, hw, remaining_s - elapsed)
                last_hb = now
            if now - last_status >= cfg.STATUS_INTERVAL_S:
                status.save("running")
                last_status = now
        status.save("running")

    @staticmethod
    def _beat(heartbeat, combo, workers, cameras, hw, remaining_s) -> None:
        estado_camaras = {c.name: {"ok": c.is_ok(), "frame_id": c.frame_id,
                                   "reconexiones": c.reconnections} for c in cameras}
        todas_ok = all(e["ok"] for e in estado_camaras.values())
        modelos = sorted({w.info.name for w in workers})
        heartbeat.write("measuring" if todas_ok else "camera_reconnecting",
                        combinacion=combo, modelos="+".join(modelos), camaras=estado_camaras,
                        fps_actual={w.key: round(w.current_fps, 1) for w in workers},
                        minutos_restantes=max(0.0, remaining_s) / 60,
                        disco_libre_gb=hw.latest().get("disk_free_gb"))

    # ---------------- status y senales ----------------
    def _load_or_create_status(self) -> tuple[StatusManager, bool]:
        """Reanuda (supervisor o RESUME_MODE) o crea un run nuevo."""
        mode = "auto" if self.force_resume else "never" if self.force_new else self.cfg.RESUME_MODE
        previous = None if mode == "never" else find_resumable(self.results_dir)
        if previous and mode == "ask":
            answer = input(f"Run interrumpido {previous.name}. Reanudar? (s/n): ").strip().lower()
            previous = previous if answer == "s" else None
        if previous:
            status = StatusManager.load(previous)
            status.mark_resumed()
            status.save("running")
            logger.info("Reanudando %s. Pendientes: %s", status.run_id, status.pending())
            return status, True
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status = StatusManager(self.results_dir, run_id, self.cfg.COMBINATION_ORDER,
                               self.cfg.duracion_combinacion_s(), self.cfg.SCHEMA_VERSION,
                               [s.name for s in self.camera_specs])
        status.save("running")
        return status, False

    def _install_signal_handlers(self, events: EventLog) -> None:
        """Ctrl+C, SIGTERM (Linux) y SIGBREAK (Windows) detienen ordenadamente."""
        def handler(signum, _frame):
            events.emit("signal_received", senal=signal.Signals(signum).name)
            self.stop_event.set()

        signal.signal(signal.SIGINT, handler)
        for name in ("SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), handler)
