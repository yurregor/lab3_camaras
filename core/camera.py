"""
Captura de UNA camara en UN hilo, compartida por todos los modelos de esa camara.

Base: camera.py del Lab 2. Unico cambio para multicamara: el hilo se llama
"cam-capture-<nombre>" para distinguir las camaras en logs y en la CPU por hilo.
Cada camara es una instancia independiente: si una se cae, las demas siguen.

1. Cada frame recibe un numero (frame_id) -> los modelos solo procesan frames
   NUEVOS y se sabe cuantos se saltaron.
2. threading.Condition -> los workers duermen hasta que llega un frame nuevo.
3. Reconexion que NUNCA termina el hilo.
4. Timeouts de apertura/lectura de FFmpeg para que read() no se bloquee.
5. Fuentes "file" (mp4 en bucle) y "webcam" para probar en el PC.
"""
import logging
import platform
import threading
import time
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Lee la fuente de video en un hilo y publica el frame mas reciente."""

    def __init__(self, name: str, source, source_type: str, cfg,
                 on_event: Callable[..., None] | None = None):
        self.name = name
        self.source = int(source) if source_type == "webcam" else source
        self.source_type = source_type
        self.cfg = cfg
        self._emit = on_event or (lambda tipo, **d: None)
        self._cond = threading.Condition()   # protege frame/frame_id y despierta workers
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._capture_ts = 0.0               # perf_counter del ultimo frame
        self._cap = None
        self._thread: threading.Thread | None = None
        self.running = False
        self.frames_captured = 0
        self.frames_dropped = 0              # lecturas fallidas (igual que Lab 1)
        self.reconnections = 0
        self.input_resolution = (0, 0)
        self.last_frame_wall = 0.0           # time.time() del ultimo frame
        self._file_frame_period = 0.0

    # ---------------- ciclo de vida ----------------
    def start(self) -> bool:
        """Abre la fuente e inicia el hilo. Retorna True si hubo primer frame.

        El hilo se inicia aunque la primera conexion falle: el mismo hilo
        se encarga de reintentar.
        """
        connected = self._open()
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        name=f"cam-capture-{self.name}", daemon=True)
        self._thread.start()
        return connected

    def stop(self) -> None:
        """Detiene el hilo ANTES de liberar cap (evita corromper FFmpeg)."""
        self.running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.cfg.CAMERA_READ_TIMEOUT_MS / 1000 + 2)
        self._release()

    @property
    def native_thread_id(self) -> int | None:
        return self._thread.native_id if self._thread else None

    # ---------------- lectura para los workers ----------------
    def wait_new_frame(self, last_id: int, timeout: float):
        """Espera un frame con id > last_id. Retorna (frame_id, frame, capture_ts).

        Si vence el timeout o la camara se detiene, retorna (last_id, None, 0.0).
        El frame NO se copia: el hilo de captura reemplaza el objeto, nunca lo modifica.
        """
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._frame_id > last_id or not self.running, timeout)
            if not ok or not self.running:
                return last_id, None, 0.0
            return self._frame_id, self._frame, self._capture_ts

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def seconds_since_last_frame(self) -> float:
        return time.time() - self.last_frame_wall if self.last_frame_wall else float("inf")

    def is_ok(self) -> bool:
        """True si llego un frame nuevo hace menos de STALE_FRAME_TIMEOUT_S."""
        return self.seconds_since_last_frame() < self.cfg.STALE_FRAME_TIMEOUT_S

    def get_stats(self) -> dict:
        return {
            "input_resolution": f"{self.input_resolution[0]}x{self.input_resolution[1]}",
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "reconnections": self.reconnections,
            "frame_id_actual": self._frame_id,
        }

    # ---------------- internos ----------------
    def _capture_loop(self) -> None:
        while self.running:
            ok, frame = self._read()
            if ok:
                self._publish(frame)
                self._throttle_file_source()
            else:
                self.frames_dropped += 1
                if self.source_type == "file" and self._cap is not None and self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # fin del video: volver a empezar
                    continue
                self._reconnect_with_backoff()

    def _read(self):
        try:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            ok, frame = self._cap.read()
            return (ok and frame is not None), frame
        except cv2.error as exc:
            logger.warning("[%s] Error de lectura: %s", self.name, exc)
            return False, None

    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._frame_id += 1
            self._capture_ts = time.perf_counter()
            self._cond.notify_all()        # despierta a TODOS los workers de esta camara
        self.frames_captured += 1
        self.last_frame_wall = time.time()
        if self.input_resolution == (0, 0):
            self.input_resolution = (frame.shape[1], frame.shape[0])

    def _throttle_file_source(self) -> None:
        """Con un .mp4, entregar frames al ritmo del video y no lo mas rapido posible."""
        if self.source_type == "file" and self.cfg.FILE_SIMULATE_REALTIME and self._file_frame_period:
            time.sleep(self._file_frame_period)

    def _reconnect_with_backoff(self) -> None:
        """Reintenta abrir la fuente con espera progresiva hasta lograrlo o parar."""
        t_down = time.time()
        backoff = self.cfg.RECONNECT_INITIAL_BACKOFF_S
        self._emit("camera_disconnected", camara=self.name)
        attempt = 0
        while self.running:
            attempt += 1
            self._release()
            self._sleep_interruptible(backoff)
            if self.running and self._open():
                self.reconnections += 1
                self._emit("camera_reconnected", camara=self.name, intentos=attempt,
                           downtime_s=round(time.time() - t_down, 1))
                return
            logger.warning("[%s] Reconexion fallida (intento %d), espera %ss",
                           self.name, attempt, backoff)
            backoff = min(backoff * 2, self.cfg.RECONNECT_MAX_BACKOFF_S)

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(0.2)

    def _open(self) -> bool:
        """Crea el VideoCapture segun el tipo de fuente y lee un primer frame."""
        self._cap = self._create_capture()
        if self._cap is None or not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.source_type == "file":
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            self._file_frame_period = 1.0 / fps
        ok, frame = self._read()
        if ok:
            self._publish(frame)
        return ok

    def _create_capture(self):
        """Backend por sistema operativo: MSMF (Windows) / V4L2 (Linux) para webcam."""
        if self.source_type == "webcam":
            backend = cv2.CAP_MSMF if platform.system() == "Windows" else cv2.CAP_V4L2
            return cv2.VideoCapture(self.source, backend)
        if self.source_type == "file":
            return cv2.VideoCapture(str(self.source))
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.cfg.CAMERA_OPEN_TIMEOUT_MS,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.cfg.CAMERA_READ_TIMEOUT_MS]
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)

    def _release(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except cv2.error:
            pass
        self._cap = None
