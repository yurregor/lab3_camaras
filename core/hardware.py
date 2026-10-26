"""
Monitor de hardware en UN hilo (hw-monitor). Sin cambios de logica.

Con varias camaras, "cpu_hilos" trae una entrada por cada hilo registrado:
cam-capture-cam1, cam-capture-cam2, worker-cam1-yolo11n, ... y "otros".

Regla de la decision 11: cada metrica se prueba UNA vez en probe(). Si falla,
se desactiva toda la corrida y no aparece en hardware.jsonl. Sin reintentos.
"""
import glob
import json
import logging
import platform
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil

from core.writers import Compressor, JsonlWriter

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
GPU_FREQ_PATTERNS = (
    "/sys/class/drm/card*/gt_act_freq_mhz",                 # driver i915
    "/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq",  # driver xe
)
TEMP_SENSORS = ("coretemp", "k10temp", "cpu_thermal", "acpitz")


class HardwareMonitor(threading.Thread):
    """Muestrea hardware, escribe hardware.jsonl y publica el ultimo valor."""

    def __init__(self, cfg, run_dir: Path, writer: JsonlWriter, compressor: Compressor,
                 emit, run_id: str):
        super().__init__(name="hw-monitor", daemon=True)
        self.cfg = cfg
        self.run_dir = run_dir
        self.writer = writer
        self.compressor = compressor
        self.emit = emit
        self.run_id = run_id
        self.stop_event = threading.Event()
        self.enabled: dict[str, bool] = {}
        self.combinacion: str | None = None
        self.low_disk = False
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._history: deque = deque(maxlen=max(1, int(cfg.HW_HISTORY_S / cfg.HW_SAMPLE_INTERVAL_S)))
        self._threads: dict[int, str] = {}
        self._prev: dict = {}
        self._gpu_freq_path: str | None = None
        self._gpu_top_proc: subprocess.Popen | None = None
        self._gpu_top_file = run_dir / "intel_gpu_top.json"
        self._t0 = time.time()

    # ---------------- API para otros hilos ----------------
    def register_thread(self, label: str, native_id: int | None) -> None:
        """Los workers y las camaras se registran para medir CPU por hilo."""
        if native_id is None:
            return
        with self._lock:
            self._threads[native_id] = label

    def unregister_thread(self, native_id: int) -> None:
        with self._lock:
            self._threads.pop(native_id, None)

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def window_means(self, t_start: float, t_end: float) -> dict:
        """Promedios y maximos de hardware entre dos instantes (para summary.jsonl)."""
        with self._lock:
            rows = [r for r in self._history if t_start <= r["_wall"] <= t_end]

        def values(key: str) -> list:
            return [r[key] for r in rows if r.get(key) is not None]

        def mean(vals: list):
            return round(sum(vals) / len(vals), 1) if vals else None

        def maximum(vals: list):
            return round(max(vals), 1) if vals else None

        cpu, ram = values("cpu_total"), values("proc_rss_mb")
        temp, gpu = values("temp_cpu"), values("gpu_render_busy_pct")
        return {"cpu_mean": mean(cpu), "cpu_max": maximum(cpu),
                "ram_mean_mb": mean(ram), "ram_max_mb": maximum(ram),
                "temp_cpu_max": maximum(temp), "gpu_render_busy_mean": mean(gpu)}

    # ---------------- ciclo del hilo ----------------
    def probe(self) -> dict[str, bool]:
        """Prueba cada metrica habilitada UNA vez y deja el resultado en self.enabled."""
        checks = {
            "cpu": lambda: psutil.cpu_percent(interval=None),
            "cpu_per_core": lambda: psutil.cpu_percent(interval=None, percpu=True),
            "cpu_freq": lambda: psutil.cpu_freq().current,
            "load_avg": self._check_load_avg,
            "ram": psutil.virtual_memory,
            "swap": psutil.swap_memory,
            "process": lambda: (self._process.memory_info(), self._process.cpu_percent(None)),
            "threads_cpu": self._process.threads,
            "disk_io": lambda: psutil.disk_io_counters().read_bytes,
            "disk_usage": lambda: psutil.disk_usage(str(self.run_dir.resolve())),
            "net_io": lambda: psutil.net_io_counters().bytes_recv,
            "temp": self._read_temp_or_fail,
            "gpu_freq_sysfs": self._find_gpu_freq_path,
            "gpu_intel_gpu_top": self._start_intel_gpu_top,
        }
        for name, check in checks.items():
            if not self.cfg.HW_ENABLE.get(name, False):
                continue
            try:
                check()
                self.enabled[name] = True
            except Exception as exc:          # cualquier fallo desactiva la metrica
                self.enabled[name] = False
                self.emit("hw_metric_disabled", metrica=name, motivo=str(exc)[:200])
        self.emit("hw_probe_result", activas=[k for k, v in self.enabled.items() if v],
                  desactivadas=[k for k, v in self.enabled.items() if not v])
        return self.enabled

    def run(self) -> None:
        interval = self.cfg.HW_SAMPLE_INTERVAL_S
        next_t = time.time()
        while not self.stop_event.is_set():
            try:
                sample = self._sample()
                with self._lock:
                    self._latest = sample
                    self._history.append(sample)
                self.writer.write({k: v for k, v in sample.items() if not k.startswith("_")})
                self.compressor.poll()
            except Exception:
                logger.exception("Error muestreando hardware (se continua)")
            next_t += interval
            self.stop_event.wait(max(0.0, next_t - time.time()))

    def stop(self) -> None:
        self.stop_event.set()
        if self._gpu_top_proc and self._gpu_top_proc.poll() is None:
            self._gpu_top_proc.terminate()
        if self.is_alive():
            self.join(timeout=5)
        self.writer.close()

    # ---------------- muestreo ----------------
    def _sample(self) -> dict:
        now = time.time()
        dt = max(1e-3, now - self._prev.get("_wall", now - self.cfg.HW_SAMPLE_INTERVAL_S))
        s = {"schema_version": self.cfg.SCHEMA_VERSION, "run_id": self.run_id,
             "combinacion": self.combinacion, "ts": datetime.now().isoformat(),
             "t": round(now - self._t0, 1), "_wall": now}
        on = self.enabled.get
        if on("cpu"):
            s["cpu_total"] = psutil.cpu_percent(interval=None)
        if on("cpu_per_core"):
            s["cpu_per_core"] = psutil.cpu_percent(interval=None, percpu=True)
        if on("cpu_freq"):
            s["cpu_freq_mhz"] = round(psutil.cpu_freq().current)
        if on("load_avg"):
            s["load_avg_1m"] = round(psutil.getloadavg()[0], 2)
        if on("ram"):
            vm = psutil.virtual_memory()
            s["ram_used_mb"] = round(vm.used / 1e6, 1)
            s["ram_percent"] = vm.percent
        if on("swap"):
            s["swap_used_mb"] = round(psutil.swap_memory().used / 1e6, 1)
        if on("process"):
            s["proc_rss_mb"] = round(self._process.memory_info().rss / 1e6, 1)
            s["proc_cpu_percent"] = self._process.cpu_percent(None)
            s["proc_threads"] = self._process.num_threads()
        if on("threads_cpu"):
            s["cpu_hilos"] = self._threads_cpu(dt)
        if on("disk_io"):
            s.update(self._disk_io(dt))
        if on("disk_usage"):
            s["disk_free_gb"] = round(psutil.disk_usage(str(self.run_dir.resolve())).free / 1e9, 1)
            self._check_disk(s["disk_free_gb"])
        if on("net_io"):
            s.update(self._net_io(dt))
        if on("temp"):
            s["temp_cpu"] = self._read_temp()
        if on("gpu_freq_sysfs"):
            s["gpu_freq_mhz"] = self._read_gpu_freq()
        if on("gpu_intel_gpu_top"):
            s.update(self._read_intel_gpu_top())
        self._prev["_wall"] = now
        return s

    def _threads_cpu(self, dt: float) -> dict:
        """% de UN nucleo usado por cada hilo registrado y por el resto del proceso."""
        with self._lock:
            labels = dict(self._threads)
        current = {t.id: t.user_time + t.system_time for t in self._process.threads()}
        prev = self._prev.get("threads", {})
        self._prev["threads"] = current
        out, total_known = {}, 0.0
        for tid, label in labels.items():
            delta = current.get(tid, 0.0) - prev.get(tid, current.get(tid, 0.0))
            out[label] = round(100 * delta / dt, 1)
            total_known += delta
        total = sum(current.values()) - sum(prev.get(t, v) for t, v in current.items())
        out["otros"] = round(100 * max(0.0, total - total_known) / dt, 1)
        return out

    def _disk_io(self, dt: float) -> dict:
        io = psutil.disk_io_counters()
        prev = self._prev.get("disk", io)
        self._prev["disk"] = io
        return {"disk_read_mb_s": round((io.read_bytes - prev.read_bytes) / 1e6 / dt, 2),
                "disk_write_mb_s": round((io.write_bytes - prev.write_bytes) / 1e6 / dt, 2)}

    def _net_io(self, dt: float) -> dict:
        io = psutil.net_io_counters()
        prev = self._prev.get("net", io)
        self._prev["net"] = io
        return {"net_rx_mbps": round((io.bytes_recv - prev.bytes_recv) * 8 / 1e6 / dt, 2),
                "net_tx_mbps": round((io.bytes_sent - prev.bytes_sent) * 8 / 1e6 / dt, 2)}

    def _check_disk(self, free_gb: float) -> None:
        low = free_gb < self.cfg.MIN_FREE_DISK_GB
        if low and not self.low_disk:
            self.emit("low_disk", disk_free_gb=free_gb, accion="se deja de escribir detections")
        self.low_disk = low

    # ---------------- sensores opcionales ----------------
    @staticmethod
    def _check_load_avg() -> None:
        if IS_WINDOWS:
            raise RuntimeError("en Windows psutil crearia un hilo extra para emularlo")
        psutil.getloadavg()

    def _read_temp(self) -> float | None:
        temps = psutil.sensors_temperatures()
        for name in TEMP_SENSORS:
            if temps.get(name):
                return round(max(t.current for t in temps[name]), 1)
        return None

    def _read_temp_or_fail(self) -> float:
        if not hasattr(psutil, "sensors_temperatures"):
            raise RuntimeError("psutil no expone temperaturas en este SO")
        value = self._read_temp()
        if value is None:
            raise RuntimeError("sin sensores de temperatura (comun en maquinas virtuales)")
        return value

    def _find_gpu_freq_path(self) -> str:
        if IS_WINDOWS:
            raise RuntimeError("sysfs solo existe en Linux")
        for pattern in GPU_FREQ_PATTERNS:
            for path in glob.glob(pattern):
                Path(path).read_text()
                self._gpu_freq_path = path
                return path
        raise RuntimeError("no hay archivo de frecuencia de GPU en /sys/class/drm")

    def _read_gpu_freq(self) -> int | None:
        try:
            return int(Path(self._gpu_freq_path).read_text().strip())
        except (OSError, ValueError):
            return None

    def _start_intel_gpu_top(self) -> None:
        if IS_WINDOWS:
            raise RuntimeError("intel_gpu_top solo existe en Linux")
        cmd = [*self.cfg.INTEL_GPU_TOP_CMD, "-o", str(self._gpu_top_file)]
        self._gpu_top_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
        time.sleep(5)
        if self._gpu_top_proc.poll() is not None or not self._read_intel_gpu_top():
            if self._gpu_top_proc.poll() is None:
                self._gpu_top_proc.terminate()
            raise RuntimeError("intel_gpu_top no arranco o no produjo datos")

    def _read_intel_gpu_top(self) -> dict:
        if self._gpu_top_proc is None or self._gpu_top_proc.poll() is not None:
            return {}
        try:
            with open(self._gpu_top_file, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 32768))
                text = fh.read().decode("utf-8", "ignore")
            start = text.rfind("{", 0, text.rfind('"period"'))
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except (OSError, ValueError):
            return {}
        return _parse_gpu_top(data)


def _parse_gpu_top(data: dict) -> dict:
    engines = data.get("engines", {})

    def busy(prefix: str):
        for name, eng in engines.items():
            if name.startswith(prefix):
                return eng.get("busy")
        return None

    out = {"gpu_render_busy_pct": busy("Render"),
           "gpu_video_busy_pct": busy("Video/"),
           "gpu_freq_actual_mhz": data.get("frequency", {}).get("actual")}
    power = data.get("power", {})
    if power:
        out["gpu_power_w"] = power.get("GPU")
    return {k: v for k, v in out.items() if v is not None}
