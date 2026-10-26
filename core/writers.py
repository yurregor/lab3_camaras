"""
Escritura de archivos JSONL con lotes, rotacion por periodo y compresion (sin cambios).

- JsonlWriter: acumula lineas y las escribe cada N. Cambia de archivo cada
  ROTATE_EVERY_S. NO crea hilos.
- Compressor: comprime a .gz los archivos ya cerrados (gzip externo en Linux,
  modulo gzip de Python en Windows, dentro del hilo hw-monitor).
"""
import gzip
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_ROTATED_NAME = re.compile(r"_\d{8}_\d{6}(_p\d+)?\.jsonl$")


class Compressor:
    """Comprime archivos cerrados sin bloquear a los workers."""

    def __init__(self, enabled: bool, on_done: Callable[[dict], None] | None = None):
        self.enabled = enabled
        self.on_done = on_done
        self._gzip_bin = shutil.which("gzip")
        self._nice_bin = shutil.which("nice")
        self._lock = threading.Lock()
        self._pending: list[Path] = []
        self._running: list[tuple[subprocess.Popen, Path, float]] = []

    def submit(self, path: Path) -> None:
        """Encola un archivo cerrado. Retorna de inmediato."""
        if not self.enabled or not path.exists():
            return
        with self._lock:
            if self._gzip_bin:
                cmd = [self._gzip_bin, "-f", str(path)]
                if self._nice_bin:
                    cmd = [self._nice_bin, "-n", "19", *cmd]
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                self._running.append((proc, path, time.time()))
            else:
                self._pending.append(path)

    def poll(self) -> None:
        """Revisa compresiones terminadas (o comprime una pendiente sin gzip)."""
        with self._lock:
            still = []
            for proc, path, t0 in self._running:
                if proc.poll() is None:
                    still.append((proc, path, t0))
                else:
                    self._notify(path, t0, proc.returncode == 0)
            self._running = still
            path = self._pending.pop(0) if self._pending else None
        if path is not None:
            self._compress_in_python(path)

    def wait_all(self, timeout_s: float = 120) -> None:
        """Al cerrar el proceso: espera las compresiones en curso."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.poll()
            with self._lock:
                if not self._running and not self._pending:
                    return
            time.sleep(0.5)

    def _compress_in_python(self, path: Path) -> None:
        t0 = time.time()
        try:
            with open(path, "rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            self._notify(path, t0, True)
        except OSError as exc:
            logger.warning("No se pudo comprimir %s: %s", path, exc)
            self._notify(path, t0, False)

    def _notify(self, path: Path, t0: float, ok: bool) -> None:
        if self.on_done:
            self.on_done({"archivo": path.name, "ok": ok,
                          "inicio": datetime.fromtimestamp(t0).isoformat(),
                          "duracion_s": round(time.time() - t0, 1)})


class JsonlWriter:
    """Un archivo JSONL logico que rota cada `rotate_every_s` segundos (thread-safe)."""

    def __init__(self, directory: Path, prefix: str, batch_size: int,
                 rotate_every_s: int | None, compressor: Compressor | None = None):
        self.directory = directory
        self.prefix = prefix
        self.batch_size = batch_size
        self.rotate_every_s = rotate_every_s
        self.compressor = compressor
        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self._file = None
        self._path: Path | None = None
        self._period_end = 0.0
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._close_file_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        self._ensure_file_locked()
        self._file.write("\n".join(self._buffer) + "\n")
        self._file.flush()
        self._buffer.clear()

    def _ensure_file_locked(self) -> None:
        now = time.time()
        if self._file is not None and now < self._period_end:
            return
        self._close_file_locked()
        if self.rotate_every_s:
            start = now - (now % self.rotate_every_s)
            self._period_end = start + self.rotate_every_s
            stamp = datetime.fromtimestamp(start).strftime("%Y%m%d_%H%M%S")
            name = f"{self.prefix}_{stamp}.jsonl"
        else:
            self._period_end = float("inf")
            name = f"{self.prefix}.jsonl"
        self._path = self._unused_path(name)
        self._file = open(self._path, "a", encoding="utf-8", newline="\n")

    def _unused_path(self, name: str) -> Path:
        """Evita pisar un .gz existente del mismo periodo (pasa al reanudar en la misma hora)."""
        path = self.directory / name
        part = 2
        while Path(f"{path}.gz").exists():
            path = self.directory / name.replace(".jsonl", f"_p{part}.jsonl")
            part += 1
        return path

    def _close_file_locked(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        if self.compressor and self.rotate_every_s and self._path:
            self.compressor.submit(self._path)


def compress_orphans(directory: Path, compressor: Compressor, min_age_s: float) -> int:
    """Comprime archivos ROTADOS que quedaron sin comprimir por un cierre abrupto."""
    count = 0
    if not directory.exists():
        return 0
    for path in directory.rglob("*.jsonl"):
        if _ROTATED_NAME.search(path.name) and time.time() - path.stat().st_mtime > min_age_s:
            compressor.submit(path)
            count += 1
    return count
