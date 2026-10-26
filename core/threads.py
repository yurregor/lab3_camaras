"""
Presupuesto de hilos del laboratorio multicamara.

Por que existe: con varios modelos sobre varias camaras, cada libreria
(OpenVINO-CPU, torch, OpenMP, OpenCV) intenta usar TODOS los nucleos por su
cuenta -> sobresuscripcion, cambios de contexto y FPS inestables. Aqui se
CUENTAN los hilos disponibles y se reparten.

Reparto:
    reservadas = RESERVED_CPUS_BASE                 (SO + MainThread + hw-monitor)
               + CPUS_PER_CAMERA * camaras           (decodificar cada camara)
    workers    = modelos simultaneos * camaras       (un hilo por par camara-modelo)
    hilos_por_worker = (CPUs - reservadas) // workers   (o THREADS_PER_MODEL)

Si no alcanza ni 1 hilo por worker, o el computo estimado supera las CPUs,
hay SOBRESUSCRIPCION y main.py NO arranca (ABORT_ON_OVERSUBSCRIPTION = True).

IMPORTANTE: apply_thread_env() debe llamarse ANTES de importar numpy, cv2,
torch, openvino o ultralytics, porque esas librerias leen las variables de
entorno de hilos una sola vez, al importarse.

Este modulo solo usa la libreria estandar a proposito.
"""
import os
from dataclasses import dataclass, asdict

# Hilos de Python fijos (ademas de 1 de captura por camara y 1 worker por par camara-modelo).
#   MainThread: orquestador + heartbeat + status.
#   hw-monitor: muestreo de hardware + revision de compresiones.
FIXED_PYTHON_THREADS = ("MainThread", "hw-monitor")


@dataclass(frozen=True)
class ThreadBudget:
    """Reparto de hilos calculado al iniciar. Se registra en events.jsonl."""
    cpus_disponibles: int
    cpus_reservadas: int
    camaras: int
    modelos_simultaneos: int
    workers: int
    hilos_por_modelo: int          # = hilos por worker (nombre del Lab 2, se conserva)
    hilos_opencv: int
    hilos_ffmpeg_por_camara: int
    hilos_python_fijos: int
    hilos_python_captura: int
    hilos_python_workers: int
    hilos_computo_estimados: int
    sobresuscripcion: bool
    motivo: str

    def as_dict(self) -> dict:
        return asdict(self)


def available_cpus() -> int:
    """CPUs que el proceso realmente puede usar.

    En Linux respeta la afinidad (taskset, contenedores); en Windows no existe
    sched_getaffinity y se usa os.cpu_count().
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def compute_budget(n_models: int, n_cameras: int, reserved_base: int, cpus_per_camera: int,
                   per_model: int | None, opencv_threads: int | None, ffmpeg_threads: int,
                   cpus: int | None = None) -> ThreadBudget:
    """Calcula cuantos hilos de computo recibe cada worker (par camara-modelo).

    `cpus` solo se pasa en las pruebas; en ejecucion se detecta.
    Dentro de un worker el pre (OpenCV), la inferencia (OpenVINO) y el post
    (torch) ocurren EN SECUENCIA, por eso comparten el mismo numero de hilos.
    """
    cpus = cpus or available_cpus()
    n_cameras, n_models = max(1, n_cameras), max(1, n_models)
    workers = n_models * n_cameras
    reserved = reserved_base + cpus_per_camera * n_cameras
    usable = cpus - reserved
    threads = per_model or max(1, usable // workers)   # minimo 1 (si no alcanza: sobresuscripcion)
    computo = threads * workers + ffmpeg_threads * n_cameras + reserved_base
    motivo = ""
    if usable < workers:
        motivo = (f"quedan {max(0, usable)} CPUs para {workers} workers "
                  f"({n_cameras} camaras x {n_models} modelos): no alcanza 1 hilo por worker")
    elif computo > cpus:
        motivo = f"computo estimado {computo} > {cpus} CPUs disponibles"
    return ThreadBudget(
        cpus_disponibles=cpus, cpus_reservadas=reserved, camaras=n_cameras,
        modelos_simultaneos=n_models, workers=workers, hilos_por_modelo=threads,
        hilos_opencv=opencv_threads or threads, hilos_ffmpeg_por_camara=ffmpeg_threads,
        hilos_python_fijos=len(FIXED_PYTHON_THREADS), hilos_python_captura=n_cameras,
        hilos_python_workers=workers, hilos_computo_estimados=computo,
        sobresuscripcion=bool(motivo), motivo=motivo,
    )


def apply_thread_env(budget: ThreadBudget) -> None:
    """Fija las variables de entorno de hilos (antes de importar librerias).

    setdefault: si el usuario ya exporto un valor en la consola, se respeta.
    """
    n = str(budget.hilos_por_modelo)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)
    os.environ.setdefault("OPENCV_FFMPEG_THREADS", str(budget.hilos_ffmpeg_por_camara))


def apply_library_threads(budget: ThreadBudget) -> None:
    """Limita los hilos de OpenCV y torch (se llama despues de importarlos)."""
    import cv2
    cv2.setNumThreads(budget.hilos_opencv)
    try:
        import torch
        torch.set_num_threads(budget.hilos_por_modelo)
    except ImportError:
        pass
