"""
Configuracion del Laboratorio 3: inferencia concurrente OpenVINO FP16 MULTICAMARA.

Base: config.py del Lab 2. Cambios: seccion 3 (presupuesto por camara y
bloqueo por sobresuscripcion) y seccion 4 (varias camaras: replicar o lista).

Es la UNICA fuente de parametros del laboratorio. Ningun modulo de core/
tiene numeros "magicos": todo lo que se puede ajustar vive aqui.

Regla: este archivo NO importa librerias pesadas (cv2, torch, openvino,
ultralytics). Asi main.py y run_lab3.py pueden leerlo ANTES de fijar las
variables de entorno de hilos.

Funciona igual en Windows (pruebas) y Linux (IntelNUC).
"""
import os
from pathlib import Path

# ==============================================================
# 1. RUTAS
# ==============================================================
# LAB_DIR = carpeta donde esta ESTE archivo (.../lab3_multicamara).
LAB_DIR = Path(__file__).resolve().parent
#   BASE_MODELS = Path(r"D:\modelos")   (Windows)   |   Path("/home/usuario/modelos")   (Linux)
BASE_MODELS = LAB_DIR.parent / "edge_benchmark" / "modelos"
# Resultados del Lab 3 (separados del Lab 2): .../lab3_multicamara/results
RESULTS_DIR = LAB_DIR / "results"

# ==============================================================
# 2. DISPOSITIVO DE INFERENCIA
# ==============================================================
# "gpu" -> IntelNUC (Intel Iris Xe). "cpu" -> pruebas en el PC sin GPU.
DEVICE = "gpu" # TODO
OPENVINO_DEVICE_MAP = {"cpu": "intel:cpu", "gpu": "intel:gpu"}
# Si DEVICE="gpu" y OpenVINO no ve la GPU: False = no arrancar (recomendado en la NUC).
FALLBACK_TO_CPU_IF_NO_GPU = False

# ==============================================================
# 3. PRESUPUESTO DE HILOS (ver core/threads.py)
# ==============================================================
# CPUs que NO se entregan a los modelos:
#   RESERVED_CPUS_BASE           -> SO + hilo principal + monitor de hardware.
#   CPUS_PER_CAMERA x camaras    -> decodificar cada camara (2560x1440 ~ 1 nucleo).
RESERVED_CPUS_BASE = 1
CPUS_PER_CAMERA = 1
# Hilos por worker (par camara-modelo) para OpenVINO-CPU, torch, OpenMP y OpenCV.
# None = automatico: (CPUs - reservadas) // (modelos x camaras).
THREADS_PER_MODEL = None
OPENCV_THREADS = None            # None = igual a THREADS_PER_MODEL
FFMPEG_THREADS = 1               # por camara; 1 evita el error 'async_lock' del Lab 1
APPLY_OV_CPU_THREADS = True      # solo DEVICE="cpu"
# True = si el presupuesto da sobresuscripcion, main.py NO arranca (comprometeria el desempeno).
ABORT_ON_OVERSUBSCRIPTION = True

# ==============================================================
# 4. CAMARAS
# ==============================================================
# "replicar" -> abre la MISMA camara base CAMERA_REPLICAS veces (cam1, cam2, ...),
#               cada copia con su propia conexion. Sirve para simular N camaras con una.
# "lista"    -> usa las camaras de CAMERAS (pueden ser distintas y de tipos distintos).
CAMERA_MODE = "replicar"
CAMERA_REPLICAS = 2
# Maximo de camaras permitido (validacion). Subirlo solo si el presupuesto de hilos alcanza.
MAX_CAMERAS = 4

# --- Camara BASE (modo "replicar") ---
# Fuente: "rtsp" (NUC) | "file" (video .mp4 en el PC) | "webcam" (camara del PC; sin replicas)
CAMERA_SOURCE_TYPE = "rtsp"
# La URL NO se escribe aqui (tiene usuario y clave). Se lee de la variable de entorno:
#   Linux / tmux : export LAB2_RTSP_URL='rtsp://usuario:clave@IP:554/ruta'
#   PowerShell   : $env:LAB2_RTSP_URL = "rtsp://usuario:clave@IP:554/ruta"
RTSP_URL = os.environ.get("LAB2_RTSP_URL", "")
# Ruta COMPLETA al video (fuente "file"). En Windows usa r"...".
VIDEO_FILE_PATH = r""
WEBCAM_INDEX = 0

# --- Lista de camaras (modo "lista") ---
# type: "rtsp" (url_env = NOMBRE de la variable de entorno con la URL) | "file" (path) | "webcam" (index)
#   export LAB3_RTSP_URL_1='rtsp://...'   export LAB3_RTSP_URL_2='rtsp://...'
CAMERAS = [
    {"name": "cam1", "type": "rtsp", "url_env": "LAB3_RTSP_URL_1"},
    {"name": "cam2", "type": "rtsp", "url_env": "LAB3_RTSP_URL_2"},
    # {"name": "cam3", "type": "file", "path": r"C:\videos\prueba.mp4"},
]

# Comportamiento comun a todas las camaras
FILE_SIMULATE_REALTIME = True    # "file": entregar frames al ritmo real y en bucle
CAMERA_OPEN_TIMEOUT_MS = 10000
CAMERA_READ_TIMEOUT_MS = 10000
RECONNECT_INITIAL_BACKOFF_S = 2
RECONNECT_MAX_BACKOFF_S = 60
STALE_FRAME_TIMEOUT_S = 10       # sin frame nuevo en N s -> heartbeat marca esa camara ok=false

# ==============================================================
# 5. INFERENCIA (iguales al Lab 1 y 2)
# ==============================================================
MODEL_FORMAT = "OPENVINOFP16"
CONF_THRESHOLD = 0.25
IMGSZ = 640
CLASSES = None
WARMUP_FRAMES = 15
FRAME_WAIT_TIMEOUT_S = 1.0
MAX_CONSECUTIVE_ERRORS = 50

# ==============================================================
# 6. COMBINACIONES Y DURACION
# ==============================================================
# Cada combinacion = modelos que corren AL MISMO TIEMPO sobre TODAS las camaras.
# COMBINATIONS = {
#     "det+pose": ["yolo11n", "yolo11n-pose"],
#     "det+seg": ["yolo11n", "yolo11n-seg"],
#     "pose+seg": ["yolo11n-pose", "yolo11n-seg"],
# }
# COMBINATION_ORDER = ["det+pose", "det+seg", "pose+seg"]

COMBINATIONS = {
    "det":  ["yolo11n"],
    "pose": ["yolo11n-pose"],
    "seg":  ["yolo11n-seg"],
}
COMBINATION_ORDER = ["det", "pose", "seg"]

DURACION_COMBINACION_MIN = 240 # TODO
RESUME_MODE = "ask"              # "auto" | "ask" | "never"
SUPERVISOR_ESPERA_TRAS_FALLO_S = 30

# ==============================================================
# 7. REGISTRO DE DATOS
# ==============================================================
SCHEMA_VERSION = "3.0"           # 3.0 = archivos por camara + heartbeat por camara
BATCH_WRITE_FRAMES = 30
SUMMARY_WINDOW_S = 600
ROTATE_EVERY_S = 3600
COMPRESS_CLOSED_FILES = True
SAVE_DETECTIONS = True
WRITE_EMPTY_DETECTIONS = False
SAVE_KEYPOINTS = True
SAVE_POLYGONS = True
POLYGON_EPSILON_PX = 2.0
COORD_DECIMALS = 0
MIN_FREE_DISK_GB = 20

# ==============================================================
# 8. EVIDENCIA VISUAL
# ==============================================================
SAVE_FRAMES = False
SAVE_FRAMES_EVERY_S = 3600

# ==============================================================
# 9. HARDWARE
# ==============================================================
HW_SAMPLE_INTERVAL_S = 2
HW_ENABLE = {
    "cpu": True, "cpu_per_core": True, "cpu_freq": True,
    "load_avg": True,             # en Windows se desactiva solo
    "ram": True, "swap": True, "process": True, "threads_cpu": True,
    "disk_io": True, "disk_usage": True, "net_io": True,
    "temp": True,                 # Linux; en VMs suele no existir
    "gpu_freq_sysfs": True,       # solo Linux
    "gpu_intel_gpu_top": True,    # solo Linux con intel-gpu-tools y sudo
}
INTEL_GPU_TOP_CMD = ["sudo", "-n", "intel_gpu_top", "-J", "-s", "2000"]
HW_HISTORY_S = 3600

# ==============================================================
# 10. HEARTBEAT, STATUS Y LOGS
# ==============================================================
HEARTBEAT_INTERVAL_S = 10
STATUS_INTERVAL_S = 30
MAIN_LOOP_TICK_S = 1.0
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"


def camera_source():
    """Fuente de la camara BASE (modo "replicar") segun CAMERA_SOURCE_TYPE."""
    return {"rtsp": RTSP_URL, "file": VIDEO_FILE_PATH, "webcam": WEBCAM_INDEX}[CAMERA_SOURCE_TYPE]


def duracion_combinacion_s() -> float:
    return DURACION_COMBINACION_MIN * 60


def max_models_per_combination() -> int:
    """Mayor numero de modelos simultaneos en cualquier combinacion."""
    return max(len(COMBINATIONS[name]) for name in COMBINATION_ORDER)


def openvino_device() -> str:
    return OPENVINO_DEVICE_MAP[DEVICE]


def validate_config() -> list[str]:
    """Revisa la configuracion GENERAL y devuelve TODOS los errores juntos.

    Las camaras se validan en core.camera_factory.camera_spec_errors() (SRP).
    """
    errors = []
    if DEVICE not in OPENVINO_DEVICE_MAP:
        errors.append(f"DEVICE debe ser 'cpu' o 'gpu' (es {DEVICE!r})")
    if RESUME_MODE not in ("auto", "ask", "never"):
        errors.append(f"RESUME_MODE invalido: {RESUME_MODE!r}")
    if not COMBINATION_ORDER or len(set(COMBINATION_ORDER)) != len(COMBINATION_ORDER):
        errors.append("COMBINATION_ORDER debe tener combinaciones sin repetir")
    for name in COMBINATION_ORDER:
        if name not in COMBINATIONS:
            errors.append(f"COMBINATION_ORDER usa una combinacion inexistente: {name}")
    if not BASE_MODELS.is_dir():
        errors.append(f"No existe la carpeta de modelos BASE_MODELS = {BASE_MODELS}")
    positives = {
        "DURACION_COMBINACION_MIN": DURACION_COMBINACION_MIN, "SUMMARY_WINDOW_S": SUMMARY_WINDOW_S,
        "ROTATE_EVERY_S": ROTATE_EVERY_S, "HW_SAMPLE_INTERVAL_S": HW_SAMPLE_INTERVAL_S,
        "BATCH_WRITE_FRAMES": BATCH_WRITE_FRAMES, "HEARTBEAT_INTERVAL_S": HEARTBEAT_INTERVAL_S,
        "SAVE_FRAMES_EVERY_S": SAVE_FRAMES_EVERY_S, "FRAME_WAIT_TIMEOUT_S": FRAME_WAIT_TIMEOUT_S,
        "MAX_CAMERAS": MAX_CAMERAS,
    }
    for name, value in positives.items():
        if value <= 0:
            errors.append(f"{name} debe ser > 0")
    if RESERVED_CPUS_BASE < 0 or CPUS_PER_CAMERA < 0:
        errors.append("RESERVED_CPUS_BASE y CPUS_PER_CAMERA deben ser >= 0")
    if THREADS_PER_MODEL is not None and THREADS_PER_MODEL < 1:
        errors.append("THREADS_PER_MODEL debe ser None o >= 1")
    return errors
