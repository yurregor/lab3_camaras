"""
Ejecuta el Laboratorio 3 multicamara (Windows y Linux).

Normalmente se lanza run_lab3.py (supervisor). Tambien se puede lanzar directo:
    python main.py
    python main.py --reanudar      # fuerza continuar el run interrumpido
    python main.py --nuevo         # fuerza un run nuevo

El ORDEN de este archivo importa:
1. Leer config y resolver las camaras (solo libreria estandar).
2. Calcular el presupuesto de hilos; si hay sobresuscripcion, NO arrancar.
3. Fijar las variables de entorno de hilos.
4. Recien entonces importar cv2 / numpy / torch / openvino (via core.*).
"""
import argparse
import logging
import sys

import config
from core.camera_factory import build_camera_specs, camera_spec_errors
from core.exit_codes import EXIT_COMPLETED, EXIT_CONFIG_ERROR, EXIT_INTERRUPTED
from core.threads import apply_thread_env, compute_budget


def setup_logging() -> None:
    """Log a consola y a results/lab3.log con el hilo en cada linea."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.RESULTS_DIR / "lab3.log", encoding="utf-8")])


def fail_config(message: str) -> None:
    """Error que no se arregla relanzando: el supervisor se detiene con este codigo."""
    print(f"\n[CONFIG] {message}\n", file=sys.stderr)
    sys.exit(EXIT_CONFIG_ERROR)


def check_device(logger) -> None:
    """Falla rapido si se pidio GPU y OpenVINO no la ve (o cae a CPU si se permite)."""
    from core.engine import available_openvino_devices
    devices = available_openvino_devices()
    logger.info("Dispositivos OpenVINO disponibles: %s", devices)
    if config.DEVICE == "gpu" and not any(d.startswith("GPU") for d in devices):
        if not config.FALLBACK_TO_CPU_IF_NO_GPU:
            fail_config("DEVICE='gpu' pero OpenVINO no ve ninguna GPU (revisa drivers / /dev/dri).")
        logger.warning("Sin GPU: se continua en CPU (FALLBACK_TO_CPU_IF_NO_GPU=True)")
        config.DEVICE = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 3: inferencia concurrente multicamara")
    grupo = parser.add_mutually_exclusive_group()          # no permite usar los dos a la vez
    grupo.add_argument("--reanudar", action="store_true", help="continuar el run interrumpido")
    grupo.add_argument("--nuevo", action="store_true", help="iniciar un run nuevo")
    return parser.parse_args()


def build_budget(n_cameras: int):
    """Presupuesto de hilos; si hay sobresuscripcion y ABORT_ON_OVERSUBSCRIPTION, no arranca."""
    budget = compute_budget(config.max_models_per_combination(), n_cameras,
                            config.RESERVED_CPUS_BASE, config.CPUS_PER_CAMERA,
                            config.THREADS_PER_MODEL, config.OPENCV_THREADS, config.FFMPEG_THREADS)
    if budget.sobresuscripcion and config.ABORT_ON_OVERSUBSCRIPTION:
        fail_config("Sobresuscripcion de hilos: " + budget.motivo +
                    f".\n  Presupuesto: {budget.as_dict()}"
                    "\n  Opciones: menos camaras (CAMERA_REPLICAS / CAMERAS), menos modelos por"
                    " combinacion, o ajustar RESERVED_CPUS_BASE / CPUS_PER_CAMERA en config.py.")
    return budget


def main() -> int:
    args = parse_args()
    errors = config.validate_config() + camera_spec_errors(config)
    if errors:
        fail_config("Configuracion invalida:\n  - " + "\n  - ".join(errors))
    specs = build_camera_specs(config)

    budget = build_budget(len(specs))
    apply_thread_env(budget)          # ANTES de importar librerias pesadas

    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Camaras (%s): %s", config.CAMERA_MODE, [s.describe() for s in specs])
    logger.info("Presupuesto de hilos: %s", budget.as_dict())
    logger.info("Combinaciones: %s | %s min cada una",
                config.COMBINATION_ORDER, config.DURACION_COMBINACION_MIN)
    if budget.sobresuscripcion:
        logger.warning("Sobresuscripcion (%s): se continua porque ABORT_ON_OVERSUBSCRIPTION=False",
                       budget.motivo)

    from core.threads import apply_library_threads
    apply_library_threads(budget)
    check_device(logger)

    from core.orchestrator import COMPLETED, Orchestrator
    result = Orchestrator(config, budget, specs, force_resume=args.reanudar,
                          force_new=args.nuevo).run()
    return EXIT_COMPLETED if result == COMPLETED else EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
