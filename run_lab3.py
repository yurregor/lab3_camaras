"""
Supervisor del Laboratorio 3. Funciona en Windows y Linux. ESTE es el que se lanza.

Igual al run_lab2.py del Lab 2 (sin cambios de logica):
1. Si main.py se cae por un error inesperado, lo relanza y continua donde iba.
2. Ctrl+C NO detiene la corrida de inmediato: pregunta primero. main.py corre
   en su propio grupo de procesos, asi que el Ctrl+C de la consola SOLO le
   llega a este supervisor.
3. Con RESUME_MODE="ask", la pregunta de reanudar la hace este supervisor.

Decide segun el codigo de salida de main.py (core/exit_codes.py):
    0  todas las combinaciones terminaron      -> termina
    2  error de configuracion / sobresuscripcion -> termina (relanzar no lo arregla)
    4  detenido por el usuario                   -> termina
    otro (caida inesperada)                      -> espera y relanza

Uso:  python -u run_lab3.py
Solo libreria estandar: no carga nada pesado.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from core.exit_codes import EXIT_COMPLETED, EXIT_CONFIG_ERROR, EXIT_INTERRUPTED
from core.status import find_resumable

HERE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
STOP_CODES = {EXIT_COMPLETED: "todas las combinaciones terminaron",
              EXIT_CONFIG_ERROR: "error de configuracion",
              EXIT_INTERRUPTED: "detenido por el usuario"}
CONFIRM_WORD = "s"           # respuesta que hay que escribir para detener
POLL_S = 0.5                 # cada cuanto se revisa si main.py termino o hubo Ctrl+C


def log(message: str) -> None:
    print(f"[supervisor {datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def record_restart(code: int) -> None:
    """Deja rastro de cada relanzamiento en results/reinicios.jsonl."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "reinicios.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now().isoformat(), "codigo_salida": code}) + "\n")


def install_ctrl_c_flag(current: dict) -> None:
    """Ctrl+C solo marca la bandera; la pregunta se hace en el bucle principal."""
    def handler(_signum, _frame):
        current["ctrl_c"] = True
    signal.signal(signal.SIGINT, handler)


def confirm_stop(current: dict) -> bool:
    """Pregunta si se quiere detener. Cualquier respuesta distinta de 's' = continuar."""
    try:
        answer = input("\n[supervisor] Ctrl+C recibido. La corrida SIGUE ejecutandose.\n"
                       "[supervisor] DETENER ejecucion? (s/n) ")
    except (EOFError, OSError, KeyboardInterrupt):
        print()
        answer = ""
    current["ctrl_c"] = False
    return answer.strip().lower() == CONFIRM_WORD


def handle_ctrl_c(current: dict, proc: subprocess.Popen | None, user_stop: bool) -> bool:
    """Atiende un Ctrl+C pendiente. Retorna el nuevo valor de user_stop."""
    current["ctrl_c"] = False
    if user_stop:
        log("Ya se pidio detener; esperando el cierre ordenado...")
        return True
    if confirm_stop(current):
        log("Deteniendo: main.py guarda resumenes y status (puede tardar unos segundos)...")
        if proc is not None:
            request_child_stop(proc)
        return True
    log("Continua la ejecucion normalmente (main.py nunca se detuvo).")
    return False


def start_child(extra_args: list[str]) -> subprocess.Popen:
    """Lanza main.py en un grupo de procesos PROPIO (el Ctrl+C de la consola no le llega)."""
    cmd = [sys.executable, str(HERE / "main.py"), *extra_args]
    if IS_WINDOWS:
        return subprocess.Popen(cmd, cwd=HERE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, cwd=HERE, process_group=0)


def request_child_stop(proc: subprocess.Popen) -> None:
    """Pide a main.py que se detenga ORDENADAMENTE (SIGTERM en Linux, CTRL_BREAK en Windows)."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGTERM)


def run_child(extra_args: list[str], current: dict) -> tuple[int, bool]:
    """Ejecuta main.py y espera a que termine. Retorna (codigo, detenido_por_usuario)."""
    proc = start_child(extra_args)
    current["proc"] = proc
    user_stop = False
    while True:
        code = proc.poll()
        if code is not None:
            return code, user_stop or current["detener"]
        if current["ctrl_c"]:
            user_stop = handle_ctrl_c(current, proc, user_stop)
        time.sleep(POLL_S)


def install_forwarding(current: dict) -> None:
    """SIGTERM/SIGHUP al supervisor (kill, tmux kill-session): detener main.py ordenadamente."""
    def handler(_signum, _frame):
        current["detener"] = True
        proc = current["proc"]
        if proc is None or proc.poll() is not None:
            sys.exit(EXIT_INTERRUPTED)
        request_child_stop(proc)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def sleep_or_confirm(seconds: float, current: dict) -> bool:
    """Espera antes de relanzar. Retorna False si el usuario confirma detener."""
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(POLL_S, max(0.0, end - time.time())))
        if current["ctrl_c"] and handle_ctrl_c(current, None, False):
            return False
    return True


def initial_args() -> list[str]:
    """Con RESUME_MODE="ask", la pregunta la hace el supervisor (esta en primer plano)."""
    if config.RESUME_MODE != "ask":
        return []
    previous = find_resumable(config.RESULTS_DIR)
    if previous is None:
        return []
    try:
        answer = input(f"Run interrumpido {previous.name}. Reanudar? (s/n): ")
    except (EOFError, OSError):
        answer = "s"
    flag = "--reanudar" if answer.strip().lower() == "s" else "--nuevo"
    log("Se reanuda el run anterior." if flag == "--reanudar" else "Se inicia un run NUEVO.")
    return [flag]


def main() -> int:
    current = {"proc": None, "detener": False, "ctrl_c": False}
    install_ctrl_c_flag(current)
    install_forwarding(current)
    args: list[str] = initial_args()
    while True:
        log(f"Lanzando main.py {' '.join(args)}")
        code, user_stop = run_child(args, current)
        if user_stop or code in STOP_CODES:
            log(f"Fin (codigo {code}: {STOP_CODES.get(code, 'detenido por el usuario')}).")
            return code if code == EXIT_CONFIG_ERROR else 0
        record_restart(code)
        log(f"main.py se cayo (codigo {code}). Relanzo en "
            f"{config.SUPERVISOR_ESPERA_TRAS_FALLO_S} s y continua donde iba.")
        if not sleep_or_confirm(config.SUPERVISOR_ESPERA_TRAS_FALLO_S, current):
            log("Cancelado por el usuario.")
            return 0
        args = ["--reanudar"]


if __name__ == "__main__":
    sys.exit(main())
