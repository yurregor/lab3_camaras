"""Presupuesto de hilos multicamara (CPUs fijas para que el test sea reproducible)."""
from core.threads import compute_budget


def budget(models, cams, cpus=8, per_model=None):
    return compute_budget(models, cams, reserved_base=1, cpus_per_camera=1,
                          per_model=per_model, opencv_threads=None, ffmpeg_threads=1, cpus=cpus)


def test_nuc_2_camaras_2_modelos_cabe():
    b = budget(2, 2)
    assert (b.cpus_reservadas, b.workers, b.hilos_por_modelo) == (3, 4, 1)
    assert not b.sobresuscripcion


def test_nuc_1_camara_igual_que_lab2():
    b = budget(2, 1)
    assert b.hilos_por_modelo == 3 and not b.sobresuscripcion


def test_nuc_4_camaras_2_modelos_no_cabe():
    b = budget(2, 4)
    assert b.sobresuscripcion and "no alcanza" in b.motivo


def test_hilos_forzados_que_exceden_cpus():
    b = budget(2, 2, per_model=3)
    assert b.sobresuscripcion and "computo estimado" in b.motivo
