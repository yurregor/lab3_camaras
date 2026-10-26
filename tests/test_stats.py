from core.stats import WindowStats


def test_resumen_basico():
    w = WindowStats()
    for _ in range(10):
        w.add(fps=30, lat=10, pre=1, inf=8, post=1, edad=20, n_det=2, conf_mean=0.8, saltados=1)
    s = w.summarize(duracion_s=2.0)
    assert s["fps_mean"] == 5.0 and s["cobertura_pct"] == 50.0


def test_ventana_vacia_no_falla():
    s = WindowStats().summarize(10)
    assert s["fps_mean"] == 0 and s["latency_p50_ms"] is None
