"""Fabrica de camaras: modo replicar, modo lista y validaciones."""
from types import SimpleNamespace

from core.camera_factory import build_camera_specs, camera_spec_errors


def cfg(**kw):
    base = dict(CAMERA_MODE="replicar", CAMERA_REPLICAS=3, MAX_CAMERAS=4,
                CAMERA_SOURCE_TYPE="rtsp", RTSP_URL="rtsp://u:p@1.2.3.4/x",
                VIDEO_FILE_PATH="", WEBCAM_INDEX=0, CAMERAS=[])
    base.update(kw)
    c = SimpleNamespace(**base)
    c.camera_source = lambda: {"rtsp": c.RTSP_URL, "file": c.VIDEO_FILE_PATH,
                               "webcam": c.WEBCAM_INDEX}[c.CAMERA_SOURCE_TYPE]
    return c


def test_replicar_crea_n_camaras_misma_fuente():
    specs = build_camera_specs(cfg())
    assert [s.name for s in specs] == ["cam1", "cam2", "cam3"]
    assert {s.source for s in specs} == {"rtsp://u:p@1.2.3.4/x"}
    assert camera_spec_errors(cfg()) == []


def test_describe_oculta_credenciales():
    assert build_camera_specs(cfg())[0].describe()["fuente"] == "***"


def test_replicar_sin_url_falla():
    assert any("LAB2_RTSP_URL" in e for e in camera_spec_errors(cfg(RTSP_URL="")))


def test_maximo_de_camaras_configurable():
    assert any("MAX_CAMERAS" in e for e in camera_spec_errors(cfg(CAMERA_REPLICAS=5)))
    assert camera_spec_errors(cfg(CAMERA_REPLICAS=5, MAX_CAMERAS=6)) == []


def test_webcam_no_se_replica():
    assert camera_spec_errors(cfg(CAMERA_SOURCE_TYPE="webcam", CAMERA_REPLICAS=2))


def test_lista_mixta_rtsp_y_file(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("URL_A", "rtsp://a")
    c = cfg(CAMERA_MODE="lista", CAMERAS=[
        {"name": "entrada", "type": "rtsp", "url_env": "URL_A"},
        {"name": "bodega", "type": "file", "path": str(video)}])
    assert camera_spec_errors(c) == []
    specs = build_camera_specs(c)
    assert [(s.name, s.source_type, s.source) for s in specs] == [
        ("entrada", "rtsp", "rtsp://a"), ("bodega", "file", str(video))]


def test_lista_reporta_todos_los_errores(monkeypatch):
    monkeypatch.delenv("NO_EXISTE", raising=False)
    c = cfg(CAMERA_MODE="lista", CAMERAS=[
        {"name": "a", "type": "rtsp", "url_env": "NO_EXISTE"},
        {"name": "b", "type": "file", "path": "/no/existe.mp4"},
        {"name": "c", "type": "otro"}])
    assert len(camera_spec_errors(c)) == 3


def test_nombres_repetidos(monkeypatch):
    monkeypatch.setenv("U", "rtsp://a")
    c = cfg(CAMERA_MODE="lista", CAMERAS=[{"name": "x", "type": "rtsp", "url_env": "U"}] * 2)
    assert any("repetidos" in e for e in camera_spec_errors(c))
