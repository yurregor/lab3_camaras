"""Punta a punta con modelos falsos y .mp4: 2 camaras replicadas y modo lista."""
import gzip
import json
import threading
from datetime import datetime

import config
from core.camera_factory import build_camera_specs
from core.orchestrator import COMPLETED, INTERRUPTED, Orchestrator
from core.threads import compute_budget
from fakes import FakeModel
from test_camera_frame_id import make_video

DURACION_S = 3


def _prepare(tmp_path, monkeypatch, **extra):
    for task, name in (("deteccion", "yolo11n"), ("pose", "yolo11n-pose"),
                       ("segmentacion", "yolo11n-seg")):
        folder = tmp_path / "modelos" / task / "yolo11" / "OPENVINOFP16" / f"{name}_openvino_model"
        folder.mkdir(parents=True)
        (folder / f"{name}.xml").write_text("<xml/>")
    video = tmp_path / "v.mp4"
    make_video(video, n=60)
    overrides = dict(
        BASE_MODELS=tmp_path / "modelos", RESULTS_DIR=tmp_path / "results",
        CAMERA_MODE="replicar", CAMERA_REPLICAS=2, CAMERA_SOURCE_TYPE="file",
        VIDEO_FILE_PATH=str(video), DEVICE="cpu", MIN_FREE_DISK_GB=0,
        WARMUP_FRAMES=2, SUMMARY_WINDOW_S=1, HEARTBEAT_INTERVAL_S=1, STATUS_INTERVAL_S=1,
        HW_SAMPLE_INTERVAL_S=0.5, HW_ENABLE={**config.HW_ENABLE, "gpu_intel_gpu_top": False},
        COMBINATIONS={"det+pose": ["yolo11n", "yolo11n-pose"], "det+seg": ["yolo11n", "yolo11n-seg"]},
        COMBINATION_ORDER=["det+pose", "det+seg"], DURACION_COMBINACION_MIN=DURACION_S / 60)
    overrides.update(extra)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)
    return video


def _orch(force_resume=False):
    specs = build_camera_specs(config)
    budget = compute_budget(2, len(specs), 1, 1, None, None, 1, cpus=16)
    return Orchestrator(config, budget, specs, force_resume=force_resume,
                        model_factory=lambda info: FakeModel(info, config))


def _lines(run_dir, pattern):
    out = []
    for p in run_dir.glob(pattern + "*"):
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh]
    return out


def _status(tmp_path):
    return json.loads(next((tmp_path / "results").glob("run_*_status.json")).read_text())


def test_dos_camaras_replicadas(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    assert config.validate_config() == []
    assert _orch().run() == COMPLETED
    status = _status(tmp_path)
    assert status["status"] == "completed" and status["camaras"] == ["cam1", "cam2"]
    run_dir = tmp_path / "results" / status["run_id"]

    # Un archivo de frames por (camara, modelo), y solo frames unicos en cada uno
    for cam in ("cam1", "cam2"):
        for model in ("yolo11n", "yolo11n-pose"):
            frames = _lines(run_dir, f"*_det+pose_frames_{cam}_{model}_")
            assert frames and {f["camara"] for f in frames} == {cam}
            ids = [f["frame_id"] for f in frames]
            assert ids == sorted(set(ids))
    dets = _lines(run_dir, "*_det+seg_detections_cam2_yolo11n-seg_")
    assert dets[0]["camara"] == "cam2" and "polygon" in dets[0]["detecciones"][0]

    summary = _lines(run_dir, "*_det+pose_summary_")
    assert {(s["camara"], s["modelo"]) for s in summary} == {
        ("cam1", "yolo11n"), ("cam1", "yolo11n-pose"), ("cam2", "yolo11n"), ("cam2", "yolo11n-pose")}
    assert all(s["n_camaras_config"] == 2 for s in summary)

    events = _lines(run_dir, "*_events.jsonl")
    start = next(e for e in events if e["tipo"] == "combo_start")
    assert start["workers"] == 4
    end_1 = next(e for e in events if e["tipo"] == "combo_end" and e["combinacion"] == "det+pose")
    start_2 = next(e for e in events if e["tipo"] == "combo_start" and e["combinacion"] == "det+seg")
    gap = (datetime.fromisoformat(start_2["ts"]) - datetime.fromisoformat(end_1["ts"])).total_seconds()
    assert gap < 1.0
    assert not any(e["tipo"] == "camera_disconnected" for e in events)

    hb = json.loads((tmp_path / "results" / "heartbeat.json").read_text())
    assert hb["status"] == "finished"


def test_heartbeat_por_camara(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch, DURACION_COMBINACION_MIN=20 / 60)
    orch = _orch()
    threading.Timer(4.0, orch.stop_event.set).start()
    seen = {}

    def snap():
        seen.update(json.loads((tmp_path / "results" / "heartbeat.json").read_text()))
    threading.Timer(3.0, snap).start()
    assert orch.run() == INTERRUPTED
    assert set(seen["camaras"]) == {"cam1", "cam2"} and seen["camaras_ok"] is True
    assert {"cam1/yolo11n", "cam2/yolo11n-pose"} <= set(seen["fps_actual"])


def test_modo_lista(tmp_path, monkeypatch):
    video = _prepare(tmp_path, monkeypatch, COMBINATION_ORDER=["det+pose"])
    monkeypatch.setattr(config, "CAMERA_MODE", "lista")
    monkeypatch.setattr(config, "CAMERAS", [{"name": "entrada", "type": "file", "path": str(video)},
                                            {"name": "bodega", "type": "file", "path": str(video)},
                                            {"name": "patio", "type": "file", "path": str(video)}])
    assert _orch().run() == COMPLETED
    run_dir = tmp_path / "results" / _status(tmp_path)["run_id"]
    assert {s["camara"] for s in _lines(run_dir, "*_summary_")} == {"entrada", "bodega", "patio"}


def test_reanuda_con_los_segundos_faltantes(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    orch = _orch()
    threading.Timer(1.5, orch.stop_event.set).start()
    assert orch.run() == INTERRUPTED
    assert 0 < _status(tmp_path)["combos"]["det+pose"]["segundos"] < DURACION_S
    assert _orch(force_resume=True).run() == COMPLETED
    status = _status(tmp_path)
    assert status["reinicios"] == 1
    assert abs(status["combos"]["det+pose"]["segundos"] - DURACION_S) < 1.5
