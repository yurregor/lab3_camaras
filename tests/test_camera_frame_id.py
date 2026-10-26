import threading
from types import SimpleNamespace

import cv2
import numpy as np

from core.camera import CameraStream

CFG = SimpleNamespace(FILE_SIMULATE_REALTIME=True, CAMERA_READ_TIMEOUT_MS=2000,
                      RECONNECT_INITIAL_BACKOFF_S=1, RECONNECT_MAX_BACKOFF_S=2,
                      STALE_FRAME_TIMEOUT_S=10)


def make_video(path, n=30, fps=30):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        out.write(np.full((48, 64, 3), i, np.uint8))
    out.release()


def test_ids_crecientes_y_sin_repetidos(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    cam = CameraStream("t", str(video), "file", CFG)
    assert cam.start()
    seen, last = [], cam.frame_id
    for _ in range(40):
        fid, frame, _ = cam.wait_new_frame(last, 1.0)
        assert frame is not None and fid > last
        seen.append(fid)
        last = fid
    cam.stop()
    assert seen == sorted(set(seen))


def test_hilo_con_nombre_de_camara(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    cam = CameraStream("cam7", str(video), "file", CFG)
    cam.start()
    assert cam._thread.name == "cam-capture-cam7" and cam.is_ok()
    cam.stop()


def test_dos_consumidores_ven_frames_nuevos(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    cam = CameraStream("t", str(video), "file", CFG)
    cam.start()
    results = {}

    def consume(key):
        last, got = cam.frame_id, 0
        while got < 10:
            fid, frame, _ = cam.wait_new_frame(last, 1.0)
            if frame is not None:
                last, got = fid, got + 1
        results[key] = got

    threads = [threading.Thread(target=consume, args=(k,)) for k in "ab"]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    cam.stop()
    assert results == {"a": 10, "b": 10}
