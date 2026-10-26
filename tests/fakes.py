"""Objetos falsos para probar sin OpenVINO ni Ultralytics."""
import time
from types import SimpleNamespace

import numpy as np

from core.engine import InferenceResult


class _Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.cls)


def fake_result(task: str, n: int = 2):
    """Imita un Results de Ultralytics con n personas."""
    boxes = _Boxes(xyxy=np.array([[10.4, 20.6, 110.2, 220.9]] * n, dtype=np.float32).reshape(n, 4),
                   conf=np.array([0.9] * n, dtype=np.float32),
                   cls=np.zeros(n, dtype=np.float32))
    res = SimpleNamespace(boxes=boxes, names={0: "person"}, keypoints=None, masks=None,
                          speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 1.0},
                          plot=lambda: np.zeros((10, 10, 3), np.uint8))
    if task == "pose":
        res.keypoints = SimpleNamespace(data=np.random.rand(n, 17, 3).astype(np.float32) * 100)
    if task == "segmentacion":
        square = np.array([[0, 0], [50, 0], [50, 1], [50, 50], [0, 50]], dtype=np.float32)
        res.masks = SimpleNamespace(xy=[square] * n)
    return res


class FakeModel:
    """Misma interfaz publica que core.engine.OpenVinoModel."""

    def __init__(self, info, cfg, delay_s: float = 0.005):
        self.info, self.cfg, self.delay_s = info, cfg, delay_s
        self.device = cfg.openvino_device()
        self.load_time_s = 0.0

    def load(self):
        self.load_time_s = 0.01
        return self.load_time_s

    def predict(self, frame):
        time.sleep(self.delay_s)
        return InferenceResult(fake_result(self.info.task), self.delay_s * 1000, 1.0, 5.0, 1.0)

    def limit_cpu_threads(self):
        return False

    def effective_device(self):
        return "FAKE"

    def release(self):
        pass
