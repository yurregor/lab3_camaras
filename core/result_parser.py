"""
Convierte un Results de Ultralytics en diccionarios listos para JSON (sin cambios).

Funciones PURAS: se prueban sin modelo (tests/test_result_parser.py).
Por deteccion: class_id, class_name, conf, bbox_xyxy y, segun la tarea,
keypoints (pose) o mask_area_px + polygon (segmentacion).
"""
import cv2
import numpy as np


def _to_numpy(value) -> np.ndarray:
    """Tensor de torch o array -> numpy (sin depender de importar torch)."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _round_list(arr: np.ndarray, decimals: int) -> list:
    """Redondea y convierte a lista; con 0 decimales usa enteros (menos bytes)."""
    if decimals == 0:
        return np.rint(arr).astype(int).tolist()
    return np.round(arr.astype(float), decimals).tolist()


def simplify_polygon(points: np.ndarray, epsilon_px: float) -> np.ndarray:
    """Reduce los vertices de un poligono (Douglas-Peucker). 0 = no simplificar."""
    if epsilon_px <= 0 or len(points) < 4:
        return points
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)


def parse_detections(result, cfg) -> list[dict]:
    """Lista de detecciones de un frame, con la informacion propia de la tarea."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls).astype(int)
    names = getattr(result, "names", {}) or {}
    keypoints = _keypoints(result) if cfg.SAVE_KEYPOINTS else None
    polygons = _polygons(result) if cfg.SAVE_POLYGONS else None

    detections = []
    for i in range(len(classes)):
        det = {"class_id": int(classes[i]),
               "class_name": names.get(int(classes[i]), str(classes[i])),
               "conf": round(float(confs[i]), 4),
               "bbox_xyxy": _round_list(xyxy[i], cfg.COORD_DECIMALS)}
        if keypoints is not None and i < len(keypoints):
            det["keypoints"] = _format_keypoints(keypoints[i], cfg.COORD_DECIMALS)
        if polygons is not None and i < len(polygons):
            det.update(_format_polygon(polygons[i], cfg))
        detections.append(det)
    return detections


def frame_counts(detections: list[dict]) -> dict:
    """Resumen del frame para frames.jsonl: cantidad, por clase y confianza media."""
    por_clase: dict[str, int] = {}
    for det in detections:
        por_clase[det["class_name"]] = por_clase.get(det["class_name"], 0) + 1
    confs = [d["conf"] for d in detections]
    return {"n_detections": len(detections),
            "conf_mean": round(float(np.mean(confs)), 4) if confs else None,
            "n_por_clase": por_clase}


def _keypoints(result) -> np.ndarray | None:
    kp = getattr(result, "keypoints", None)
    if kp is None or getattr(kp, "data", None) is None:
        return None
    return _to_numpy(kp.data)


def _polygons(result) -> list | None:
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    return list(masks.xy)


def _format_keypoints(kps: np.ndarray, decimals: int) -> list:
    out = []
    for point in kps:
        xy = _round_list(point[:2], decimals)
        conf = round(float(point[2]), 3) if len(point) > 2 else None
        out.append([*xy, conf])
    return out


def _format_polygon(points, cfg) -> dict:
    """Area de la mascara (sobre el poligono original) y poligono simplificado."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return {"mask_area_px": 0, "polygon": []}
    area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
    simple = simplify_polygon(pts, cfg.POLYGON_EPSILON_PX)
    return {"mask_area_px": int(round(area)),
            "polygon": _round_list(simple, cfg.COORD_DECIMALS)}
