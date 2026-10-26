import gzip
import json
import time

from core.writers import Compressor, JsonlWriter


def test_lotes_y_cierre(tmp_path):
    w = JsonlWriter(tmp_path, "x", batch_size=3, rotate_every_s=None)
    for i in range(5):
        w.write({"i": i})
    w.close()
    assert [json.loads(l)["i"] for l in (tmp_path / "x.jsonl").read_text().splitlines()] == [0, 1, 2, 3, 4]


def test_rotacion_y_compresion(tmp_path):
    comp = Compressor(True)
    w = JsonlWriter(tmp_path, "r", batch_size=1, rotate_every_s=1, compressor=comp)
    w.write({"a": 1})
    time.sleep(1.1)
    w.write({"a": 2})
    w.close()
    comp.wait_all(timeout_s=10)
    gz = sorted(tmp_path.glob("r_*.jsonl.gz"))
    assert len(gz) == 2
    with gzip.open(gz[0], "rt") as fh:
        assert json.loads(fh.readline())["a"] == 1


def test_compresion_sin_gzip_externo(tmp_path):
    comp = Compressor(True)
    comp._gzip_bin = None
    w = JsonlWriter(tmp_path, "w", batch_size=1, rotate_every_s=3600, compressor=comp)
    w.write({"a": 1})
    w.close()
    comp.wait_all(timeout_s=10)
    assert len(list(tmp_path.glob("w_*.jsonl.gz"))) == 1


def test_no_pisa_gz_existente(tmp_path):
    comp = Compressor(True)
    for value in (1, 2):
        w = JsonlWriter(tmp_path, "s", batch_size=1, rotate_every_s=3600, compressor=comp)
        w.write({"v": value})
        w.close()
        comp.wait_all(timeout_s=10)
    assert len(list(tmp_path.glob("s_*.jsonl.gz"))) == 2
