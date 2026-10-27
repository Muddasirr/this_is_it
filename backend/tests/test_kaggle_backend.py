import io
import json
import zipfile


def test_result_zip_becomes_detections_thumbs_and_spotted(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.services.kaggle_backend import KaggleTrackBackend

    monkeypatch.setenv("VISION_THUMB_DIR", str(tmp_path / "thumbs"))
    get_settings.cache_clear()
    det = {"track_key": "A-p1", "camera_label": "A", "start_ms": 0, "end_ms": 1000, "confidence": 0.9,
           "bbox_series": [{"ts_ms": 0, "x1": 1, "y1": 2, "x2": 3, "y2": 4}], "class_name": "person",
           "features": {"thumb_path": "/kaggle/working/thumbs/A/p1.jpg"}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps({"detections": [det], "stats": {"frames_processed": 10},
                                               "spotted": [{"ts_ms": 500, "event_type": "pass", "confidence": 0.5}]}))
        zf.writestr("thumbs/p1.jpg", b"jpg")
    b = KaggleTrackBackend()
    dets = b._parse(buf.getvalue(), "A", "job1", 12.0)
    assert len(dets) == 1 and dets[0].track_key == "A-p1"
    thumb = dets[0].features["thumb_path"]
    assert thumb.endswith("/A/p1.jpg") and open(thumb, "rb").read() == b"jpg"
    assert b.spotted["A"][0]["event_type"] == "pass"
    assert b.last_run_stats["backend"] == "kaggle" and b.last_run_stats["frames_processed"] == 10
    get_settings.cache_clear()


def test_retry_rides_out_network_blips(monkeypatch):
    import requests

    from app.services import kaggle_backend

    monkeypatch.setattr(kaggle_backend.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("Temporary failure in name resolution")
        return "ok"

    assert kaggle_backend._retry(flaky) == "ok" and len(calls) == 3


def test_retry_does_not_retry_client_errors(monkeypatch):
    import pytest
    import requests

    from app.services import kaggle_backend

    monkeypatch.setattr(kaggle_backend.time, "sleep", lambda s: None)
    resp = requests.Response()
    resp.status_code = 403
    calls = []

    def forbidden():
        calls.append(1)
        raise requests.exceptions.HTTPError("403", response=resp)

    with pytest.raises(requests.exceptions.HTTPError):
        kaggle_backend._retry(forbidden)
    assert len(calls) == 1


def test_restarted_worker_reattaches_to_its_gpu_run(tmp_path, monkeypatch):
    """A worker killed mid-job must not lose (or re-pay for) the GPU run: the second attempt on the
    same video skips the upload/push, waits on the recorded job and takes its result."""
    from app.services import kaggle_backend
    from app.services.kaggle_backend import KaggleTrackBackend

    monkeypatch.setattr(kaggle_backend, "STATE_DIR", tmp_path / "state")
    video = tmp_path / "A.mp4"
    video.write_bytes(b"video" * 100)

    class Api:
        def get_config_value(self, k):
            return "me"

        def kernels_status(self, kernel):
            return "KernelWorkerStatus.COMPLETE"

        def kernels_output(self, kernel, path, **kw):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(Path(path) / "result.zip", "w") as zf:
                zf.writestr("result.json", json.dumps({"job": "abc123", "detections": [], "stats": {}}))

        def dataset_delete(self, *a, **kw):
            pass

    monkeypatch.setattr(kaggle_backend, "_api", Api)
    pushed = []
    monkeypatch.setattr(KaggleTrackBackend, "_push", lambda self, *a, **kw: pushed.append(1))
    state = kaggle_backend._state_file(str(video), "A")
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"job_id": "abc123", "dataset": "courtlens-job-abc123",
                                 "pushed_at": kaggle_backend.time.time()}))
    b = KaggleTrackBackend()
    assert b._remote(str(video), "A") == [] and not pushed
    assert b.last_run_stats["kaggle_job"] == "abc123" and not state.exists()


def test_result_from_another_job_is_rejected():
    import pytest

    from app.services.kaggle_backend import KaggleTrackBackend

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps({"job": "other", "detections": []}))
    with pytest.raises(RuntimeError, match="expected mine"):
        KaggleTrackBackend()._parse(buf.getvalue(), "A", "mine", 1.0)
