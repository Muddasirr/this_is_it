def _video(path, w, h):
    import cv2
    import numpy as np

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    out.write(np.zeros((h, w, 3), np.uint8))
    out.release()
    return str(path)


def test_football_broadcast_uses_the_referee_aware_detector(tmp_path, monkeypatch):
    """11v11 broadcast frames: people from the football fine-tune at 640 (it has a referee class),
    the ball pass keeps the general model. Futsal and wide panoramas keep the general model."""
    from app.core.config import get_settings
    from app.services.yolo_backend import YoloTrackBackend

    weights = tmp_path / "yolov5x_football.pt"
    weights.write_bytes(b"x" * 2000)
    monkeypatch.setenv("YOLO_MODEL_FOOTBALL", str(weights))
    broadcast = _video(tmp_path / "tv.mp4", 1920, 1080)
    panorama = _video(tmp_path / "pano.mp4", 4096, 1080)

    monkeypatch.setenv("COURT_FORMAT", "11v11")
    get_settings.cache_clear()
    b = YoloTrackBackend()
    general = b.model_name
    assert b._football_profile(broadcast) == 640
    assert b.model_name == str(weights.resolve()) and b.ball_model_path == general
    assert YoloTrackBackend()._football_profile(panorama) == 0

    monkeypatch.setenv("COURT_FORMAT", "5v5")
    get_settings.cache_clear()
    futsal = YoloTrackBackend()
    assert futsal._football_profile(broadcast) == 0 and futsal.model_name == general
    get_settings.cache_clear()


def test_track_is_a_referee_when_enough_of_its_frames_say_so():
    """Decided over the whole track, not the first frame: the detector misses refs often but
    almost never calls a player one."""
    from app.services.yolo_backend import track_class

    def obs(*names):
        return [{"class_name": n} for n in names]

    assert track_class(obs("person", "referee", "person", "referee", "person")) == "referee"
    assert track_class(obs("referee", *["person"] * 19)) == "person"  # one stray call
    assert track_class(obs("person", *["person"] * 8, "referee")) == "person"
    assert track_class(obs("sports_ball", "sports_ball")) == "sports_ball"
