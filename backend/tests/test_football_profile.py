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


def test_football_panorama_gets_the_panorama_ball_model(tmp_path, monkeypatch):
    """11v11 fixed panorama: general model for people, the SoccerTrack-tuned ball model for the
    ball pass. Futsal, broadcast frames and an explicit VISION_BALL_MODEL are left alone."""
    from app.core.config import get_settings
    from app.services.yolo_backend import YoloTrackBackend

    ball = tmp_path / "ball_st2.pt"
    ball.write_bytes(b"x" * 2000)
    monkeypatch.setenv("VISION_BALL_MODEL_PANORAMA", str(ball))
    monkeypatch.setenv("YOLO_MODEL_FOOTBALL", str(tmp_path / "missing.pt"))
    panorama = _video(tmp_path / "pano.mp4", 4096, 1080)
    broadcast = _video(tmp_path / "tv.mp4", 1920, 1080)

    monkeypatch.setenv("COURT_FORMAT", "11v11")
    get_settings.cache_clear()
    b = YoloTrackBackend()
    general = b.model_name
    assert b._football_profile(panorama) == 0
    assert b.ball_model_path == str(ball.resolve()) and b.model_name == general
    tv = YoloTrackBackend()
    tv._football_profile(broadcast)
    assert tv.ball_model_path != str(ball.resolve())

    monkeypatch.setenv("VISION_BALL_MODEL", "custom.pt")
    get_settings.cache_clear()
    assert YoloTrackBackend()._football_profile(panorama) == 0
    monkeypatch.delenv("VISION_BALL_MODEL")

    monkeypatch.setenv("COURT_FORMAT", "5v5")
    get_settings.cache_clear()
    futsal = YoloTrackBackend()
    futsal._football_profile(panorama)
    assert futsal.ball_model_path == ""
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


def test_spotter_sees_the_pitch_band_of_a_football_panorama(tmp_path, monkeypatch):
    """The SoccerTrack spotter was trained on a 16:9 window over the pitch rows, not the full
    panorama height (foreground lettering, stands): crop to where the players stand."""
    from types import SimpleNamespace

    from app.core.config import get_settings
    from app.services.action_spotting import pitch_band, write_model_video

    players = [SimpleNamespace(class_name="person", bbox_series=[
        {"ts_ms": i * 40, "x1": 100.0 + i, "x2": 130.0 + i, "y1": 200.0 + (i % 5) * 100, "y2": 260.0 + (i % 5) * 100}
        for i in range(100)])]
    assert pitch_band(players, 1080) == (140, 700)
    assert pitch_band([], 1080) == (0, 1080)  # no players → whole height

    import shutil

    if not shutil.which("ffmpeg"):
        return  # the band choice is covered above; the encode needs ffmpeg (worker image)
    video = _video(tmp_path / "pano.mp4", 4096, 1080)
    monkeypatch.setenv("COURT_FORMAT", "11v11")
    get_settings.cache_clear()
    assert write_model_video(video, players, str(tmp_path / "m.mp4"))["band"] == [140, 700]
    monkeypatch.setenv("COURT_FORMAT", "5v5")
    get_settings.cache_clear()
    assert write_model_video(video, players, str(tmp_path / "m.mp4"))["band"] == [0, 1080]
    get_settings.cache_clear()
