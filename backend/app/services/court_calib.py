"""Image → court metres from clicked court landmarks.

A fixed facility camera only needs calibrating once: staff click known court
landmarks on a frame. Player position = foot point (bottom-centre of the box),
as in football_analysis (gtm_pass ViewTransformer).

Two models, chosen by how many landmarks were clicked:
  * 4 corners → plain perspective transform (gtm_pass). Exact only for a flat,
    undistorted view; on a wide fixed camera (SoccerTrack v2 panorama) it is off
    by ~11 m on average.
  * ≥ 8 landmarks (corners + halfway-line ends + centre-circle crossings) →
    radial lens distortion (k1, k2, vertical stretch) + perspective, fitted by
    least squares. Same camera: 0.93 m mean error on held-out landmarks.

Distance is measured between samples ≥ MIN_STEP_S apart and steps faster than
MAX_SPEED_MS are dropped — raw per-frame box jitter otherwise adds metres a
player never ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

FUTSAL_LENGTH_M = 40.0
FUTSAL_WIDTH_M = 20.0
MIN_STEP_S = 0.5
MAX_SPEED_MS = 10.0  # ~36 km/h — faster than any futsal sprint ⇒ tracking glitch
TOP_SPEED_WINDOW_S = 1.0
MIN_POINTS_DISTORTION = 8

Mapper = Callable[[np.ndarray], np.ndarray]


def corner_court_points(length: float, width: float) -> list[list[float]]:
    """Court metres of the 4 corners in click order: near-left, far-left, far-right, near-right."""
    return [[0.0, width], [0.0, 0.0], [length, 0.0], [length, width]]


def validate_calibration(cal: dict[str, Any]) -> dict[str, Any]:
    """Normalise a calibration payload and fit its model; raises ValueError when unusable.

    `image_points` are clicked pixels; `court_points` their court metres (origin at
    the far-left corner, x along the length, y towards the camera). Without
    `court_points`, exactly 4 points are read as the corners.
    """
    img = cal.get("image_points") or []
    length = float(cal.get("court_length_m") or FUTSAL_LENGTH_M)
    width = float(cal.get("court_width_m") or FUTSAL_WIDTH_M)
    if not (10 <= length <= 120 and 5 <= width <= 90):
        raise ValueError("court size out of range")
    court = cal.get("court_points") or (corner_court_points(length, width) if len(img) == 4 else None)
    if court is None or len(court) != len(img) or len(img) < 4:
        raise ValueError("need ≥ 4 image_points with matching court_points (or exactly 4 corners)")
    if any(len(p) != 2 for p in img) or any(len(p) != 2 for p in court):
        raise ValueError("points must be [x, y]")
    arr = np.asarray(img, dtype=np.float64)
    # Area of the clicked hull (shoelace on the first 4) — collinear clicks cannot define a court
    x, y = arr[:4, 0], arr[:4, 1]
    if 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 100:
        raise ValueError("corners are (nearly) collinear")
    out = {
        "image_points": arr.tolist(),
        "court_points": np.asarray(court, dtype=np.float64).tolist(),
        "court_length_m": length,
        "court_width_m": width,
        "frame_w": cal.get("frame_w"),
        "frame_h": cal.get("frame_h"),
    }
    out.update(_fit(out))
    return out


def _norm_frame(cal: dict[str, Any]) -> tuple[np.ndarray, float]:
    pts = np.asarray(cal["image_points"], dtype=np.float64)
    w = float(cal.get("frame_w") or (pts[:, 0].max() * 1.05))
    h = float(cal.get("frame_h") or (pts[:, 1].max() * 1.05))
    return np.array([w / 2.0, h / 2.0]), w / 2.0


def _undistort(xy: np.ndarray, centre: np.ndarray, scale: float, k1: float, k2: float, ky: float) -> np.ndarray:
    n = (xy - centre) / scale
    n = n * np.array([1.0, 1.0 + ky])  # stitched panoramas stretch vertically
    r2 = (n**2).sum(axis=1, keepdims=True)
    return n * (1.0 + k1 * r2 + k2 * r2 * r2)


def _apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
    return h[:, :2] / h[:, 2:3]


def _fit(cal: dict[str, Any]) -> dict[str, Any]:
    import cv2

    img = np.asarray(cal["image_points"], dtype=np.float64)
    court = np.asarray(cal["court_points"], dtype=np.float64)
    if len(img) < MIN_POINTS_DISTORTION:
        H, _ = cv2.findHomography(np.float32(img), np.float32(court), 0)
        model = {"model": "homography", "params": (H / H[2, 2]).ravel().tolist()}
        err = np.linalg.norm(_apply_h(H, img) - court, axis=1)
    else:
        from scipy.optimize import least_squares

        centre, scale = _norm_frame(cal)

        def residual(p: np.ndarray) -> np.ndarray:
            H = np.append(p[3:], 1.0).reshape(3, 3)
            return (_apply_h(H, _undistort(img, centre, scale, *p[:3])) - court).ravel()

        H0, _ = cv2.findHomography(np.float32(_undistort(img, centre, scale, 0, 0, 0)), np.float32(court), 0)
        p0 = np.concatenate([[0.0, 0.0, 0.0], (H0 / H0[2, 2]).ravel()[:8]])
        p = least_squares(residual, p0, method="lm" if 2 * len(img) >= len(p0) else "trf").x
        model = {"model": "radial_homography", "params": p.tolist()}
        err = np.abs(residual(p)).reshape(-1, 2)
        err = np.linalg.norm(err, axis=1)
    model["fit_error_m"] = round(float(err.mean()), 3)
    return model


def mapper(cal: dict[str, Any]) -> Mapper:
    """Pixel → court-metres function for a validated calibration (fits legacy payloads on the fly)."""
    if "model" not in cal:
        cal = validate_calibration(cal)
    p = np.asarray(cal["params"], dtype=np.float64)
    if cal["model"] == "homography":
        H = p.reshape(3, 3)
        return lambda xy: _apply_h(H, np.asarray(xy, dtype=np.float64).reshape(-1, 2))
    centre, scale = _norm_frame(cal)
    H = np.append(p[3:], 1.0).reshape(3, 3)
    return lambda xy: _apply_h(H, _undistort(np.asarray(xy, dtype=np.float64).reshape(-1, 2), centre, scale, *p[:3]))


def homography(cal: dict[str, Any]) -> np.ndarray:
    """3×3 pixel → metre transform from the 4 corners (near-left, far-left, far-right, near-right)."""
    import cv2

    src = np.asarray(cal["image_points"][:4], dtype=np.float32)
    dst = np.asarray(corner_court_points(float(cal["court_length_m"]), float(cal["court_width_m"])), dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_court(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    return _apply_h(H, np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2))


def foot_points(series: list[dict[str, Any]], cal: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(t_seconds, foot xy in the calibration's pixel space) for a bbox series."""
    pts = sorted(series, key=lambda p: p.get("ts_ms", 0))
    t = np.array([float(p.get("ts_ms", 0)) / 1000.0 for p in pts])
    xy = np.array([[(p.get("x1", 0) + p.get("x2", 0)) / 2.0, p.get("y2", 0)] for p in pts], dtype=np.float64)
    # Calibration clicked on a frame of another size (e.g. proxy) → rescale
    fw, fh = cal.get("frame_w"), cal.get("frame_h")
    if pts and fw and fh and pts[0].get("frame_w") and pts[0].get("frame_h"):
        xy[:, 0] *= float(fw) / float(pts[0]["frame_w"])
        xy[:, 1] *= float(fh) / float(pts[0]["frame_h"])
    return t, xy


ON_COURT_MARGIN_M = 1.5
ON_COURT_MIN_FRAC = 0.6


def on_court_fraction(
    series: list[dict[str, Any]], cal: dict[str, Any], to_metres: Mapper | None = None,
    margin_m: float = ON_COURT_MARGIN_M,
) -> float:
    """Share of a track's foot points inside the court (+ margin) — bench / crowd / staff are ~0."""
    if not series:
        return 0.0
    to_metres = to_metres or mapper(cal)
    _, xy = foot_points(series, cal)
    court = to_metres(xy)
    L, W = float(cal["court_length_m"]), float(cal["court_width_m"])
    inside = (
        (court[:, 0] >= -margin_m) & (court[:, 0] <= L + margin_m)
        & (court[:, 1] >= -margin_m) & (court[:, 1] <= W + margin_m)
    )
    return float(inside.mean())


def track_motion(series: list[dict[str, Any]], cal: dict[str, Any], to_metres: Mapper | None = None) -> dict[str, float]:
    """Metres run and top speed (km/h) for one track on a calibrated camera."""
    if len(series) < 2:
        return {"distance_m": 0.0, "top_speed_kmh": 0.0}
    to_metres = to_metres or mapper(cal)
    t, xy = foot_points(series, cal)
    court = to_metres(xy)
    L, W = float(cal["court_length_m"]), float(cal["court_width_m"])
    inside = (court[:, 0] > -2) & (court[:, 0] < L + 2) & (court[:, 1] > -2) & (court[:, 1] < W + 2)
    t, court = t[inside], court[inside]
    dist, last = 0.0, 0
    for i in range(1, len(t)):
        dt = t[i] - t[last]
        if dt < MIN_STEP_S:
            continue
        step = float(np.linalg.norm(court[i] - court[last]))
        if step / dt <= MAX_SPEED_MS:
            dist += step
        last = i
    top, j = 0.0, 0
    for i in range(len(t)):
        while j < i and t[i] - t[j] > TOP_SPEED_WINDOW_S:
            j += 1
        dt = t[i] - t[j]
        if dt >= 0.5 * TOP_SPEED_WINDOW_S:
            v = float(np.linalg.norm(court[i] - court[j])) / dt
            if v <= MAX_SPEED_MS:
                top = max(top, v)
    return {"distance_m": round(dist, 1), "top_speed_kmh": round(top * 3.6, 1)}
