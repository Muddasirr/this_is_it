"""Ball speed from one camera: average speed from the racket to the first floor bounce.

A full 3D flight cannot be recovered from one camera behind the court: shots travel mostly along the
viewing direction, so a free parabola fit drifts tens of metres in depth (EVAL §10). Two points of the
flight are pinned down, though:
  contact  the ball's camera ray where it passes above the hitter's feet (feet from the floor homography)
  bounce   the ball on the floor (z = 0), where the homography is exact; found as the first sharp turn of
           the ball's image path from moving down-screen to up-screen, on the opponent's half
speed = |bounce - contact| / flight time. It is the average over the flight (below the launch speed: drag,
arc); shots volleyed before they bounce, or hit into the glass first, get none. The camera (focal length +
pose) comes from the floor calibration and is used for the contact height.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import court as C

MAX_KMH = 140.0  # above this the "bounce" was a turn in mid-air (the floor point is then far off)
TURN_PX = 3.0  # px / frame change of vertical image velocity at a bounce (1080p, ~30 fps)


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    mirror: bool = False  # the court frame is mirrored w.r.t. the camera (x -> -x internally)
    floor_err_px: float = 0.0  # mean px error reproducing the floor calibration

    def _w(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, float)
        if self.mirror:
            X[..., 0] = -X[..., 0]
        return X

    @property
    def centre(self) -> np.ndarray:
        return self._w(-self.R.T @ self.t)

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) metres (x across, y along the court, z up) -> (N, 2) undistorted px."""
        x = (self.K @ (self.R @ self._w(X).T + self.t[:, None])).T
        return x[:, :2] / x[:, 2:3]

    def ray(self, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undistorted px -> (camera centre, unit direction) in court metres."""
        d = self._w(self.R.T @ np.linalg.solve(self.K, np.array([px[0], px[1], 1.0])))
        return self.centre, d / np.linalg.norm(d)


def camera_from_calibration(calib: C.Calibration) -> Camera | None:
    """Pinhole camera (square pixels, principal point at the image centre) that best reproduces the floor
    calibration: focal length searched; pose from the homography (both signs), refined by PnP on a grid of
    court points; both handednesses of the court frame are tried, so a calibration that labels the
    sidelines either way works. The camera must sit above the floor and
    raised points must move up in the image."""
    import cv2

    w, h = calib.image_size
    grid = np.array([[x, y] for x in np.linspace(0, C.WIDTH, 6) for y in np.linspace(0, C.LENGTH, 11)])
    img = calib.undistort(calib.to_px(grid)).astype(np.float64)
    best = None
    for mirror in (False, True):
        obj = np.c_[(-grid[:, 0] if mirror else grid[:, 0]), grid[:, 1], np.zeros(len(grid))].astype(np.float64)
        for f in np.geomspace(0.4 * w, 6 * w, 60):
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
            # pose from the homography itself: K [r1 r2 t] ~ (floor -> image), both signs; then refined
            Gm = np.linalg.inv(np.diag([-1.0, 1.0, 1.0]) @ calib.H if mirror else calib.H)
            A = np.linalg.solve(K, Gm)
            lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
            for sign in (1.0, -1.0):
                r1, r2, t0 = sign * lam * A[:, 0], sign * lam * A[:, 1], sign * lam * A[:, 2]
                U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
                R0 = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, cv2.Rodrigues(R0)[0], t0.reshape(3, 1), True,
                                              cv2.SOLVEPNP_ITERATIVE)
                cam = Camera(K, cv2.Rodrigues(rvec)[0], tvec.ravel(), mirror)
                if not ok or cam.centre[2] <= 0:
                    continue
                base, up = cam.project(np.array([[C.WIDTH / 2, C.NET_Y, 0.0], [C.WIDTH / 2, C.NET_Y, 1.0]]))
                if up[1] >= base[1]:
                    continue
                err = float(np.linalg.norm(cam.project(np.c_[grid, np.zeros(len(grid))]) - img, axis=1).mean())
                if best is None or err < best[0]:
                    best = (err, cam)
    if best is None:
        return None
    best[1].floor_err_px = best[0]
    return best[1]


def contact_point(cam: Camera, ball_px_u: np.ndarray, feet_m: np.ndarray) -> np.ndarray | None:
    """Point on the ball's camera ray closest to the vertical line above the hitter's feet."""
    o, d = cam.ray(ball_px_u)
    # minimise horizontal distance to (fx, fy): along the ray, x,y = o + s d
    dh = d[:2]
    if np.linalg.norm(dh) < 1e-9:
        return None
    s = float(np.dot(feet_m - o[:2], dh) / np.dot(dh, dh))
    p = o + s * d
    if not 0.1 <= p[2] <= 3.8:
        return None
    return p


def shot_speed(cam: Camera, calib: C.Calibration, ball_xy: np.ndarray, observed: np.ndarray, f: int, end: int,
               feet_m: np.ndarray, fps: float) -> dict | None:
    """Average ball speed from the hit at frame f to its first bounce (ball positions before `end`).
    Returns {"speed_kmh", "bounce_m", "flight_s", "contact_h_m"} or None."""
    n = len(ball_xy)
    k0 = next((k for k in (f, f + 1, f - 1, f + 2) if 0 <= k < n and observed[k]), None)
    if k0 is None or np.isnan(feet_m).any():
        return None
    S = contact_point(cam, calib.undistort(ball_xy[k0][None])[0], feet_m)
    if S is None:
        return None
    ks = [k for k in range(k0 + 2, min(end, n)) if observed[k]]
    if len(ks) < 3:
        return None
    p = ball_xy[ks]
    toward = 1.0 if feet_m[1] < C.NET_Y else -1.0
    turn = TURN_PX * calib.image_size[1] / 1080 * 30 / fps
    for a in range(1, len(ks) - 1):
        k_prev, k, k_next = ks[a - 1], ks[a], ks[a + 1]
        if k - k_prev > 2 or k_next - k > 2:
            continue
        v1 = (p[a] - p[a - 1]) / (k - k_prev)
        v2 = (p[a + 1] - p[a]) / (k_next - k)
        if not (v1[1] > 0 and v2[1] - v1[1] < -turn):  # moving down-screen, then sharply up: the floor
            continue
        P = calib.to_m(p[a][None])[0]
        if not ((P[1] - C.NET_Y) * toward > 0.3 and -0.3 <= P[0] <= C.WIDTH + 0.3 and 0 <= P[1] <= C.LENGTH):
            continue
        T = (k - k0) / fps
        v = float(np.linalg.norm(np.r_[P, 0.0] - S) / T * 3.6)
        if v > MAX_KMH:
            continue
        return {"speed_kmh": round(v, 1), "bounce_m": [round(float(P[0]), 2), round(float(P[1]), 2)],
                "flight_s": round(T, 2), "contact_h_m": round(float(S[2]), 2)}
    return None
