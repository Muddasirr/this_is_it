"""Per-frame perception: ball candidates + player pose. Same array layout for live video and
the Kaggle held-out caches, so everything downstream is shared between product and eval."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
K_BALL = 5
K_PLAYER = 10
COCO_SPORTS_BALL = 32


@dataclass
class Detections:
    fps: float
    size: tuple[int, int]  # (w, h)
    ball: np.ndarray  # (n, K_BALL, 5) xyxy conf, conf 0 = empty
    pose_box: np.ndarray  # (n, K_PLAYER, 5)
    pose_kpt: np.ndarray  # (n, K_PLAYER, 17, 3)
    pose_done: np.ndarray  # (n,) bool: pose model ran on this frame
    cuts: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # (n,) scene cut before frame
    step: int = 1  # analysed frame i = video frame i * step (high-fps video is sampled down to ~30 fps)

    @property
    def n(self) -> int:
        return len(self.ball)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, fps=self.fps, size=np.array(self.size), ball=self.ball, pose_box=self.pose_box,
                            pose_kpt=self.pose_kpt.astype(np.float16), pose_done=self.pose_done, cuts=self.cuts,
                            step=self.step)

    @classmethod
    def load(cls, path: Path, fps: float = 30.0, size=(1920, 1080)) -> "Detections":
        z = np.load(path, allow_pickle=True)
        n = len(z["ball"])
        done = z["pose_done"] if "pose_done" in z.files else np.ones(n, bool)
        cuts = z["cuts"] if "cuts" in z.files else np.zeros(n, bool)
        return cls(float(z["fps"]) if "fps" in z.files else fps,
                   tuple(z["size"].tolist()) if "size" in z.files else size,
                   z["ball"].astype(np.float32), z["pose_box"].astype(np.float32),
                   z["pose_kpt"].astype(np.float32), done, cuts, int(z["step"]) if "step" in z.files else 1)


def default_weights() -> dict[str, str]:
    ball = MODELS_DIR / "ball.pt"
    pose = MODELS_DIR / "pose.pt"
    return {
        "ball": str(ball) if ball.exists() else "",
        "pose": str(pose) if pose.exists() else str(MODELS_DIR / "yolo11m-pose.pt"),
    }


class Detector:
    """Runs the ball + pose models over a video.

    ball_imgsz 1536 keeps an ~8 px ball detectable at 1080p; pose every `pose_stride` frames
    (players move slowly relative to 30 fps; the tracker interpolates in between).
    """

    def __init__(self, ball_w: str | None = None, pose_w: str | None = None, device: str | None = None,
                 ball_imgsz: int = 1536, pose_imgsz: int = 1280, pose_stride: int = 1):
        from ultralytics import YOLO

        w = default_weights()
        self.ball_w = ball_w if ball_w is not None else w["ball"]
        self.pose_w = pose_w or w["pose"]
        self.device = device or os.environ.get("PADEL_DEVICE") or ("cuda" if _cuda() else "cpu")
        self.ball_model = YOLO(self.ball_w) if self.ball_w else YOLO(str(MODELS_DIR / "yolo11m.pt"))
        self.ball_classes = None if self.ball_w else [COCO_SPORTS_BALL]
        self.pose_model = YOLO(self.pose_w)
        self.ball_imgsz, self.pose_imgsz, self.pose_stride = ball_imgsz, pose_imgsz, max(1, pose_stride)

    def run(self, video: str | Path, max_frames: int = 0, batch: int = 8,
            progress: Callable[[float], None] | None = None, target_fps: float = 30.0) -> Detections:
        """The hit / stroke models were trained at 30 fps (PT100): 50/60 fps video is sampled down."""
        cap = cv2.VideoCapture(str(video))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / target_fps)))
        fps = src_fps / step
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        n = (n_video + step - 1) // step
        if max_frames:
            n = min(n, max_frames)
        ball = np.zeros((n, K_BALL, 5), np.float32)
        pbox = np.zeros((n, K_PLAYER, 5), np.float32)
        pkpt = np.zeros((n, K_PLAYER, 17, 3), np.float32)
        done = np.zeros(n, bool)
        cuts = np.zeros(n, bool)
        prev_sig = None
        f0 = 0
        extra = {"half": True} if self.device != "cpu" else {}
        while f0 < n:
            frames = []
            while len(frames) < batch and f0 + len(frames) < n:
                ok, fr = cap.read()
                for _ in range(step - 1):  # skip to the next sampled frame
                    cap.grab()
                if not ok:
                    break
                frames.append(fr)
            if not frames:
                break
            for j, fr in enumerate(frames):
                sig = cv2.calcHist([cv2.resize(fr, (64, 36))], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
                sig = cv2.normalize(sig, None).flatten()
                if prev_sig is not None and cv2.compareHist(prev_sig, sig, cv2.HISTCMP_BHATTACHARYYA) > 0.5:
                    cuts[f0 + j] = True
                prev_sig = sig
            res = self.ball_model.predict(frames, imgsz=self.ball_imgsz, conf=0.03, max_det=K_BALL,
                                          classes=self.ball_classes, verbose=False, device=self.device, **extra)
            for j, r in enumerate(res):
                k = len(r.boxes)
                if k:
                    ball[f0 + j, :k, :4] = r.boxes.xyxy.cpu().numpy()
                    ball[f0 + j, :k, 4] = r.boxes.conf.cpu().numpy()
            idx = [j for j in range(len(frames)) if (f0 + j) % self.pose_stride == 0]
            if idx:
                res = self.pose_model.predict([frames[j] for j in idx], imgsz=self.pose_imgsz, conf=0.2,
                                              max_det=K_PLAYER, classes=[0], verbose=False, device=self.device,
                                              **extra)
                for j, r in zip(idx, res):
                    done[f0 + j] = True
                    k = len(r.boxes)
                    if k:
                        pbox[f0 + j, :k, :4] = r.boxes.xyxy.cpu().numpy()
                        pbox[f0 + j, :k, 4] = r.boxes.conf.cpu().numpy()
                        pkpt[f0 + j, :k] = r.keypoints.data.cpu().numpy()
            f0 += len(frames)
            if progress:
                progress(f0 / n)
        cap.release()
        n = f0
        return Detections(fps, size, ball[:n], pbox[:n], pkpt[:n], done[:n], cuts[:n], step)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False
