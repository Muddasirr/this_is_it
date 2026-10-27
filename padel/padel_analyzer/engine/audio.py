"""Racket-impact sound detection (audio half of hit detection).

Model after Decorte et al., "Multi-Modal Hit Detection and Positional Analysis in Padel Competitions"
(CVPRW 2024): 40 log-Mel bands, FFT 2048, 50 % overlap, sequences of 256 frames, a small SED-net style
CRNN (3 conv blocks pooling frequency -> 2 bidirectional GRUs -> per-frame sigmoid), focal loss.
They report event F1 0.92 at a 250 ms collar on broadcast padel. Audio is camera-independent, so it is the
hit cue that survives club cameras where the ball is small or occluded; it is fused with the visual
features in events.py rather than used alone (racket hits on the floor/frame, shouts and glass knocks fool it).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
N_FFT = 2048
HOP = 1024  # 50 % overlap -> 23.2 ms frames
N_MELS = 40
SEQ = 256
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(video: str | Path, sr: int = SR) -> np.ndarray | None:
    """Mono float32 waveform, or None when the file has no audio track."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, np.float32).copy()


def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            fb[m - 1, k] = (k - a) / max(b - a, 1)
        for k in range(b, c):
            fb[m - 1, k] = (c - k) / max(c - b, 1)
    return fb


def logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(frames, N_MELS) log-Mel energies; frame i is centred at i * HOP / sr seconds."""
    pad = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(pad) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None] + HOP * np.arange(n)[:, None]
    win = np.hanning(N_FFT).astype(np.float32)
    spec = np.abs(np.fft.rfft(pad[idx] * win, axis=1)) ** 2
    mel = spec @ _mel_filters(sr, N_FFT, N_MELS).T
    return np.log(mel + 1e-8).astype(np.float32)


def frame_times(n_frames: int, sr: int = SR) -> np.ndarray:
    return np.arange(n_frames) * HOP / sr


def build_model():
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, pool):
                return nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.MaxPool2d((1, pool)), nn.Dropout(0.2))

            self.conv = nn.Sequential(block(1, 5), block(64, 2), block(64, 2))  # freq 40 -> 2
            self.gru1 = nn.GRU(64 * 2, 32, batch_first=True, bidirectional=True)
            self.gru2 = nn.GRU(64, 16, batch_first=True, bidirectional=True)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):  # x: (B, T, F)
            z = self.conv(x[:, None])  # (B, 64, T, 2)
            z = z.permute(0, 2, 1, 3).flatten(2)  # (B, T, 128)
            z, _ = self.gru1(z)
            z, _ = self.gru2(z)
            return self.fc2(self.fc1(z)).squeeze(-1)  # logits (B, T)

    return CRNN()


def focal_loss(logits, target, alpha: float = 0.75, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    a = target * alpha + (1 - target) * (1 - alpha)
    return (a * (1 - pt) ** gamma * bce).mean()


class AudioHitModel:
    """Per-frame racket-hit probability from a match's audio track."""

    def __init__(self, path: Path | str | None = None, device: str | None = None):
        import torch

        p = Path(path) if path else MODEL_DIR / "audio_hit.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.mean, self.std = np.float32(ck["mean"]), np.float32(ck["std"])
        self.threshold = float(ck.get("threshold", 0.5))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = build_model()
        m.load_state_dict(ck["state_dict"])
        self.model = m.eval().to(self.device)

    @classmethod
    def available(cls) -> bool:
        return (MODEL_DIR / "audio_hit.pt").exists()

    def probs(self, y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
        """(times_s, prob) over the whole waveform, sliding SEQ-frame windows with 50 % overlap."""
        import torch

        feat = (logmel(y, sr) - self.mean) / self.std
        n = len(feat)
        acc = np.zeros(n, np.float32)
        cnt = np.zeros(n, np.float32)
        step = SEQ // 2
        starts = list(range(0, max(1, n - SEQ + 1), step)) or [0]
        if starts[-1] + SEQ < n:
            starts.append(n - SEQ)
        with torch.no_grad():
            for s in starts:
                x = feat[s:s + SEQ]
                if len(x) < SEQ:
                    x = np.pad(x, ((0, SEQ - len(x)), (0, 0)))
                p = torch.sigmoid(self.model(torch.from_numpy(x)[None].to(self.device))).cpu().numpy()[0]
                k = min(SEQ, n - s)
                acc[s:s + k] += p[:k]
                cnt[s:s + k] += 1
        return frame_times(n, sr), acc / np.maximum(cnt, 1)


def peaks(times: np.ndarray, prob: np.ndarray, thr: float, min_gap_s: float = 0.15) -> np.ndarray:
    """Hit times (s): local maxima above thr, at least min_gap_s apart."""
    cand = np.where((prob >= thr) & (prob >= np.roll(prob, 1)) & (prob >= np.roll(prob, -1)))[0]
    out: list[int] = []
    for i in cand[np.argsort(-prob[cand])]:
        if all(abs(times[i] - times[j]) >= min_gap_s for j in out):
            out.append(i)
    return np.sort(times[out]) if out else np.zeros(0)


def video_features(times: np.ndarray, prob: np.ndarray, n_video: int, fps: float, step: int = 1,
                   thr: float = 0.5) -> np.ndarray:
    """(n_video, 3) per analysed video frame: max audio prob within ±1 frame, within ±3 frames,
    and signed seconds to the nearest audio peak (clipped to ±1 s). NaN columns when there is no audio."""
    t_frame = np.arange(n_video) * step / fps
    dt = step / fps
    out = np.full((n_video, 3), np.nan, np.float32)
    if len(prob) == 0:
        return out
    # max-pool the audio probability onto video frame times
    for c, half in ((0, 1.0), (1, 3.0)):
        lo = np.searchsorted(times, t_frame - half * dt)
        hi = np.searchsorted(times, t_frame + half * dt, side="right")
        out[:, c] = [prob[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi)]
    pk = peaks(times, prob, thr)
    if len(pk):
        j = np.clip(np.searchsorted(pk, t_frame), 1, len(pk)) - 1
        d1 = pk[j] - t_frame
        d2 = np.where(j + 1 < len(pk), pk[np.minimum(j + 1, len(pk) - 1)] - t_frame, np.inf)
        d = np.where(np.abs(d1) <= np.abs(d2), d1, d2)
        out[:, 2] = np.clip(d, -1, 1)
    else:
        out[:, 2] = 1.0
    return out
