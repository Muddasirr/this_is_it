import numpy as np

from padel_analyzer.engine import audio as A


def _clicks(times_s, dur_s=4.0, sr=A.SR):
    """Noise floor + short broadband 'pock' transients at the given times."""
    rng = np.random.default_rng(0)
    y = 0.002 * rng.standard_normal(int(dur_s * sr)).astype(np.float32)
    for t in times_s:
        i = int(t * sr)
        n = int(0.006 * sr)
        y[i:i + n] += (0.8 * np.exp(-np.arange(n) / (0.001 * sr)) * rng.standard_normal(n)).astype(np.float32)
    return y


def test_logmel_shape_and_timing():
    y = _clicks([1.0])
    m = A.logmel(y)
    assert m.shape[1] == A.N_MELS
    assert abs(len(m) - len(y) / A.HOP) <= 2
    t = A.frame_times(len(m))
    assert abs(t[np.argmax(m.sum(1))] - 1.0) < 2 * A.HOP / A.SR  # energy peak lands on the click


def test_peaks_respect_threshold_and_gap():
    t = np.arange(0, 3, 0.01)
    p = np.zeros_like(t)
    for c in (0.5, 0.55, 1.5, 2.5):
        p += np.exp(-((t - c) / 0.02) ** 2)
    pk = A.peaks(t, p, thr=0.5, min_gap_s=0.15)
    assert len(pk) == 3 and abs(pk[1] - 1.5) < 0.02


def test_video_features_align_to_frames():
    times = np.arange(0, 4, A.HOP / A.SR)
    prob = np.exp(-((times - 2.0) / 0.03) ** 2)
    f = A.video_features(times, prob, n_video=120, fps=30.0, thr=0.5)
    assert f.shape == (120, 3)
    assert f[60, 0] == f[:, 0].max() and f[60, 0] > 0.9  # 2.0 s at 30 fps carries the peak (±1-frame max-pool)
    assert abs(f[60, 2]) < 0.05 and f[30, 2] > 0.9  # seconds to the nearest audio peak, clipped at 1 s
    none = A.video_features(times, np.zeros(0), n_video=10, fps=30.0)
    assert np.isnan(none).all()


def test_crnn_forward_shape():
    import torch

    m = A.build_model()
    out = m(torch.zeros(2, A.SEQ, A.N_MELS))
    assert out.shape == (2, A.SEQ)
