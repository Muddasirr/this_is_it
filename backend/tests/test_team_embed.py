"""SigLIP-style team split (roboflow/sports recipe) on synthetic embeddings."""

import numpy as np

from app.services.team_embed import assign_kits_by_embedding, split_teams


def _cloud(centre, n, rng, spread=0.05):
    pts = np.asarray(centre, dtype=float) + spread * rng.standard_normal((n, len(centre)))
    return (pts / np.linalg.norm(pts, axis=1, keepdims=True)).tolist()


def test_two_teams_plus_small_ref_cluster():
    rng = np.random.default_rng(0)
    dims = 32
    a, b, ref = np.eye(dims)[0], np.eye(dims)[1], np.eye(dims)[2]
    embs = _cloud(a, 7, rng) + _cloud(b, 6, rng) + _cloud(ref, 2, rng)
    kits = split_teams(embs)
    assert set(kits[:7]) == {"A"} and set(kits[7:13]) == {"B"} and set(kits[13:]) == {"X"}


def test_two_teams_without_third_colour_uses_k2():
    rng = np.random.default_rng(1)
    dims = 32
    embs = _cloud(np.eye(dims)[0], 6, rng) + _cloud(np.eye(dims)[1], 6, rng)
    kits = split_teams(embs)
    assert len(set(kits[:6])) == 1 and len(set(kits[6:])) == 1 and kits[0] != kits[6]
    assert "X" not in kits


class _T:
    def __init__(self, emb, dur=10_000):
        self.features = {"team_emb": emb} if emb is not None else {}
        self.start_ms, self.end_ms = 0, dur


def test_assign_keeps_colour_kit_for_tracks_without_embedding_relabelled():
    rng = np.random.default_rng(2)
    dims = 16
    ea, eb = _cloud(np.eye(dims)[0], 4, rng), _cloud(np.eye(dims)[1], 4, rng)
    # Colour split agreed on the teams but called them the other way round
    metas = [{"tracks": [_T(e)], "kit": "B"} for e in ea] + [{"tracks": [_T(e)], "kit": "A"} for e in eb]
    metas.append({"tracks": [_T(None)], "kit": "A"})  # colour said "team of eb"
    assert assign_kits_by_embedding(metas)
    assert metas[-1]["kit"] == metas[4]["kit"]  # follows eb's team after relabel
    assert {m["kit"] for m in metas[:4]} != {m["kit"] for m in metas[4:8]}


def test_assign_skips_when_embeddings_missing():
    metas = [{"tracks": [_T(None)], "kit": "A"} for _ in range(4)]
    assert not assign_kits_by_embedding(metas)


def test_embed_crops_hands_torch_contiguous_rgb(monkeypatch):
    """Kaggle crash: BGR→RGB view with a negative stride reached torch."""
    import numpy as np
    import torch

    from app.services import team_embed

    seen = []

    class _Proc:
        def __call__(self, images, return_tensors):
            seen.extend(images)
            for im in images:
                torch.from_numpy(im)  # raises on negative strides
            return type("B", (), {"to": lambda self, d: {"pixel_values": torch.zeros(len(images), 3, 4, 4)}})()

    class _Model:
        def __call__(self, **kw):
            n = kw["pixel_values"].shape[0]
            return type("O", (), {"pooler_output": torch.ones(n, 8)})()

    monkeypatch.setattr(team_embed, "_load", lambda device: (_Model(), _Proc(), device))
    crop = np.zeros((40, 20, 3), dtype=np.uint8)
    crop[..., 0] = 255  # blue in BGR
    out = team_embed.embed_crops([crop, crop], device="cpu")
    assert out.shape == (2, 8)
    assert all(im.flags["C_CONTIGUOUS"] and im[0, 0, 2] == 255 for im in seen)  # RGB: blue last
