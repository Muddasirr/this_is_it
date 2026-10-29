"""CPU kernel: pair-identity linker variants on saved features (padel-identity) — no video, no GPU.

Per segment: far-pair and near-pair mean features (shirt histogram or DINOv2). Truth: which scoreboard pair
plays the far end (padel-identity truth_*.json). Score = segments correct up to one global relabelling.
  V0  production: prototypes from segment 0, keep/swap per segment
  V1  running prototypes per (pair, end), same-end comparisons first
  V2  side-centred features (far/near look different: subtract each end's mean) + spectral sign of the
      segment-agreement matrix + switch penalty (ends change rarely)
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

code = Path(sorted(glob.glob("/kaggle/input/**/padel_analyzer/__init__.py", recursive=True))[0]).parents[1]
repo = Path("/tmp/padel")
shutil.copytree(code, repo)
labels = Path(sorted(glob.glob("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json", recursive=True))[0]).parent
(repo / "data" / "pt100").mkdir(parents=True, exist_ok=True)
shutil.copytree(labels, repo / "data" / "pt100" / "labels")
sys.path.insert(0, str(repo))
from padel_analyzer.engine import court as C  # noqa: E402
from padel_analyzer.engine import players as P  # noqa: E402
from padel_analyzer.engine.detect import Detections  # noqa: E402
from padel_analyzer.eval import pt100  # noqa: E402


def first(p):
    return Path(sorted(glob.glob(p, recursive=True))[0])


def cos_d(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def segment_feats(segments, det_of, feats):
    out = []
    for a, b in segments:
        sl = []
        for s in range(4):
            v = [feats[(f, det_of[f, s])] for f in range(a, b) if det_of[f, s] >= 0 and (f, det_of[f, s]) in feats]
            sl.append((np.mean(v, 0), len(v)) if v else (None, 0))
        far = [v for v, c in sl[:2] if v is not None]
        near = [v for v, c in sl[2:] if v is not None]
        out.append({"F": np.mean(far, 0) if far else None, "N": np.mean(near, 0) if near else None,
                    "w": min(sum(c for _, c in sl[:2]), sum(c for _, c in sl[2:]))})
    return out


def v0(seg, dist):
    o = [0] * len(seg)
    if seg[0]["F"] is None:
        return o
    pF, pN = seg[0]["F"], seg[0]["N"]
    for k in range(1, len(seg)):
        if seg[k]["F"] is None or seg[k]["N"] is None:
            o[k] = o[k - 1]
            continue
        keep = dist(seg[k]["F"], pF) + dist(seg[k]["N"], pN)
        swap = dist(seg[k]["F"], pN) + dist(seg[k]["N"], pF)
        o[k] = 0 if keep <= swap else 1
    return o


def v1(seg, dist):
    """running prototypes per (pair, end); compare with the same end when that prototype exists"""
    o = [0] * len(seg)
    proto = {}
    k0 = next((k for k, s in enumerate(seg) if s["F"] is not None and s["N"] is not None), None)
    if k0 is None:
        return o
    proto[(0, "F")], proto[(1, "N")] = [seg[k0]["F"]], [seg[k0]["N"]]
    for k in range(len(seg)):
        if k <= k0 or seg[k]["F"] is None or seg[k]["N"] is None:
            o[k] = o[k - 1] if k else 0
            continue

        def ref(team, end):
            got = proto.get((team, end)) or proto.get((team, "N" if end == "F" else "F"))
            return np.mean(got[-30:], 0)
        keep = dist(seg[k]["F"], ref(0, "F")) + dist(seg[k]["N"], ref(1, "N"))
        swap = dist(seg[k]["F"], ref(1, "F")) + dist(seg[k]["N"], ref(0, "N"))
        o[k] = 0 if keep <= swap else 1
        far_t = o[k]
        proto.setdefault((far_t, "F"), []).append(seg[k]["F"])
        proto.setdefault((1 - far_t, "N"), []).append(seg[k]["N"])
    return o


def v2(seg, lam=0.3):
    ok = [s["F"] is not None and s["N"] is not None and s["w"] >= 2 for s in seg]
    idx = [k for k in range(len(seg)) if ok[k]]
    if len(idx) < 2:
        return [0] * len(seg)
    Fm = np.mean([seg[k]["F"] for k in idx], 0)
    Nm = np.mean([seg[k]["N"] for k in idx], 0)
    Fc = {k: seg[k]["F"] - Fm for k in idx}
    Nc = {k: seg[k]["N"] - Nm for k in idx}
    cs = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))  # noqa: E731
    m = len(idx)
    S = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                a, b = idx[i], idx[j]
                S[i, j] = cs(Fc[a], Fc[b]) + cs(Nc[a], Nc[b]) - cs(Fc[a], Nc[b]) - cs(Nc[a], Fc[b])
    w, V = np.linalg.eigh(S)
    u = V[:, -1]
    # Viterbi on the chain: unary = how strongly the spectral vector says 0/1, penalty per change of ends
    unary = np.c_[-u, u] / (np.abs(u).max() + 1e-9)  # cost of state 0 / 1
    cost = unary[0].copy()
    back = []
    for i in range(1, m):
        stay, switch = cost, cost[::-1] + lam
        choice = (switch < stay).astype(int)
        cost = np.minimum(stay, switch) + unary[i]
        back.append(choice)
    st = int(np.argmin(cost))
    path = [st]
    for ch in reversed(back):
        st = st if not ch[st] else 1 - st
        path.append(st)
    path = path[::-1]
    o, cur = [0] * len(seg), 0
    lab = dict(zip(idx, path))
    for k in range(len(seg)):
        cur = lab.get(k, cur)
        o[k] = cur
    return o


def score(o, truth):
    pairs = [(o[int(i)], r) for i, r in truth.items() if int(i) < len(o)]
    agree = sum(a == b for a, b in pairs)
    return f"{max(agree, len(pairs) - agree)}/{len(pairs)}"


res = {}
for key in ("F", "M"):
    name = pt100.MATCHES[key]
    g = pt100.load(key)
    z = np.load(first(f"/kaggle/input/**/cache/{name}_heldout.npz"), allow_pickle=True)
    n = len(z["ball"])
    det = Detections(30.0, (1920, 1080), z["ball"].astype(np.float32), z["pose_box"].astype(np.float32),
                     z["pose_kpt"].astype(np.float32), np.ones(n, bool), np.zeros(n, bool))
    cap = {}
    orig = P._link_identity
    P._link_identity = lambda segments, det_of, hists, n: (cap.update(segments=segments, det_of=det_of), orig(segments, det_of, hists, n))[1]
    fz = np.load(first(f"/kaggle/input/**/out/features_{key}.npz"))
    keys = [tuple(k) for k in fz["keys"].tolist()]
    hist = dict(zip(keys, fz["hist"]))
    dino = dict(zip(keys, fz["dino"].astype(np.float32)))
    tr = P.track(det, C.Calibration(g.H, (1920, 1080)), hist)
    P._link_identity = orig
    truth = json.loads(first(f"/kaggle/input/**/out/truth_{key}.json").read_text())["far_row"]
    out = {"production_track_hist": score([int(tr.team[a, 0]) for a, b in tr.segments], truth)}
    tr_d = P.track(det, C.Calibration(g.H, (1920, 1080)), dino)
    out["production_track_dino"] = score([int(tr_d.team[a, 0]) for a, b in tr_d.segments], truth)
    for fname, feats, dist in (("hist", hist, P.feature_dist), ("dino", dino, cos_d)):
        seg = segment_feats(cap["segments"], cap["det_of"], feats)
        out[f"V0_{fname}"] = score(v0(seg, dist), truth)
        out[f"V1_{fname}"] = score(v1(seg, dist), truth)
        for lam in (0.0, 0.3, 1.0):
            out[f"V2_{fname}_lam{lam}"] = score(v2(seg, lam), truth)
    res[key] = out
    print(key, json.dumps(out), flush=True)
Path("/kaggle/working/idlab.json").write_text(json.dumps(res, indent=1))
