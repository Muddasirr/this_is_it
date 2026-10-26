import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type VideoAsset } from "./api";

// Fast ground-truth labelling: play the match video, press a key at the moment of each event.
// Labels stay in this browser (localStorage) until exported — they never mix into match stats.
// Export → JSON for scripts/eval/score_labels.py and futsal fine-tuning.

type Label = { ts_ms: number; type: string; team: string };

const TYPE_KEYS: Record<string, string> = { p: "pass", d: "dribble", s: "shot", g: "goal" };
const TEAM_KEYS: Record<string, string> = { "1": "team1", "2": "team2" };

export function LabelPage() {
  const { facilityId = "", matchId = "" } = useParams();
  const storeKey = `courtlens-labels-${matchId}`;
  const video = useRef<HTMLVideoElement>(null);
  const [assets, setAssets] = useState<VideoAsset[]>([]);
  const [assetId, setAssetId] = useState("");
  const [src, setSrc] = useState("");
  const [team, setTeam] = useState("team1");
  const [labels, setLabels] = useState<Label[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(storeKey) || "[]");
    } catch {
      return [];
    }
  });
  const [message, setMessage] = useState("");

  useEffect(() => {
    api.listVideos(facilityId, matchId).then((v) => {
      setAssets(v);
      if (v.length) setAssetId(v[0].id);
    });
  }, [facilityId, matchId]);

  useEffect(() => {
    if (assetId) api.getPlaybackUrl(facilityId, matchId, assetId).then((r) => setSrc(r.url));
  }, [facilityId, matchId, assetId]);

  useEffect(() => {
    try {
      localStorage.setItem(storeKey, JSON.stringify(labels));
    } catch {
      /* storage unavailable: export still works */
    }
  }, [labels, storeKey]);

  const mark = useCallback(
    (type: string) => {
      const v = video.current;
      if (!v) return;
      const ts_ms = Math.round(v.currentTime * 1000);
      setLabels((prev) => [...prev, { ts_ms, type, team }].sort((a, b) => a.ts_ms - b.ts_ms));
      setMessage(`${type} · ${team} @ ${(ts_ms / 1000).toFixed(2)} s`);
    },
    [team],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      const v = video.current;
      const k = e.key.toLowerCase();
      if (TYPE_KEYS[k]) mark(TYPE_KEYS[k]);
      else if (TEAM_KEYS[k]) setTeam(TEAM_KEYS[k]);
      else if (k === "z") setLabels((prev) => prev.slice(0, -1));
      else if (k === " " && v) {
        e.preventDefault();
        if (v.paused) v.play();
        else v.pause();
      } else if (k === "arrowleft" && v) v.currentTime = Math.max(0, v.currentTime - 2);
      else if (k === "arrowright" && v) v.currentTime += 2;
      else if (k === "," && v) v.playbackRate = Math.max(0.25, v.playbackRate - 0.25);
      else if (k === "." && v) v.playbackRate = Math.min(2, v.playbackRate + 0.25);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mark]);

  function exportJson() {
    const body = { match_id: matchId, asset_id: assetId, labelled_at: new Date().toISOString(), events: labels };
    const blob = new Blob([JSON.stringify(body, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `labels-${matchId}.json`;
    a.click();
  }

  const counts = labels.reduce<Record<string, number>>((acc, l) => ({ ...acc, [l.type]: (acc[l.type] || 0) + 1 }), {});

  return (
    <div className="bc-page">
      <header className="bc-nav">
        <Link to={`/facilities/${facilityId}/matches/${matchId}`}>← Match</Link>
        <span className="bc-nav-title">Label events (ground truth)</span>
      </header>
      <section className="panel">
        {assets.length > 1 && (
          <select value={assetId} onChange={(e) => setAssetId(e.target.value)}>
            {assets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.object_key.split("/").slice(-3).join("/")}
              </option>
            ))}
          </select>
        )}
        {src ? (
          <video ref={video} src={src} controls playsInline style={{ width: "100%", maxHeight: "70vh" }} />
        ) : (
          <p>Loading video…</p>
        )}
        <p className="bc-est-note">
          <b>P</b> pass · <b>D</b> dribble · <b>S</b> shot · <b>G</b> goal at the moment it happens ·{" "}
          <b>1</b>/<b>2</b> team in possession (now <b>{team}</b>) · <b>Space</b> play/pause · <b>←/→</b> 2 s ·{" "}
          <b>,</b>/<b>.</b> slower/faster · <b>Z</b> undo
        </p>
        <p>
          {Object.entries(counts)
            .map(([k, v]) => `${k} ${v}`)
            .join(" · ") || "No labels yet"}{" "}
          {message && <span className="bc-est-note">— last: {message}</span>}
        </p>
        <button className="primary" type="button" onClick={exportJson} disabled={!labels.length}>
          Export labels ({labels.length})
        </button>{" "}
        <button
          type="button"
          onClick={() => {
            if (window.confirm("Delete all labels for this match in this browser?")) setLabels([]);
          }}
          disabled={!labels.length}
        >
          Clear
        </button>
        <ol className="bc-label-list">
          {labels
            .slice(-15)
            .reverse()
            .map((l, i) => (
              <li key={`${l.ts_ms}-${i}`}>
                {(l.ts_ms / 1000).toFixed(2)} s — {l.type} ({l.team})
              </li>
            ))}
        </ol>
      </section>
    </div>
  );
}
