import { type FormEvent, useEffect, useState } from "react";
import { BrowserRouter, Link, Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import {
  api,
  CAMERA_LABELS,
  type CameraLabel,
  type CameraSync,
  type Facility,
  type FacilityMember,
  type FacilityRole,
  type RunStats,
  type MatchCamera,
  type Match,
  type MatchEvent,
  type MatchStats,
  type Player,
  type ProcessingJob,
  type Team,
  type Track,
  type VideoAsset,
  uploadFile,
} from "./api";
import "./App.css";
import { LabelPage } from "./LabelPage";

function useAuth() {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(!!localStorage.getItem("token"));
  useEffect(() => {
    if (!localStorage.getItem("token")) {
      setReady(true);
      return;
    }
    api
      .me()
      .then(() => setAuthed(true))
      .catch(() => {
        localStorage.removeItem("token");
        setAuthed(false);
      })
      .finally(() => setReady(true));
  }, []);
  return { ready, authed, setAuthed };
}

function AuthPage({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      if (mode === "register") await api.register(email, fullName, password);
      await api.login(email, password);
      onAuthed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    }
  }

  return (
    <div className="page auth">
      <header className="brand-block">
        <p className="brand">CourtLens</p>
        <h1>Match stats for your court</h1>
        <p className="lede">
          Upload match video, review goals, shots and passes, then share a report with coaches and players.
        </p>
      </header>
      <form className="panel" onSubmit={submit}>
        <div className="tabs">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>
            Sign in
          </button>
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>
            Create account
          </button>
        </div>
        {mode === "register" && (
          <label>
            Full name
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} required />
          </label>
        )}
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
        </label>
        {error && <p className="error">{error}</p>}
        <button type="submit" className="primary">
          {mode === "login" ? "Sign in" : "Register & continue"}
        </button>
      </form>
    </div>
  );
}

function FacilitiesPage() {
  const [facilities, setFacilities] = useState<Facility[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  async function load() {
    setFacilities(await api.listFacilities());
  }

  useEffect(() => {
    load().catch((e) => setError(String(e.message || e)));
  }, []);

  async function create(e: FormEvent) {
    e.preventDefault();
    await api.createFacility(name);
    setName("");
    await load();
  }

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="brand">CourtLens</p>
          <h1>Your courts</h1>
        </div>
        <div className="inline">
          <Link to="/phase0">Phase 0 feasibility</Link>
        <button
          type="button"
            className="ghost"
            onClick={() => {
              localStorage.removeItem("token");
              window.location.href = "/";
            }}
          >
            Sign out
          </button>
        </div>
      </header>
      <form className="panel inline" onSubmit={create}>
        <input placeholder="New facility name" value={name} onChange={(e) => setName(e.target.value)} required />
        <button type="submit" className="primary">
          Create
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      <ul className="list">
        {facilities.map((f) => (
          <li key={f.id}>
            <Link to={`/facilities/${f.id}`}>{f.name}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

function FacilityHome() {
  const { facilityId = "" } = useParams();
  const [teams, setTeams] = useState<Team[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [teamName, setTeamName] = useState("");
  const [playerName, setPlayerName] = useState("");
  const [shirt, setShirt] = useState(10);
  const [matchTitle, setMatchTitle] = useState("");
  const [members, setMembers] = useState<FacilityMember[]>([]);
  const [memberEmail, setMemberEmail] = useState("");
  const [memberRole, setMemberRole] = useState<FacilityRole>("staff");
  const [staffMessage, setStaffMessage] = useState("");
  const [runs, setRuns] = useState<Record<string, RunStats>>({});

  async function refresh() {
    const [t, p, m, mem] = await Promise.all([
      api.listTeams(facilityId),
      api.listPlayers(facilityId),
      api.listMatches(facilityId),
      api.listMembers(facilityId).catch(() => [] as FacilityMember[]),
    ]);
    setTeams(t);
    setPlayers(p);
    setMatches(m);
    setMembers(mem);
    const rs = await Promise.all(m.map((x) => api.getRunStats(facilityId, x.id).catch(() => null)));
    setRuns(Object.fromEntries(m.map((x, i) => [x.id, rs[i]]).filter(([, r]) => r && (r as RunStats).available)));
  }

  useEffect(() => {
    refresh().catch(console.error);
  }, [facilityId]);

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="brand">CourtLens</p>
          <h1>Squad & fixtures</h1>
        </div>
        <Link to="/">All courts</Link>
      </header>

      <section className="grid-2">
        <form
          className="panel"
          onSubmit={async (e) => {
            e.preventDefault();
            await api.createTeam(facilityId, teamName);
            setTeamName("");
            await refresh();
          }}
        >
          <h2>Teams</h2>
          <input value={teamName} onChange={(e) => setTeamName(e.target.value)} placeholder="Team name" required />
          <button className="primary" type="submit">
            Add team
          </button>
          <ul className="list compact">
            {teams.map((t) => (
              <li key={t.id}>{t.name}</li>
            ))}
          </ul>
        </form>

        <form
          className="panel"
          onSubmit={async (e) => {
            e.preventDefault();
            await api.createPlayer(facilityId, playerName, shirt, teams[0]?.id);
            setPlayerName("");
            await refresh();
          }}
        >
          <h2>Players</h2>
          <input value={playerName} onChange={(e) => setPlayerName(e.target.value)} placeholder="Player name" required />
          <input type="number" value={shirt} onChange={(e) => setShirt(Number(e.target.value))} min={0} max={99} />
          <button className="primary" type="submit">
            Add player
          </button>
          <ul className="list compact">
            {players.map((p) => (
              <li key={p.id}>
                {p.avatar_url ? <img className="player-thumb" src={p.avatar_url} alt="" /> : null}
                #{p.default_shirt_number ?? "—"} {p.display_name}
              </li>
            ))}
          </ul>
        </form>
      </section>

      {Object.keys(runs).length > 0 && (
        <section className="panel">
          <h2>Processing runs</h2>
          <p className="lede">
            Detection quality per match (camera A). Low boxes/track = tracker keeps losing players; ball coverage drives
            possession, passes and shots.
          </p>
          <div className="table-scroll">
            <table className="run-table">
              <thead>
                <tr>
                  <th>Match</th>
                  <th>Model</th>
                  <th className="num">Video s</th>
                  <th className="num">× realtime</th>
                  <th className="num">Player tracks</th>
                  <th className="num">Boxes / track</th>
                  <th className="num">Players</th>
                  <th className="num">Ball cov.</th>
                  <th className="num">Events</th>
                  <th className="num">ReID</th>
                  <th className="num">OCR</th>
                </tr>
              </thead>
              <tbody>
                {matches
                  .filter((m) => runs[m.id])
                  .map((m) => {
                    const r = runs[m.id];
                    const c = r.cameras?.[0] ?? { camera: "A" };
                    return (
                      <tr key={m.id}>
                        <td>
                          <Link to={`/facilities/${facilityId}/matches/${m.id}`}>{m.title}</Link>
                        </td>
                        <td>{c.model ?? "—"}</td>
                        <td className="num">{fmt(c.video_seconds_probed)}</td>
                        <td className="num">{fmt(c.realtime_factor, 2)}</td>
                        <td className="num">{c.person_tracks ?? "—"}</td>
                        <td className="num">{fmt(c.boxes_per_track, 1)}</td>
                        <td className="num">{r.counts?.players ?? "—"}</td>
                        <td className="num">{fmt(c.ball_coverage_pct, 1, "%")}</td>
                        <td className="num">{r.counts?.canonical_events ?? "—"}</td>
                        <td className="num">{c.reid_tracks ?? "—"}</td>
                        <td className="num">{c.jersey_ocr_tracks ?? "—"}</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="panel">
        <h2>Staff</h2>
        <p className="lede">
          Owners add people who already have a CourtLens account. Staff run matches; reviewers can view.
        </p>
        <form
          className="inline"
          onSubmit={async (e) => {
            e.preventDefault();
            try {
              await api.addMember(facilityId, memberEmail, memberRole);
              setMemberEmail("");
              setStaffMessage("");
              await refresh();
            } catch (err) {
              setStaffMessage(err instanceof Error ? err.message : "Could not add member");
            }
          }}
        >
          <input
            type="email"
            value={memberEmail}
            onChange={(e) => setMemberEmail(e.target.value)}
            placeholder="staff@court.com"
            required
          />
          <select value={memberRole} onChange={(e) => setMemberRole(e.target.value as FacilityRole)}>
            <option value="staff">Staff</option>
            <option value="reviewer">Reviewer</option>
            <option value="owner">Owner</option>
          </select>
          <button className="primary" type="submit">
            Add
          </button>
        </form>
        {staffMessage && <p className="lede">{staffMessage}</p>}
        <ul className="list compact">
          {members.map((m) => (
            <li key={m.id}>
              {m.email ?? m.user_id} <span className="badge">{m.role}</span>
              {m.role !== "owner" && (
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await api.removeMember(facilityId, m.user_id);
                      await refresh();
                    } catch (err) {
                      setStaffMessage(err instanceof Error ? err.message : "Could not remove member");
                    }
                  }}
                >
                  Remove
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>Matches</h2>
        <form
          className="inline"
          onSubmit={async (e) => {
            e.preventDefault();
            await api.createMatch(facilityId, matchTitle);
            setMatchTitle("");
            await refresh();
          }}
        >
          <input value={matchTitle} onChange={(e) => setMatchTitle(e.target.value)} placeholder="Match title" required />
          <button className="primary" type="submit">
            Create match
          </button>
        </form>
        <ul className="list">
          {matches.map((m) => (
            <li key={m.id}>
              <Link to={`/facilities/${facilityId}/matches/${m.id}`}>
                {m.title} <span className="badge">{m.status}</span>
              </Link>
            </li>
          ))}
          </ul>
      </section>
    </div>
  );
}

function formatMatchClock(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function heatFill(v: number): string {
  // Cold → hot over grass (Opta-style)
  if (v < 0.2) return `rgba(56, 189, 248, ${0.15 + v * 0.55})`;
  if (v < 0.45) return `rgba(250, 204, 21, ${0.28 + v * 0.45})`;
  if (v < 0.7) return `rgba(249, 115, 22, ${0.4 + v * 0.4})`;
  return `rgba(239, 68, 68, ${0.5 + v * 0.45})`;
}

function PitchHeatmap({
  grid,
  label,
}: {
  grid: number[][] | undefined;
  label: string;
}) {
  const uid = `ph-${label.replace(/\W+/g, "").slice(0, 18) || "map"}`;
  if (!grid?.length) {
    return <p className="bc-empty">Heatmap unavailable for this selection.</p>;
  }
  const rows = grid.length;
  const cols = grid[0]?.length || 1;
  // Standard pitch proportions (metres)
  const PW = 105;
  const PH = 68;
  const cellW = PW / cols;
  const cellH = PH / rows;
  const blobs: { x: number; y: number; r: number; fill: string; key: string }[] = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = grid[y][x];
      if (v < 0.06) continue;
      blobs.push({
        key: `${y}-${x}`,
        x: (x + 0.5) * cellW,
        y: (y + 0.5) * cellH,
        r: Math.max(cellW, cellH) * (0.7 + v * 1.1),
        fill: heatFill(v),
      });
    }
  }
  const grassId = `${uid}-grass`;
  const blurId = `${uid}-blur`;

  return (
    <div className="bc-heat-wrap">
      <div className="bc-pitch" role="img" aria-label={label}>
        <svg viewBox={`0 0 ${PW} ${PH}`} className="bc-pitch-svg" preserveAspectRatio="xMidYMid meet">
          <defs>
            <pattern id={grassId} width="6" height="68" patternUnits="userSpaceOnUse">
              <rect width="3" height="68" fill="#1f7a42" />
              <rect x="3" width="3" height="68" fill="#1a6b39" />
            </pattern>
            <filter id={blurId} x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="2.2" />
            </filter>
          </defs>
          <rect width={PW} height={PH} fill={`url(#${grassId})`} rx="1.2" />
          {/* Markings */}
          <g fill="none" stroke="rgba(255,255,255,0.88)" strokeWidth="0.35">
            <rect x="1.5" y="1.5" width={PW - 3} height={PH - 3} />
            <line x1={PW / 2} y1="1.5" x2={PW / 2} y2={PH - 1.5} />
            <circle cx={PW / 2} cy={PH / 2} r="9.15" />
            <circle cx={PW / 2} cy={PH / 2} r="0.55" fill="rgba(255,255,255,0.9)" stroke="none" />
            {/* Left penalty */}
            <rect x="1.5" y={(PH - 40.32) / 2} width="16.5" height="40.32" />
            <rect x="1.5" y={(PH - 18.32) / 2} width="5.5" height="18.32" />
            <circle cx="11" cy={PH / 2} r="0.45" fill="rgba(255,255,255,0.9)" stroke="none" />
            {/* Right penalty */}
            <rect x={PW - 1.5 - 16.5} y={(PH - 40.32) / 2} width="16.5" height="40.32" />
            <rect x={PW - 1.5 - 5.5} y={(PH - 18.32) / 2} width="5.5" height="18.32" />
            <circle cx={PW - 11} cy={PH / 2} r="0.45" fill="rgba(255,255,255,0.9)" stroke="none" />
            {/* Goals */}
            <rect x="0.35" y={(PH - 7.32) / 2} width="1.15" height="7.32" />
            <rect x={PW - 1.5} y={(PH - 7.32) / 2} width="1.15" height="7.32" />
          </g>
          <g filter={`url(#${blurId})`} style={{ mixBlendMode: "screen" }}>
            {blobs.map((b) => (
              <circle key={b.key} cx={b.x} cy={b.y} r={b.r} fill={b.fill} />
            ))}
          </g>
        </svg>
        <div className="bc-heat-legend" aria-hidden>
          <span>Low</span>
          <span className="bc-heat-scale" />
          <span>High</span>
        </div>
      </div>
      <p className="bc-heat-caption">{label}</p>
    </div>
  );
}

function PhotoLightbox({
  src,
  title,
  onClose,
}: {
  src: string;
  title: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="bc-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={onClose}
    >
      <button type="button" className="bc-lightbox-close" onClick={onClose} aria-label="Close">
        Close
      </button>
      <figure className="bc-lightbox-figure" onClick={(e) => e.stopPropagation()}>
        <img src={src} alt={title} className="bc-lightbox-img" />
        <figcaption>{title}</figcaption>
      </figure>
    </div>
  );
}

function ClipLightbox({
  src,
  title,
  onClose,
}: {
  src: string;
  title: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="bc-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={onClose}
    >
      <button type="button" className="bc-lightbox-close" onClick={onClose} aria-label="Close">
        Close
      </button>
      <div className="bc-clip-panel" onClick={(e) => e.stopPropagation()}>
        <video className="bc-clip-video" src={src} controls autoPlay playsInline />
        <div className="bc-clip-meta">
          <p className="bc-clip-title">{title}</p>
          <a className="bc-clip-download" href={src} download target="_blank" rel="noopener noreferrer">
            Download clip
          </a>
        </div>
      </div>
    </div>
  );
}

function statusPresentation(status?: string): { label: string; className: string } {
  switch (status) {
    case "processing":
      return { label: "Analysing", className: "is-live" };
    case "needs_review":
      return { label: "Ready for review", className: "is-done" };
    case "published":
      return { label: "Published", className: "is-done" };
    case "failed":
      return { label: "Failed", className: "" };
    default:
      return { label: status?.replaceAll("_", " ") || "Draft", className: "" };
  }
}

function fmt(v: number | null | undefined, digits = 0, suffix = ""): string {
  return v === null || v === undefined ? "—" : `${v.toFixed(digits)}${suffix}`;
}

// Click order. 4 corners = flat perspective; all 8 also fits lens distortion
// (SoccerTrack v2 wide camera: 11 m → 0.9 m mean error).
function courtLandmarks(L: number, W: number, R: number): { name: string; at: number[] }[] {
  return [
    { name: "near-left corner", at: [0, W] },
    { name: "far-left corner", at: [0, 0] },
    { name: "far-right corner", at: [L, 0] },
    { name: "near-right corner", at: [L, W] },
    { name: "halfway line, far end", at: [L / 2, 0] },
    { name: "halfway line, near end", at: [L / 2, W] },
    { name: "centre circle, far crossing", at: [L / 2, W / 2 - R] },
    { name: "centre circle, near crossing", at: [L / 2, W / 2 + R] },
  ];
}

function CourtCalibrationPanel({ facilityId, matchId }: { facilityId: string; matchId: string }) {
  const [cameras, setCameras] = useState<MatchCamera[]>([]);
  const [label, setLabel] = useState<CameraLabel | null>(null);
  const [frame, setFrame] = useState<{ url: string; width: number; height: number } | null>(null);
  const [points, setPoints] = useState<number[][]>([]);
  const [length, setLength] = useState(40);
  const [width, setWidth] = useState(20);
  const [radius, setRadius] = useState(3);
  const [note, setNote] = useState("");
  const landmarks = courtLandmarks(length, width, radius);

  async function load() {
    setCameras(await api.listCameras(facilityId, matchId).catch(() => [] as MatchCamera[]));
  }
  useEffect(() => {
    load();
  }, [facilityId, matchId]);

  async function open(cam: MatchCamera) {
    setNote("");
    setLabel(cam.label);
    setPoints(cam.calibration?.image_points ?? []);
    setLength(cam.calibration?.court_length_m ?? 40);
    setWidth(cam.calibration?.court_width_m ?? 20);
    try {
      setFrame(await api.getCameraFrame(facilityId, matchId, cam.label));
    } catch (err) {
      setFrame(null);
      setNote(err instanceof Error ? err.message : "Could not load a frame");
    }
  }

  function onClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!frame || points.length >= landmarks.length) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * frame.width;
    const y = ((e.clientY - r.top) / r.height) * frame.height;
    setPoints([...points, [Math.round(x), Math.round(y)]]);
  }

  async function save() {
    if (!label || !frame || points.length < 4) return;
    try {
      const saved = await api.setCalibration(facilityId, matchId, label, {
        image_points: points,
        court_points: landmarks.slice(0, points.length).map((l) => l.at),
        court_length_m: length,
        court_width_m: width,
        frame_w: frame.width,
        frame_h: frame.height,
      });
      const err = saved.calibration.fit_error_m;
      setNote(
        `Camera ${label} calibrated (${points.length} points${err !== undefined ? `, fit error ${err.toFixed(2)} m` : ""}) — distance and speed now in metres / km/h` +
          (points.length < 8 ? ". Add points 5–8 for wide or distorted cameras." : ""),
      );
      await load();
    } catch (err) {
      setNote(err instanceof Error ? err.message : "Calibration failed");
    }
  }

  return (
    <section className="panel">
      <h2>Court calibration</h2>
      <p className="lede">
        Click court landmarks once per fixed camera, in the order shown. The 4 corners are enough for a flat
        view; adding the halfway-line ends and centre-circle crossings (8 points) also corrects lens distortion —
        recommended for wide CCTV. Distance and speed switch to metres and km/h.
      </p>
      <div className="inline">
        {cameras
          .filter((c) => c.has_video)
          .map((c) => (
            <button key={c.label} type="button" className={label === c.label ? "active" : ""} onClick={() => open(c)}>
              Camera {c.label} {c.calibration ? "✓" : ""}
            </button>
          ))}
        {!cameras.some((c) => c.has_video) && <span className="lede">Upload a camera video first.</span>}
      </div>
      {frame && label && (
        <>
          <div className="calib-frame">
            <img src={frame.url} alt={`Camera ${label}`} onClick={onClick} />
            {points.map((p, i) => (
              <span
                key={i}
                className="calib-dot"
                style={{ left: `${(100 * p[0]) / frame.width}%`, top: `${(100 * p[1]) / frame.height}%` }}
              >
                {i + 1}
              </span>
            ))}
          </div>
          <p className="lede">
            {points.length < landmarks.length
              ? `Click point ${points.length + 1}: ${landmarks[points.length].name}${points.length >= 4 ? " (optional)" : ""}`
              : "All 8 landmarks set."}
          </p>
          <div className="inline">
            <label>
              Length (m) <input type="number" value={length} min={10} max={120} onChange={(e) => setLength(Number(e.target.value))} />
            </label>
            <label>
              Width (m) <input type="number" value={width} min={5} max={90} onChange={(e) => setWidth(Number(e.target.value))} />
            </label>
            <label>
              Centre circle radius (m){" "}
              <input type="number" value={radius} min={1} max={10} step={0.05} onChange={(e) => setRadius(Number(e.target.value))} />
            </label>
            <button type="button" disabled={!points.length} onClick={() => setPoints(points.slice(0, -1))}>
              Undo point
            </button>
            <button type="button" onClick={() => setPoints([])}>
              Reset
            </button>
            <button className="primary" type="button" disabled={points.length < 4} onClick={save}>
              Save calibration
            </button>
          </div>
        </>
      )}
      {note && <p className="lede">{note}</p>}
    </section>
  );
}

function moveCell(row: { distance?: number; distance_m?: number | null; avg_speed_kmh?: number | null; top_speed_kmh?: number | null }) {
  if (row.distance_m !== null && row.distance_m !== undefined) {
    return (
      <td className="num" title={`avg ${row.avg_speed_kmh ?? "—"} km/h · top ${row.top_speed_kmh ?? "—"} km/h`}>
        {Math.round(row.distance_m)} m
      </td>
    );
  }
  return <td className="num">{Math.round(row.distance ?? 0)}</td>;
}

function parseClock(text: string): number | null {
  const m = text.trim().match(/^(\d+):([0-5]?\d)(?:\.(\d))?$/);
  if (!m) return null;
  return (Number(m[1]) * 60 + Number(m[2])) * 1000 + (m[3] ? Number(m[3]) * 100 : 0);
}

// Every event staff can tag; fouls, cards, saves and set pieces are staff-only (no model yet)
const STAFF_EVENT_TYPES: [string, string][] = [
  ["goal", "Goal"],
  ["shot", "Shot"],
  ["save", "Keeper save"],
  ["pass", "Pass"],
  ["assist", "Assist"],
  ["dribble", "Dribble"],
  ["interception", "Interception / ball won"],
  ["foul", "Foul"],
  ["yellow_card", "Yellow card"],
  ["red_card", "Red card"],
  ["own_goal", "Own goal"],
  ["corner", "Corner"],
  ["kick_in", "Kick-in"],
  ["free_kick", "Free kick"],
  ["penalty", "Penalty (6 m / 10 m)"],
];

function AddEventForm({
  players,
  onAdd,
  onExport,
  teams = [],
}: {
  players: Player[];
  onAdd: (body: {
    event_type: string;
    canonical_ts_ms: number;
    player_id?: string;
    team_id?: string;
    on_target?: boolean;
  }) => Promise<void>;
  onExport: () => Promise<void>;
  teams?: { id: string; name: string }[];
}) {
  const [type, setType] = useState("goal");
  const [teamId, setTeamId] = useState("");
  const [onTarget, setOnTarget] = useState(false);
  const [clock, setClock] = useState("");
  const [playerId, setPlayerId] = useState("");
  const ts = parseClock(clock);
  return (
    <form
      className="inline"
      onSubmit={async (e) => {
        e.preventDefault();
        if (ts === null) return;
        await onAdd({
          event_type: type,
          canonical_ts_ms: ts,
          player_id: playerId || undefined,
          team_id: teamId || undefined,
          on_target: type === "shot" && onTarget ? true : undefined,
        });
        setClock("");
      }}
    >
      <select value={type} onChange={(e) => setType(e.target.value)}>
        {STAFF_EVENT_TYPES.map(([t, label]) => (
          <option key={t} value={t}>
            {label}
          </option>
        ))}
      </select>
      <input value={clock} onChange={(e) => setClock(e.target.value)} placeholder="mm:ss" size={6} />
      <select value={playerId} onChange={(e) => setPlayerId(e.target.value)}>
        <option value="">player (optional)</option>
        {players.map((p) => (
          <option key={p.id} value={p.id}>
            {p.display_name}
          </option>
        ))}
      </select>
      {teams.length > 0 && !playerId && (
        <select value={teamId} onChange={(e) => setTeamId(e.target.value)} title="For team events (corner, kick-in, foul by an unknown player)">
          <option value="">team (optional)</option>
          {teams.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      )}
      {type === "shot" && (
        <label className="inline-check">
          <input type="checkbox" checked={onTarget} onChange={(e) => setOnTarget(e.target.checked)} /> on target
        </label>
      )}
      <button className="primary" type="submit" disabled={ts === null}>
        {type === "goal" ? "Add goal" : "Add event"}
      </button>
      <button type="button" onClick={onExport} title="Confirmed events as JSON ground truth">
        Export ground truth
      </button>
    </form>
  );
}

const pctOr = (v: number | null | undefined) => (v == null ? "–" : `${v.toFixed(0)}%`);

// Rows of the team comparison: [label, value, hint]
const TEAM_STAT_ROWS: [string, (t: NonNullable<MatchStats["by_team"]>[number]) => string | number, string][] = [
  ["Score", (t) => t.score ?? t.goal, "Goals + opponent own goals"],
  ["Possession", (t) => pctOr(t.possession_pct), "Share of ball-control time (estimated)"],
  ["Shots", (t) => t.shot, "Estimated + staff-tagged"],
  ["Shots on target", (t) => t.shot_on_target ?? 0, "Followed by a goal or save, or marked by staff"],
  ["Conversion", (t) => pctOr(t.conversion_pct), "Goals ÷ shots"],
  ["Passes completed", (t) => t.pass, "Estimated"],
  ["Pass accuracy", (t) => pctOr(t.pass_accuracy_pct), "Completed ÷ (completed + intercepted)"],
  ["Assists", (t) => t.assist, ""],
  ["Dribbles", (t) => t.dribble, "Estimated"],
  ["Interceptions", (t) => t.interception ?? 0, "Ball won from the other team"],
  ["Balls lost", (t) => t.ball_lost ?? 0, "Ball given to the other team"],
  ["Keeper saves", (t) => t.save ?? 0, "Staff-tagged"],
  ["Fouls (1st / 2nd half)", (t) => `${t.foul ?? 0} (${(t.fouls_by_half ?? [0, 0]).join(" / ")})`, "Accumulated fouls: from the 6th in a half, a 10 m kick"],
  ["Yellow / red cards", (t) => `${t.yellow_card ?? 0} / ${t.red_card ?? 0}`, "Staff-tagged"],
  ["Corners", (t) => t.corner ?? 0, "Staff-tagged"],
  ["Kick-ins", (t) => t.kick_in ?? 0, "Staff-tagged"],
  ["Free kicks", (t) => t.free_kick ?? 0, "Staff-tagged"],
  ["Penalties", (t) => t.penalty ?? 0, "Staff-tagged"],
  ["Players", (t) => t.players, "Players with tracks"],
];

// Staff identity board: every person track as a crop, grouped by the player it is linked to.
// Move a crop, merge a whole player into another, or mark crops as not a player (referee, coach, crowd).
function IdentityBoard({
  tracks,
  players,
  onMove,
}: {
  tracks: Track[];
  players: Player[];
  onMove: (trackIds: string[], playerId: string | null, note: string) => Promise<void>;
}) {
  const [showAll, setShowAll] = useState<Record<string, boolean>>({});
  const people = tracks.filter((t) => t.class_name !== "sports_ball");
  const groups = new Map<string, Track[]>();
  for (const t of people) {
    const key = t.identity?.player_id || "";
    groups.set(key, [...(groups.get(key) ?? []), t]);
  }
  const name = (id: string) => players.find((p) => p.id === id)?.display_name || id.slice(0, 8);
  const order = [...groups.keys()].sort((a, b) => (a === "" ? 1 : b === "" ? -1 : name(a).localeCompare(name(b))));
  const dur = (t: Track) => t.end_ms - t.start_ms;
  const target = (current: string, onPick: (v: string | null) => void, label: string) => (
    <select
      value=""
      onChange={(e) => {
        const v = e.target.value;
        if (v) onPick(v === "__none" ? null : v);
      }}
    >
      <option value="">{label}</option>
      {current !== "" && <option value="__none">Not a player (ref / coach / crowd)</option>}
      {players
        .filter((p) => p.id !== current)
        .map((p) => (
          <option key={p.id} value={p.id}>
            #{p.default_shirt_number ?? "—"} {p.display_name}
          </option>
        ))}
    </select>
  );
  if (!people.length)
    return (
      <section className="panel">
        <h2>Player identity board</h2>
        <p className="bc-empty">No tracks yet — process a match first.</p>
      </section>
    );
  return (
    <section className="panel">
      <h2>Player identity board</h2>
      <p className="lede">
        Each crop is one tracked fragment. If a player&apos;s row shows someone else, move that crop; if two rows
        are the same person, merge them. Per-player stats follow these links.
      </p>
      {order.map((pid) => {
        const list = [...(groups.get(pid) ?? [])].sort((a, b) => dur(b) - dur(a));
        const shown = showAll[pid] ? list : list.slice(0, 18);
        const secs = Math.round(list.reduce((n, t) => n + dur(t), 0) / 1000);
        return (
          <div key={pid || "none"} className="id-group">
            <div className="id-group-head">
              <strong>{pid ? name(pid) : "Not linked to a player"}</strong>
              <span className="muted">
                {list.length} crops · {secs} s on camera
              </span>
              {pid &&
                target(pid, (to) => onMove(list.map((t) => t.id), to,
                  to ? `Merged ${name(pid)} into ${name(to)}` : `${name(pid)} marked as not a player`),
                  "Merge whole player into…")}
            </div>
            <div className="id-grid">
              {shown.map((t) => (
                <figure key={t.id} className="id-tile" title={`${t.track_key} · cam ${t.camera_label} · ${(dur(t) / 1000).toFixed(1)} s`}>
                  {t.thumb_url ? <img src={t.thumb_url} alt="" loading="lazy" /> : <div className="id-noimg">{t.track_key}</div>}
                  <figcaption>
                    {formatMatchClock(t.start_ms)} · {(dur(t) / 1000).toFixed(0)}s
                    {target(pid, (to) => onMove([t.id], to, to ? `Moved crop to ${name(to)}` : "Crop marked as not a player"), "Move…")}
                  </figcaption>
                </figure>
              ))}
            </div>
            {list.length > shown.length && (
              <button type="button" onClick={() => setShowAll({ ...showAll, [pid]: true })}>
                Show all {list.length}
              </button>
            )}
          </div>
        );
      })}
    </section>
  );
}

function TeamStatsPanel({
  teams,
  names,
}: {
  teams: NonNullable<MatchStats["by_team"]>;
  names: Record<string, string>;
}) {
  if (!teams.length) return null;
  const label = (t: (typeof teams)[number], i: number) =>
    (t.team_id && names[t.team_id]) || (t.team_id ? `Team ${i + 1}` : "Unassigned");
  return (
    <section className="panel">
      <h2>Team stats</h2>
      <div className="table-scroll">
        <table className="bc-table">
          <thead>
            <tr>
              <th>Stat</th>
              {teams.map((t, i) => (
                <th key={t.team_id ?? i} className="num">
                  {label(t, i)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TEAM_STAT_ROWS.map(([name, value, hint]) => (
              <tr key={name}>
                <td title={hint}>{name}</td>
                {teams.map((t, i) => (
                  <td key={t.team_id ?? i} className="num">
                    {value(t)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RunStatsPanel({ stats }: { stats: RunStats | null }) {
  if (!stats?.available) {
    return (
      <section className="panel">
        <h2>Run stats</h2>
        <p className="lede">No processing run recorded for this match yet.</p>
      </section>
    );
  }
  return (
    <section className="panel">
      <h2>Run stats</h2>
      <p className="lede">
        {stats.source === "offline" ? "Kaggle GPU run imported" : "Worker run"}
        {stats.wall_seconds ? ` · ${fmt(stats.wall_seconds)} s total` : ""}
        {stats.counts
          ? ` · ${stats.counts.tracks} tracks → ${stats.counts.players} players · ${stats.counts.canonical_events} events`
          : ""}
      </p>
      <div className="table-scroll">
        <table className="run-table">
          <thead>
            <tr>
              <th>Cam</th>
              <th>Model</th>
              <th className="num" title="Processed frames (every Nth)">Frames</th>
              <th className="num" title="Video seconds per wall second">× realtime</th>
              <th className="num">Player tracks</th>
              <th className="num" title="Detections per track — low means the tracker keeps losing players">Boxes / track</th>
              <th className="num">Ref tracks</th>
              <th className="num" title="Processed frames with a ball">Ball cov.</th>
              <th className="num">ReID tracks</th>
              <th className="num">OCR tracks</th>
            </tr>
          </thead>
          <tbody>
            {(stats.cameras ?? []).map((c) => (
              <tr key={c.camera}>
                <td>{c.camera}</td>
                <td>
                  {c.model ?? "—"}
                  {c.imgsz ? ` @${c.imgsz}px` : ""}
                </td>
                <td className="num">
                  {c.frames_processed ?? "—"}
                  {c.frame_stride ? ` /${c.frame_stride}` : ""}
                </td>
                <td className="num">{fmt(c.realtime_factor, 2)}</td>
                <td className="num">{c.person_tracks ?? "—"}</td>
                <td className="num">{fmt(c.boxes_per_track, 1)}</td>
                <td className="num">{c.referee_tracks ?? "—"}</td>
                <td className="num">{fmt(c.ball_coverage_pct, 1, "%")}</td>
                <td className="num">{c.reid_tracks ?? "—"}</td>
                <td className="num">{c.jersey_ocr_tracks ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!!stats.offsets?.length && (
        <p className="lede">
          Sync:{" "}
          {stats.offsets
            .map((o) => `${o.camera} ${o.offset_ms >= 0 ? "+" : ""}${o.offset_ms} ms${o.method ? ` (${o.method})` : ""}`)
            .join(" · ")}
        </p>
      )}
    </section>
  );
}

function MatchPage() {
  const { facilityId = "", matchId = "" } = useParams();
  const [match, setMatch] = useState<Match | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [videos, setVideos] = useState<VideoAsset[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [syncs, setSyncs] = useState<CameraSync[]>([]);
  const [runStats, setRunStats] = useState<RunStats | null>(null);
  const [teamNames, setTeamNames] = useState<Record<string, string>>({});
  const [offsetDraft, setOffsetDraft] = useState<Partial<Record<CameraLabel, number>>>({});
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<"center" | "setup" | "identity">("center");
  const [heatPlayerId, setHeatPlayerId] = useState<string>("match");
  const [heatKind, setHeatKind] = useState<"presence" | "onball">("presence");
  const [photoModal, setPhotoModal] = useState<{ src: string; title: string } | null>(null);
  const [clipModal, setClipModal] = useState<{ src: string; title: string } | null>(null);
  const [clipBusyId, setClipBusyId] = useState<string | null>(null);

  async function refresh() {
    const [m, p, v, j, sync] = await Promise.all([
      api.getMatch(facilityId, matchId),
      api.listPlayers(facilityId),
      api.listVideos(facilityId, matchId),
      api.listJobs(facilityId, matchId),
      api.getSync(facilityId, matchId).catch(() => [] as CameraSync[]),
    ]);
    setMatch(m);
    setPlayers(p);
    setVideos(v);
    setJobs(j);
    setSyncs(sync);
    setOffsetDraft(Object.fromEntries(sync.map((s) => [s.camera_label, s.offset_ms])));
    const [tr, ev, st, rs] = await Promise.all([
      api.listTracks(facilityId, matchId).catch(() => [] as Track[]),
      api.listEvents(facilityId, matchId).catch(() => [] as MatchEvent[]),
      api.getStats(facilityId, matchId).catch(() => null),
      api.getRunStats(facilityId, matchId).catch(() => null),
    ]);
    setTracks(tr);
    setEvents(ev);
    setStats(st);
    setRunStats(rs);
    api
      .listTeams(facilityId)
      .then((ts) => setTeamNames(Object.fromEntries(ts.map((t) => [t.id, t.name]))))
      .catch(() => undefined);
  }

  useEffect(() => {
    refresh().catch(console.error);
    const id = window.setInterval(() => {
      api.listJobs(facilityId, matchId).then(setJobs).catch(() => undefined);
    }, 4000);
    return () => clearInterval(id);
  }, [facilityId, matchId]);

  async function onUpload(label: CameraLabel, file: File | null) {
    if (!file) return;
    setBusy(true);
    setMessage(`Uploading camera ${label}…`);
    try {
      const { upload_url } = await api.getUploadUrl(facilityId, matchId, label, file.name, file.size);
      await uploadFile(upload_url, file);
      setMessage(`Camera ${label} uploaded`);
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function refreshResults() {
    setEvents(await api.listEvents(facilityId, matchId));
    setStats(await api.getStats(facilityId, matchId));
    setTracks(await api.listTracks(facilityId, matchId));
  }

  const playerName = (id: string | null) =>
    id ? players.find((p) => p.id === id)?.display_name || id.slice(0, 8) : "Unassigned";

  const status = statusPresentation(match?.status);
  const latestJob = jobs[0];
  const goals = stats?.totals.goal ?? 0;
  const shots = stats?.totals.shot ?? 0;
  const passes = stats?.totals.pass ?? 0;
  const assists = stats?.totals.assist ?? 0;
  const dribbles = stats?.totals.dribble ?? 0;
  const leaderboard = [...(stats?.by_player || [])].sort(
    (a, b) =>
      (b.possession_pct ?? 0) * 2 +
      b.goal * 10 +
      b.shot * 3 +
      b.pass -
      ((a.possession_pct ?? 0) * 2 + a.goal * 10 + a.shot * 3 + a.pass),
  );
  const timeline = [...events]
    .filter((e) => e.confirmation !== "rejected")
    .sort((a, b) => a.canonical_ts_ms - b.canonical_ts_ms);
  const selectedHeat =
    heatPlayerId === "match"
      ? heatKind === "onball"
        ? stats?.heatmap?.match_possession
        : stats?.heatmap?.match
      : heatKind === "onball"
        ? leaderboard.find((p) => p.player_id === heatPlayerId)?.possession_heatmap
        : leaderboard.find((p) => p.player_id === heatPlayerId)?.heatmap;
  const heatLabel =
    heatPlayerId === "match"
      ? heatKind === "onball"
        ? "Match on-ball heatmap"
        : "Match heatmap (all players)"
      : `${leaderboard.find((p) => p.player_id === heatPlayerId)?.display_name || "Player"} ${
          heatKind === "onball" ? "on the ball" : "movement"
        }`;

  const eventPlayer = (ev: MatchEvent) => {
    const primary =
      ev.participants.find((p) => ["passer", "shooter", "scorer", "carrier", "primary"].includes(p.role)) ||
      ev.participants[0];
    return playerName(primary?.player_id ?? null);
  };

  return (
    <div className="page broadcast">
      <header className="bc-nav">
        <p className="brand">CourtLens</p>
        <div className="bc-nav-meta">
          <Link to={`/facilities/${facilityId}/matches/${matchId}/label`}>Label events</Link>{" "}
          <Link to={`/facilities/${facilityId}`}>All matches</Link>
        </div>
      </header>

      <section className="bc-scoreboard">
        <div className="bc-scoreboard-top">
          <div>
            <p className="bc-kicker">Match centre</p>
            <h1 className="bc-match-title">{match?.title || "Match"}</h1>
          </div>
          <div className={`bc-status ${status.className}`}>
            <span className="bc-status-dot" />
            {status.label}
          </div>
        </div>

        <div className="bc-stat-strip">
          <div className="bc-stat is-goal">
            <div className="bc-stat-value">{goals}</div>
            <div className="bc-stat-label">Goals</div>
          </div>
          <div className="bc-stat is-shot">
            <div className="bc-stat-value">{shots}</div>
            <div className="bc-stat-label">Shots <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat is-pass">
            <div className="bc-stat-value">{passes}</div>
            <div className="bc-stat-label">Passes <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{assists}</div>
            <div className="bc-stat-label">Assists <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{dribbles}</div>
            <div className="bc-stat-label">Dribbles <em className="bc-est">est.</em></div>
          </div>
        </div>

        <p className="bc-est-note">
          Goals count only after staff confirm or add them on the timeline. Shots, passes, assists and dribbles are
          model estimates until reviewed.
        </p>
        <div className="bc-confidence">
          <span>Confirmed {stats?.totals.confirmed ?? 0}</span>
          <span>Estimated {stats?.totals.estimated ?? 0}</span>
          <span>Rejected {stats?.totals.rejected ?? 0}</span>
          {latestJob && latestJob.status === "running" && (
            <span>
              Pipeline {latestJob.stage || "…"} · {latestJob.progress_pct}%
            </span>
          )}
        </div>
        {latestJob && latestJob.status === "running" && (
          <div className="bc-progress">
            <div className="bc-progress-bar">
              <span style={{ width: `${latestJob.progress_pct}%` }} />
            </div>
            <p className="bc-progress-label">Match stats update when processing finishes.</p>
          </div>
        )}
      </section>

      <nav className="bc-view-tabs" aria-label="Match views">
        <button type="button" className={view === "center" ? "active" : ""} onClick={() => setView("center")}>
          Match centre
        </button>
        <button type="button" className={view === "setup" ? "active" : ""} onClick={() => setView("setup")}>
          Setup & process
        </button>
        <button type="button" className={view === "identity" ? "active" : ""} onClick={() => setView("identity")}>
          Player identity
        </button>
      </nav>

      {view === "center" && (
        <>
          <RunStatsPanel stats={runStats} />
          <TeamStatsPanel teams={stats?.by_team ?? []} names={teamNames} />
          {(stats?.pass_network ?? []).length > 0 && (
            <section className="panel">
              <h2>Passing combinations</h2>
              <table className="bc-table">
                <thead>
                  <tr>
                    <th>From</th>
                    <th>To</th>
                    <th className="num">Passes</th>
                  </tr>
                </thead>
                <tbody>
                  {(stats?.pass_network ?? []).slice(0, 10).map((e) => (
                    <tr key={`${e.from}-${e.to}`}>
                      <td>{playerName(e.from)}</td>
                      <td>{playerName(e.to)}</td>
                      <td className="num">{e.passes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
          <div className="bc-layout">
            <section className="bc-board">
              <div className="bc-board-head">
                <h2>Squad board</h2>
                <span>POS% · Min · Touches · Move · G · S · A · D · P</span>
              </div>
              {leaderboard.length ? (
                <>
                  <table className="bc-table">
                    <thead>
                      <tr>
                        <th>Player</th>
                        <th className="num" title="Possession share">POS%</th>
                        <th className="num" title="On-camera minutes">Min</th>
                        <th className="num" title="Ball touches">Touches</th>
                        <th
                          className="num"
                          title="Metres run when the camera is court-calibrated (hover for km/h); otherwise relative movement in the camera view (% of frame diagonal)."
                        >
                          Move
                        </th>
                        <th className="num" title="Goals">G</th>
                        <th className="num" title="Shots">S</th>
                        <th className="num" title="Assists">A</th>
                        <th className="num" title="Dribbles">D</th>
                        <th className="num" title="Passes">P</th>
                        <th className="num" title="Passes received">Rec</th>
                <th className="num" title="Pass accuracy: completed ÷ (completed + intercepted)">P%</th>
                <th className="num" title="Shots on target">SoT</th>
                <th className="num" title="Interceptions (ball won)">Int</th>
                <th className="num" title="Balls lost to the other team">Lost</th>
                <th className="num" title="Keeper saves">Sv</th>
                <th className="num" title="Fouls committed">F</th>
                <th className="num" title="Yellow / red cards">Cards</th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboard.map((row, idx) => {
                        const rosterPlayer = players.find((p) => p.id === row.player_id);
                        const shirt = row.shirt_number ?? rosterPlayer?.default_shirt_number ?? idx + 1;
                        const avatar = row.avatar_url || rosterPlayer?.avatar_url;
                        return (
                          <tr
                            key={row.player_id}
                            className={heatPlayerId === row.player_id ? "is-selected" : ""}
                            onClick={() => setHeatPlayerId(row.player_id)}
                          >
                            <td>
                              <div className="bc-player">
                                {avatar ? (
                                  <button
                                    type="button"
                                    className="bc-avatar-btn"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setPhotoModal({ src: avatar, title: row.display_name });
                                    }}
                                    aria-label={`Open photo of ${row.display_name}`}
                                  >
                                    <img className="bc-avatar" src={avatar} alt="" />
                                  </button>
                                ) : (
                                  <span className="bc-avatar-fallback">#{shirt}</span>
                                )}
                                <span className="bc-shirt">{shirt}</span>
                                <div>
                                  <span className="bc-player-name">{row.display_name}</span>
                                  <span className="bc-player-sub">{row.attribution}</span>
                                  <div className="bc-poss-bar" aria-hidden>
                                    <span style={{ width: `${Math.min(100, row.possession_pct ?? 0)}%` }} />
                                  </div>
                                </div>
                              </div>
                            </td>
                            <td className="num">{(row.possession_pct ?? 0).toFixed(0)}</td>
                            <td className="num">{(row.minutes ?? 0).toFixed(1)}</td>
                            <td className="num">{row.touches ?? 0}</td>
                            {moveCell(row)}
                            <td className="num">{row.goal}</td>
                            <td className="num">{row.shot}</td>
                            <td className="num">{row.assist ?? 0}</td>
                            <td className="num">{row.dribble ?? 0}</td>
                            <td className="num">{row.pass}</td>
                            <td className="num">{row.pass_received ?? 0}</td>
                  <td className="num">{pctOr(row.pass_accuracy_pct)}</td>
                  <td className="num">{row.shot_on_target ?? 0}</td>
                  <td className="num">{row.interception ?? 0}</td>
                  <td className="num">{row.ball_lost ?? 0}</td>
                  <td className="num">{row.save ?? 0}</td>
                  <td className="num">{row.foul ?? 0}</td>
                  <td className="num">
                    {(row.yellow_card ?? 0) + (row.red_card ?? 0) > 0 ? `${row.yellow_card ?? 0}Y ${row.red_card ?? 0}R` : "–"}
                  </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                              <p className="bc-table-note">
                    POS% possession · Min on camera · A assist · D dribble · Rec = passes received.
                    {leaderboard.some((r) => r.distance_m != null)
                      ? " Move is metres run on the calibrated court (hover for km/h)."
                      : " Move is relative camera distance, not metres — calibrate the court (Setup) for metres and km/h."}{" "}
                    Tap a player for their pitch map.
                  </p>
                </>
              ) : (
                <p className="bc-empty">No player stats yet. Run analysis from Setup & process.</p>
              )}
            </section>

            <section className="bc-board">
              <div className="bc-board-head">
                <h2>Match timeline</h2>
                <span>{timeline.length} moments</span>
              </div>
              <AddEventForm
                players={players}
                teams={Object.entries(teamNames).map(([id, name]) => ({ id, name }))}
                onAdd={async (body) => {
                  await api.createEvent(facilityId, matchId, body);
                  setMessage(`Added ${body.event_type} at ${formatMatchClock(body.canonical_ts_ms)}`);
                  await refreshResults();
                }}
                onExport={async () => {
                  const gt = await api.exportGroundTruth(facilityId, matchId);
                  const blob = new Blob([JSON.stringify(gt, null, 2)], { type: "application/json" });
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(blob);
                  a.download = `ground-truth-${matchId}.json`;
                  a.click();
                  setMessage(`Exported ${gt.events.length} confirmed events`);
                }}
              />
              <div className="bc-timeline">
                {timeline.length ? (
                  timeline.map((ev) => (
                    <article key={ev.id} className="bc-moment">
                      <div className="bc-clock">{formatMatchClock(ev.canonical_ts_ms)}</div>
                      <div className="bc-moment-body">
                        <span className={`bc-moment-type ${ev.event_type}`}>{ev.event_type}</span>
                        <p className="bc-moment-main">{eventPlayer(ev)}</p>
                        <p className="bc-moment-meta">
                          {ev.confirmation}
                          {ev.confidence != null ? ` · ${(ev.confidence * 100).toFixed(0)}% confidence` : ""}
                        </p>
                        <div className="bc-moment-actions">
                          <button
                            type="button"
                            className="primary"
                            onClick={async () => {
                              await api.reviewEvent(facilityId, matchId, ev.id, "accept");
                              await refreshResults();
                              setMessage("Moment confirmed");
                            }}
                          >
                            Confirm
                          </button>
                          <button
                            type="button"
                            className="danger"
                            onClick={async () => {
                              await api.reviewEvent(facilityId, matchId, ev.id, "reject");
                              await refreshResults();
                              setMessage("Moment rejected");
                            }}
                          >
                            Reject
                          </button>
                          <select
                            aria-label="Assign player"
                            defaultValue=""
                            onChange={async (e) => {
                              if (!e.target.value) return;
                              await api.reviewEvent(facilityId, matchId, ev.id, "edit", {
                                player_id: e.target.value,
                              });
                              await refreshResults();
                              setMessage("Player assigned");
                            }}
                          >
                            <option value="">Assign player</option>
                            {players.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.display_name}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            disabled={clipBusyId === ev.id}
                            onClick={async () => {
                              setClipBusyId(ev.id);
                              try {
                                const clip = await api.createClip(facilityId, matchId, ev.id);
                                setClipModal({
                                  src: clip.playback_url,
                                  title: `${ev.event_type} · ${formatMatchClock(ev.canonical_ts_ms)} · ${eventPlayer(ev)}`,
                                });
                              } catch (err) {
                                setMessage(err instanceof Error ? err.message : "Could not load clip");
                              } finally {
                                setClipBusyId(null);
                              }
                            }}
                          >
                            {clipBusyId === ev.id ? "Loading…" : "Watch clip"}
                          </button>
                        </div>
                      </div>
                    </article>
                  ))
                ) : (
                  <p className="bc-empty">Goals, shots and passes show here after the match is processed.</p>
                )}
              </div>
            </section>
          </div>

          <section className="bc-board bc-heat-board">
            <div className="bc-board-head">
              <h2>Pitch map</h2>
              <label className="bc-heat-select">
                Player
                <select value={heatPlayerId} onChange={(e) => setHeatPlayerId(e.target.value)}>
                  <option value="match">Whole match</option>
                  {leaderboard.map((p) => (
                    <option key={p.player_id} value={p.player_id}>
                      {p.display_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="bc-heat-select">
                Map
                <select value={heatKind} onChange={(e) => setHeatKind(e.target.value as "presence" | "onball")}>
                  <option value="presence">Movement</option>
                  <option value="onball">On the ball</option>
                </select>
              </label>
            </div>
            <PitchHeatmap grid={selectedHeat} label={heatLabel} />
            {stats?.heatmap?.note && <p className="bc-table-note">{stats.heatmap.note}</p>}
          </section>

          <div className="bc-footer-actions">
            <button type="button" onClick={() => refresh()}>
              Refresh stats
            </button>
            <button
              className="primary"
              type="button"
              disabled={!stats}
              onClick={async () => {
                const pub = await api.publishMatch(facilityId, matchId);
                const path = pub.report_path || (pub.share_token ? `/report/${pub.share_token}` : "");
                if (path) {
                  const url = `${window.location.origin}${path}`;
                  setMessage(`Published — share report: ${url}`);
                  try {
                    await navigator.clipboard.writeText(url);
                  } catch {
                    /* ignore clipboard failures */
                  }
                } else {
                  setMessage("Match published");
                }
                await refresh();
              }}
            >
              Publish & share report
            </button>
            {match?.status === "published" && (
              <button
                type="button"
                onClick={async () => {
                  const { revoked } = await api.revokeShare(facilityId, matchId);
                  setMessage(
                    revoked
                      ? "Share link revoked — the old URL no longer opens. Publish again for a new link."
                      : "No active share link",
                  );
                }}
              >
                Revoke share link
              </button>
            )}
          </div>
        </>
      )}

      {view === "setup" && (
        <>
          <section className="grid-2">
            <div className="panel">
              <h2>Roster</h2>
              <p className="lede">
                Optional. Processing auto-detects kit colour + jersey number; you can rename players later.
              </p>
              <ul className="list compact">
                {players.map((p) => (
                  <li key={p.id}>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={async () => {
                        await api.addRoster(facilityId, matchId, p.id, p.default_shirt_number ?? 0);
                        setMessage(`Added ${p.display_name} to roster`);
                      }}
                    >
                      Add #{p.default_shirt_number ?? "—"} {p.display_name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            <div className="panel">
              <h2>Camera upload</h2>
              <p className="lede">
                Camera A is the reference angle (wide view). Courts can run A only, or add B–D for
                more angles — goals and occlusions get more reliable with every extra camera.
              </p>
              {CAMERA_LABELS.map((label) => (
                <label className="file" key={label}>
                  Camera {label}
                  {label === "A" ? " (reference)" : ""}
                  <input
                    type="file"
                    accept="video/*"
                    disabled={busy}
                    onChange={(e) => onUpload(label, e.target.files?.[0] || null)}
                  />
                </label>
              ))}
              <ul className="list compact">
                {videos.map((v) => (
                  <li key={v.id}>
                    {v.kind} · {v.validation_status}
                    {v.width ? ` · ${v.width}×${v.height}` : ""}
            </li>
                ))}
              </ul>
            </div>
          </section>

          <CourtCalibrationPanel facilityId={facilityId} matchId={matchId} />

          <section className="panel">
            <h2>Camera sync</h2>
            <p className="lede">
              Offset of each camera onto Camera A's clock (ms): a moment at camera time t is at A time
              t + offset. Auto estimates with low confidence need a staff check.
            </p>
            {CAMERA_LABELS.filter((l) => l !== "A").map((label) => {
              const current = syncs.find((s) => s.camera_label === label);
              return (
                <form
                  key={label}
                  className="inline"
                  onSubmit={async (e) => {
                    e.preventDefault();
                    await api.updateSync(facilityId, matchId, label, offsetDraft[label] ?? 0, "Manual correction");
                    setMessage(`Camera ${label} offset saved`);
                    await refresh();
                  }}
                >
                  <span>Camera {label}</span>
                  <input
                    type="number"
                    value={offsetDraft[label] ?? 0}
                    onChange={(e) => setOffsetDraft({ ...offsetDraft, [label]: Number(e.target.value) })}
                  />
                  <button className="primary" type="submit">
                    Save offset (ms)
                  </button>
                  {current && (
                    <span className="lede">
                      {current.is_manual ? "manual" : `${current.method} · confidence ${(current.confidence ?? 0).toFixed(2)}`}
                    </span>
                  )}
                </form>
              );
            })}
          </section>

          <section className="panel">
            <h2>Processing</h2>
            <div className="inline">
              <button
                className="primary"
                type="button"
                disabled={busy}
                onClick={async () => {
                  setBusy(true);
                  try {
                    await api.startProcessing(facilityId, matchId);
                    setMessage("Processing queued");
                    setView("center");
                    await refresh();
                  } catch (err) {
                    setMessage(err instanceof Error ? err.message : "Failed to start");
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                Start processing
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={async () => {
                  await api.retryJob(facilityId, matchId);
                  setMessage("Retry queued");
                  setView("center");
                  await refresh();
                }}
              >
                Retry
              </button>
              {latestJob && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={async () => {
                    await api.cancelJob(facilityId, matchId, latestJob.id);
                    setMessage("Job cancelled");
                    await refresh();
                  }}
                >
                  Cancel latest
                </button>
              )}
            </div>
            <ul className="list compact">
              {jobs.map((j) => (
                <li key={j.id}>
                  {j.status} {j.stage ? `· ${j.stage}` : ""} · {j.progress_pct}%
                  {j.error_message ? ` · ${j.error_message}` : ""}
            </li>
              ))}
            </ul>
          </section>
        </>
      )}

      {view === "identity" && (
        <>
          <section className="panel">
            <h2>Rename players</h2>
            <p className="lede">Auto names come from kit colour + jersey number. Fix names for the court here.</p>
            <ul className="list compact">
              {players.map((p) => (
                <li key={p.id} className="inline" style={{ justifyContent: "space-between", gap: "0.75rem" }}>
                  <span>
                    {p.avatar_url ? (
                      <button
                        type="button"
                        className="bc-avatar-btn"
                        onClick={() => setPhotoModal({ src: p.avatar_url!, title: p.display_name })}
                        aria-label={`Open photo of ${p.display_name}`}
                      >
                        <img className="player-thumb" src={p.avatar_url} alt="" />
                      </button>
                    ) : null}
                    #{p.default_shirt_number ?? "—"} {p.display_name}
                  </span>
                  <form
                    className="inline"
                    onSubmit={async (e) => {
                      e.preventDefault();
                      const fd = new FormData(e.currentTarget);
                      const name = String(fd.get("name") || "").trim();
                      if (!name) return;
                      await api.renamePlayer(facilityId, p.id, name);
                      setMessage(`Renamed to ${name}`);
                      setPlayers(await api.listPlayers(facilityId));
                      setStats(await api.getStats(facilityId, matchId));
                    }}
                  >
                    <input name="name" placeholder="Correct name" defaultValue={p.display_name} required />
                    <button className="primary" type="submit">
                      Save
                    </button>
                  </form>
            </li>
              ))}
            </ul>
          </section>
          <IdentityBoard
            tracks={tracks}
            players={players}
            onMove={async (trackIds, playerId, note) => {
              await api.setTrackIdentities(facilityId, matchId, trackIds, playerId);
              setMessage(note);
              setTracks(await api.listTracks(facilityId, matchId));
              setStats(await api.getStats(facilityId, matchId));
            }}
          />
        </>
      )}

      {message && <p className="bc-toast">{message}</p>}
      {photoModal && (
        <PhotoLightbox
          src={photoModal.src}
          title={photoModal.title}
          onClose={() => setPhotoModal(null)}
        />
      )}
      {clipModal && (
        <ClipLightbox
          src={clipModal.src}
          title={clipModal.title}
          onClose={() => setClipModal(null)}
        />
      )}
    </div>
  );
}

function Phase0Page() {
  const [report, setReport] = useState<string>("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="brand">CourtLens</p>
          <h1>Phase 0 feasibility</h1>
          <p className="lede">Upload a short public match clip. Runs pretrained YOLO detect/track + throughput stats.</p>
        </div>
        <Link to="/">Back</Link>
      </header>
      <section className="panel">
        <input
          type="file"
          accept="video/*"
          disabled={busy}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setBusy(true);
            try {
              const data = await api.runFeasibility(file);
              setReport(JSON.stringify(data, null, 2));
            } catch (err) {
              setReport(err instanceof Error ? err.message : "Failed");
            } finally {
              setBusy(false);
            }
          }}
        />
        {busy && <p className="lede">Running detection… CPU may take a few minutes.</p>}
        {report && <pre style={{ whiteSpace: "pre-wrap", fontSize: "0.85rem" }}>{report}</pre>}
      </section>
    </div>
  );
}

function PublicReportPage() {
  const { shareToken = "" } = useParams();
  const [data, setData] = useState<Awaited<ReturnType<typeof api.getPublicReport>> | null>(null);
  const [error, setError] = useState("");
  const [photoModal, setPhotoModal] = useState<{ src: string; title: string } | null>(null);
  const [heatPlayerId, setHeatPlayerId] = useState("match");
  const [heatKind, setHeatKind] = useState<"presence" | "onball">("presence");

  useEffect(() => {
    api
      .getPublicReport(shareToken)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Report unavailable"));
  }, [shareToken]);

  if (error) {
    return (
      <div className="page broadcast">
        <p className="brand">CourtLens</p>
        <p className="error">{error}</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page broadcast">
        <p className="brand">CourtLens</p>
        <p className="lede">Loading match report…</p>
      </div>
    );
  }

  const board = [...data.by_player].sort(
    (a, b) => (b.possession_pct ?? 0) + b.pass - ((a.possession_pct ?? 0) + a.pass),
  );
  const selectedHeat =
    heatPlayerId === "match"
      ? heatKind === "onball"
        ? data.heatmap?.match_possession
        : data.heatmap?.match
      : heatKind === "onball"
        ? board.find((p) => (p.player_id || p.display_name) === heatPlayerId)?.possession_heatmap
        : board.find((p) => (p.player_id || p.display_name) === heatPlayerId)?.heatmap;
  const heatLabel =
    heatPlayerId === "match"
      ? heatKind === "onball"
        ? "Match on-ball heatmap"
        : "Match movement heatmap"
      : `${board.find((p) => (p.player_id || p.display_name) === heatPlayerId)?.display_name || "Player"} ${
          heatKind === "onball" ? "on the ball" : "movement"
        }`;

  return (
    <div className="page broadcast">
      <header className="bc-nav">
        <p className="brand">CourtLens</p>
        <div className="bc-nav-meta">{data.facility_name}</div>
      </header>
      <section className="bc-scoreboard">
        <div className="bc-scoreboard-top">
          <div>
            <p className="bc-kicker">Match report</p>
            <h1 className="bc-match-title">{data.title}</h1>
          </div>
          <div className="bc-status is-done">
            <span className="bc-status-dot" />
            {data.status.replaceAll("_", " ")}
          </div>
        </div>
        <div className="bc-stat-strip">
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.goal ?? 0}</div>
            <div className="bc-stat-label">Goals</div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.shot ?? 0}</div>
            <div className="bc-stat-label">Shots <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.pass ?? 0}</div>
            <div className="bc-stat-label">Passes <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.assist ?? 0}</div>
            <div className="bc-stat-label">Assists <em className="bc-est">est.</em></div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.dribble ?? 0}</div>
            <div className="bc-stat-label">Dribbles <em className="bc-est">est.</em></div>
          </div>
        </div>
        <div className="bc-confidence">
          <span>For coaches & players</span>
        </div>
      </section>

      <div className="bc-layout">
        <TeamStatsPanel teams={data.by_team ?? []} names={{}} />
        <section className="bc-board">
          <div className="bc-board-head">
                <h2>Squad board</h2>
                <span>POS% · Min · Touches · Move · G · S · A · D · P</span>
          </div>
          <table className="bc-table">
            <thead>
              <tr>
                <th>Player</th>
                <th className="num" title="Possession share">POS%</th>
                <th className="num" title="On-camera minutes">Min</th>
                <th className="num" title="Ball touches">Touches</th>
                <th
                  className="num"
                  title="Metres run when the camera is court-calibrated (hover for km/h); otherwise relative movement in the camera view (% of frame diagonal)."
                >
                  Move
                </th>
                <th className="num" title="Goals">G</th>
                <th className="num" title="Shots">S</th>
                <th className="num" title="Assists">A</th>
                <th className="num" title="Dribbles">D</th>
                <th className="num" title="Passes">P</th>
                <th className="num" title="Passes received">Rec</th>
                <th className="num" title="Pass accuracy: completed ÷ (completed + intercepted)">P%</th>
                <th className="num" title="Shots on target">SoT</th>
                <th className="num" title="Interceptions (ball won)">Int</th>
                <th className="num" title="Balls lost to the other team">Lost</th>
                <th className="num" title="Keeper saves">Sv</th>
                <th className="num" title="Fouls committed">F</th>
                <th className="num" title="Yellow / red cards">Cards</th>
              </tr>
            </thead>
            <tbody>
              {board.map((row, idx) => (
                <tr key={`${row.display_name}-${idx}`}>
                  <td>
                    <div className="bc-player">
                      {row.avatar_url ? (
                        <button
                          type="button"
                          className="bc-avatar-btn"
                          onClick={() =>
                            setPhotoModal({ src: row.avatar_url!, title: row.display_name })
                          }
                          aria-label={`Open photo of ${row.display_name}`}
                        >
                          <img className="bc-avatar" src={row.avatar_url} alt="" />
                        </button>
                      ) : (
                        <span className="bc-avatar-fallback">#{row.shirt_number ?? idx + 1}</span>
                      )}
                      <span className="bc-shirt">{row.shirt_number ?? idx + 1}</span>
                      <div>
                        <span className="bc-player-name">{row.display_name}</span>
                        <span className="bc-player-sub">{row.attribution}</span>
                      </div>
                    </div>
                  </td>
                  <td className="num">{(row.possession_pct ?? 0).toFixed(0)}</td>
                  <td className="num">{(row.minutes ?? 0).toFixed(1)}</td>
                  <td className="num">{row.touches ?? 0}</td>
                  {moveCell(row)}
                  <td className="num">{row.goal}</td>
                  <td className="num">{row.shot}</td>
                  <td className="num">{row.assist ?? 0}</td>
                  <td className="num">{row.dribble ?? 0}</td>
                  <td className="num">{row.pass}</td>
                  <td className="num">{row.pass_received ?? 0}</td>
                  <td className="num">{pctOr(row.pass_accuracy_pct)}</td>
                  <td className="num">{row.shot_on_target ?? 0}</td>
                  <td className="num">{row.interception ?? 0}</td>
                  <td className="num">{row.ball_lost ?? 0}</td>
                  <td className="num">{row.save ?? 0}</td>
                  <td className="num">{row.foul ?? 0}</td>
                  <td className="num">
                    {(row.yellow_card ?? 0) + (row.red_card ?? 0) > 0 ? `${row.yellow_card ?? 0}Y ${row.red_card ?? 0}R` : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="bc-board bc-heat-board">
          <div className="bc-board-head">
            <h2>Heatmap</h2>
            <label className="bc-heat-select">
              Player
              <select value={heatPlayerId} onChange={(e) => setHeatPlayerId(e.target.value)}>
                <option value="match">Whole match</option>
                {board.map((p, idx) => (
                  <option key={p.player_id || `${p.display_name}-${idx}`} value={p.player_id || p.display_name}>
                    {p.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label className="bc-heat-select">
              Map
              <select value={heatKind} onChange={(e) => setHeatKind(e.target.value as "presence" | "onball")}>
                <option value="presence">Movement</option>
                <option value="onball">On the ball</option>
              </select>
            </label>
          </div>
          <PitchHeatmap grid={selectedHeat} label={heatLabel} />
          {data.heatmap?.note && <p className="bc-table-note">{data.heatmap.note}</p>}
        </section>
      </div>
      {photoModal && (
        <PhotoLightbox
          src={photoModal.src}
          title={photoModal.title}
          onClose={() => setPhotoModal(null)}
        />
      )}
    </div>
  );
}

function AppShell() {
  const { ready, authed, setAuthed } = useAuth();
  const navigate = useNavigate();
  if (!ready) return <div className="page">Loading…</div>;

  return (
    <Routes>
      <Route path="/report/:shareToken" element={<PublicReportPage />} />
      <Route
        path="/"
        element={
          authed ? (
            <FacilitiesPage />
          ) : (
            <AuthPage
              onAuthed={() => {
                setAuthed(true);
                navigate("/");
              }}
            />
          )
        }
      />
      <Route path="/phase0" element={authed ? <Phase0Page /> : <Navigate to="/" />} />
      <Route path="/facilities/:facilityId" element={authed ? <FacilityHome /> : <Navigate to="/" />} />
      <Route path="/facilities/:facilityId/matches/:matchId" element={authed ? <MatchPage /> : <Navigate to="/" />} />
      <Route path="/facilities/:facilityId/matches/:matchId/label" element={authed ? <LabelPage /> : <Navigate to="/" />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
