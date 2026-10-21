import { FormEvent, useEffect, useState } from "react";
import { BrowserRouter, Link, Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import {
  api,
  type Facility,
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

  async function refresh() {
    const [t, p, m] = await Promise.all([
      api.listTeams(facilityId),
      api.listPlayers(facilityId),
      api.listMatches(facilityId),
    ]);
    setTeams(t);
    setPlayers(p);
    setMatches(m);
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

function MatchPage() {
  const { facilityId = "", matchId = "" } = useParams();
  const [match, setMatch] = useState<Match | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [videos, setVideos] = useState<VideoAsset[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [offsetMs, setOffsetMs] = useState(0);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<"center" | "setup" | "identity">("center");
  const [heatPlayerId, setHeatPlayerId] = useState<string>("match");
  const [photoModal, setPhotoModal] = useState<{ src: string; title: string } | null>(null);
  const [clipModal, setClipModal] = useState<{ src: string; title: string } | null>(null);
  const [clipBusyId, setClipBusyId] = useState<string | null>(null);

  async function refresh() {
    const [m, p, v, j, sync] = await Promise.all([
      api.getMatch(facilityId, matchId),
      api.listPlayers(facilityId),
      api.listVideos(facilityId, matchId),
      api.listJobs(facilityId, matchId),
      api.getSync(facilityId, matchId).catch(() => null),
    ]);
    setMatch(m);
    setPlayers(p);
    setVideos(v);
    setJobs(j);
    if (sync) setOffsetMs(sync.offset_ms);
    const [tr, ev, st] = await Promise.all([
      api.listTracks(facilityId, matchId).catch(() => [] as Track[]),
      api.listEvents(facilityId, matchId).catch(() => [] as MatchEvent[]),
      api.getStats(facilityId, matchId).catch(() => null),
    ]);
    setTracks(tr);
    setEvents(ev);
    setStats(st);
  }

  useEffect(() => {
    refresh().catch(console.error);
    const id = window.setInterval(() => {
      api.listJobs(facilityId, matchId).then(setJobs).catch(() => undefined);
    }, 4000);
    return () => clearInterval(id);
  }, [facilityId, matchId]);

  async function onUpload(label: "A" | "B", file: File | null) {
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
      ? stats?.heatmap?.match
      : leaderboard.find((p) => p.player_id === heatPlayerId)?.heatmap;
  const heatLabel =
    heatPlayerId === "match"
      ? "Match heatmap (all players)"
      : `${leaderboard.find((p) => p.player_id === heatPlayerId)?.display_name || "Player"} movement`;

  const eventPlayer = (ev: MatchEvent) => {
    const primary =
      ev.participants.find((p) => ["passer", "shooter", "scorer", "primary"].includes(p.role)) ||
      ev.participants[0];
    return playerName(primary?.player_id ?? null);
  };

  return (
    <div className="page broadcast">
      <header className="bc-nav">
        <p className="brand">CourtLens</p>
        <div className="bc-nav-meta">
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
            <div className="bc-stat-label">Shots</div>
          </div>
          <div className="bc-stat is-pass">
            <div className="bc-stat-value">{passes}</div>
            <div className="bc-stat-label">Passes</div>
          </div>
        </div>

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
          <div className="bc-layout">
            <section className="bc-board">
              <div className="bc-board-head">
                <h2>Squad board</h2>
                <span>POS% · Touches · Move · G · S · P</span>
              </div>
              {leaderboard.length ? (
                <>
                  <table className="bc-table">
                    <thead>
                      <tr>
                        <th>Player</th>
                        <th className="num" title="Possession share">POS%</th>
                        <th className="num" title="Ball touches">Touches</th>
                        <th
                          className="num"
                          title="Relative movement in the camera view — not metres. Path length as % of frame diagonal."
                        >
                          Move
                        </th>
                        <th className="num" title="Goals">G</th>
                        <th className="num" title="Shots">S</th>
                        <th className="num" title="Passes">P</th>
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
                            <td className="num">{row.touches ?? 0}</td>
                            <td className="num">{Math.round(row.distance ?? 0)}</td>
                            <td className="num">{row.goal}</td>
                            <td className="num">{row.shot}</td>
                            <td className="num">{row.pass}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                              <p className="bc-table-note">
                    POS% possession · Touches = times on the ball · Move = relative distance in camera view
                    (not metres). Tap a player for their pitch map; tap photo to enlarge.
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
                Pilot courts can start with Camera A only. Add Camera B when you have a second angle.
              </p>
              <label className="file">
                Camera A
                <input
                  type="file"
                  accept="video/*"
                  disabled={busy}
                  onChange={(e) => onUpload("A", e.target.files?.[0] || null)}
                />
              </label>
              <label className="file">
                Camera B
                <input
                  type="file"
                  accept="video/*"
                  disabled={busy}
                  onChange={(e) => onUpload("B", e.target.files?.[0] || null)}
                />
              </label>
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

          <section className="panel">
            <h2>Camera sync</h2>
            <form
              className="inline"
              onSubmit={async (e) => {
                e.preventDefault();
                await api.updateSync(facilityId, matchId, offsetMs, "Manual correction");
                setMessage("Sync offset saved");
              }}
            >
              <input type="number" value={offsetMs} onChange={(e) => setOffsetMs(Number(e.target.value))} />
              <button className="primary" type="submit">
                Save offset (ms)
              </button>
            </form>
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
          <section className="panel">
            <h2>Tracks & identity</h2>
            <p className="lede">Reassign a camera track to another player when auto-ID is wrong.</p>
            <ul className="list compact">
              {tracks
                .filter((t) => t.class_name !== "sports_ball")
                .slice(0, 40)
                .map((t) => (
                  <li key={t.id} className="inline" style={{ justifyContent: "space-between" }}>
                    <span>
                      {t.track_key} · cam {t.camera_label} · {t.identity?.state || "none"} ·{" "}
                      {playerName(t.identity?.player_id || null)}
                    </span>
                    <select
                      defaultValue={t.identity?.player_id || ""}
                      onChange={async (e) => {
                        const val = e.target.value || null;
                        await api.setTrackIdentity(facilityId, matchId, t.id, val);
                        setMessage("Identity updated");
                        setTracks(await api.listTracks(facilityId, matchId));
                        setStats(await api.getStats(facilityId, matchId));
                      }}
                    >
                      <option value="">Unknown</option>
                      {players.map((p) => (
                        <option key={p.id} value={p.id}>
                          #{p.default_shirt_number ?? "—"} {p.display_name}
                        </option>
                      ))}
                    </select>
            </li>
                ))}
          </ul>
            {!tracks.length && <p className="bc-empty">No tracks yet — process a match first.</p>}
          </section>
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
      ? data.heatmap?.match
      : board.find((p) => (p.player_id || p.display_name) === heatPlayerId)?.heatmap;
  const heatLabel =
    heatPlayerId === "match"
      ? "Match movement heatmap"
      : `${board.find((p) => (p.player_id || p.display_name) === heatPlayerId)?.display_name || "Player"} movement`;

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
            <div className="bc-stat-label">Shots</div>
          </div>
          <div className="bc-stat">
            <div className="bc-stat-value">{data.totals.pass ?? 0}</div>
            <div className="bc-stat-label">Passes</div>
          </div>
        </div>
        <div className="bc-confidence">
          <span>For coaches & players</span>
        </div>
      </section>

      <div className="bc-layout">
        <section className="bc-board">
          <div className="bc-board-head">
            <h2>Squad board</h2>
            <span>POS% · Touches · Move · G · S · P</span>
          </div>
          <table className="bc-table">
            <thead>
              <tr>
                <th>Player</th>
                <th className="num" title="Possession share">POS%</th>
                <th className="num" title="Ball touches">Touches</th>
                <th
                  className="num"
                  title="Relative movement in the camera view — not metres. Path length as % of frame diagonal."
                >
                  Move
                </th>
                <th className="num" title="Goals">G</th>
                <th className="num" title="Shots">S</th>
                <th className="num" title="Passes">P</th>
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
                  <td className="num">{row.touches ?? 0}</td>
                  <td className="num">{Math.round(row.distance ?? 0)}</td>
                  <td className="num">{row.goal}</td>
                  <td className="num">{row.shot}</td>
                  <td className="num">{row.pass}</td>
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
