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
        <h1>Facility match analytics</h1>
        <p className="lede">Upload two-camera futsal recordings. Review uncertain events. Publish player reports.</p>
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
          <h1>Facilities</h1>
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
          <h1>Facility workspace</h1>
        </div>
        <Link to="/">All facilities</Link>
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

function HeatmapGrid({
  grid,
  label,
}: {
  grid: number[][] | undefined;
  label: string;
}) {
  if (!grid?.length) {
    return <p className="bc-empty">Heatmap unavailable for this selection.</p>;
  }
  const w = grid[0]?.length || 1;
  return (
    <div className="bc-heat-wrap">
      <div
        className="bc-heat"
        style={{ gridTemplateColumns: `repeat(${w}, minmax(0, 1fr))` }}
        role="img"
        aria-label={label}
      >
        {grid.flatMap((row, y) =>
          row.map((v, x) => (
            <span
              key={`${y}-${x}`}
              className="bc-heat-cell"
              style={{ opacity: 0.08 + v * 0.92 }}
              title={`${Math.round(v * 100)}%`}
            />
          )),
        )}
      </div>
      <p className="bc-heat-caption">{label}</p>
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
            <p className="bc-progress-label">Broadcast stats update when analysis finishes.</p>
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
                <h2>Player stats</h2>
                <span>POS% · TCH · DIST · G · S · P</span>
              </div>
              {leaderboard.length ? (
                <>
                  <table className="bc-table">
                    <thead>
                      <tr>
                        <th>Player</th>
                        <th className="num">POS%</th>
                        <th className="num">TCH</th>
                        <th className="num">DIST</th>
                        <th className="num">G</th>
                        <th className="num">S</th>
                        <th className="num">P</th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboard.map((row, idx) => {
                        const rosterPlayer = players.find((p) => p.id === row.player_id);
                        const shirt = rosterPlayer?.default_shirt_number ?? idx + 1;
                        return (
                          <tr
                            key={row.player_id}
                            className={heatPlayerId === row.player_id ? "is-selected" : ""}
                            onClick={() => setHeatPlayerId(row.player_id)}
                          >
                            <td>
                              <div className="bc-player">
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
                    POS% possession share · TCH ball touches · DIST relative movement (not metres). Click a
                    player for their heatmap.
                  </p>
                </>
              ) : (
                <p className="bc-empty">No player stats yet. Run analysis from Setup & process.</p>
              )}
            </section>

            <section className="bc-board">
              <div className="bc-board-head">
                <h2>Key moments</h2>
                <span>{timeline.length} events</span>
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
                            onClick={async () => {
                              const clip = await api.createClip(facilityId, matchId, ev.id);
                              setMessage(`Clip ready — open from notice below`);
                              window.open(clip.playback_url, "_blank", "noopener,noreferrer");
                            }}
                          >
                            Watch clip
                          </button>
                        </div>
                      </div>
                    </article>
                  ))
                ) : (
                  <p className="bc-empty">Key moments appear here after processing — like a broadcast timeline.</p>
                )}
              </div>
            </section>
          </div>

          <section className="bc-board bc-heat-board">
            <div className="bc-board-head">
              <h2>Heatmap</h2>
              <label className="bc-heat-select">
                View
                <select value={heatPlayerId} onChange={(e) => setHeatPlayerId(e.target.value)}>
                  <option value="match">Full match</option>
                  {leaderboard.map((p) => (
                    <option key={p.player_id} value={p.player_id}>
                      {p.display_name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <HeatmapGrid grid={selectedHeat} label={heatLabel} />
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
          <span>Estimated analytics for coaches & players</span>
        </div>
      </section>

      <div className="bc-layout">
        <section className="bc-board">
          <div className="bc-board-head">
            <h2>Player board</h2>
            <span>POS% · TCH · DIST · G · S · P</span>
          </div>
          <table className="bc-table">
            <thead>
              <tr>
                <th>Player</th>
                <th className="num">POS%</th>
                <th className="num">TCH</th>
                <th className="num">DIST</th>
                <th className="num">G</th>
                <th className="num">S</th>
                <th className="num">P</th>
              </tr>
            </thead>
            <tbody>
              {board.map((row, idx) => (
                <tr key={`${row.display_name}-${idx}`}>
                  <td>
                    <div className="bc-player">
                      <span className="bc-shirt">{idx + 1}</span>
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
        <section className="bc-board">
          <div className="bc-board-head">
            <h2>Heatmap</h2>
            <span>Full match</span>
          </div>
          <HeatmapGrid grid={data.heatmap?.match} label="Match movement heatmap" />
          {data.heatmap?.note && <p className="bc-table-note">{data.heatmap.note}</p>}
        </section>
      </div>
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
