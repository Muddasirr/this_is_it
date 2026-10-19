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

  async function refresh() {
    const [m, p, v, j, sync] = await Promise.all([
      api.getMatch(facilityId, matchId),
      api.listPlayers(facilityId),
      api.listVideos(facilityId, matchId),
      api.listJobs(facilityId, matchId),
      api.getSync(facilityId, matchId),
    ]);
    setMatch(m);
    setPlayers(p);
    setVideos(v);
    setJobs(j);
    if (sync) setOffsetMs(sync.offset_ms);
    try {
      setTracks(await api.listTracks(facilityId, matchId));
      setEvents(await api.listEvents(facilityId, matchId));
      setStats(await api.getStats(facilityId, matchId));
    } catch {
      /* empty until processed */
    }
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

  const playerName = (id: string | null) =>
    id ? players.find((p) => p.id === id)?.display_name || id.slice(0, 8) : "unassigned";

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="brand">CourtLens</p>
          <h1>{match?.title || "Match"}</h1>
          <p className="lede">Status: {match?.status}</p>
        </div>
        <Link to={`/facilities/${facilityId}`}>Back</Link>
      </header>

      <section className="grid-2">
        <div className="panel">
          <h2>Roster</h2>
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
          <p className="lede">Phase 0/1: one camera is enough for smoke tests; production needs A+B.</p>
          <label className="file">
            Camera A
            <input type="file" accept="video/*" disabled={busy} onChange={(e) => onUpload("A", e.target.files?.[0] || null)} />
          </label>
          <label className="file">
            Camera B
            <input type="file" accept="video/*" disabled={busy} onChange={(e) => onUpload("B", e.target.files?.[0] || null)} />
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
              await refresh();
            }}
          >
            Retry
          </button>
          {jobs[0] && (
            <button
              type="button"
              disabled={busy}
              onClick={async () => {
                await api.cancelJob(facilityId, matchId, jobs[0].id);
                setMessage("Job cancelled");
                await refresh();
              }}
            >
              Cancel latest
            </button>
          )}
          <button type="button" onClick={() => refresh()}>
            Refresh results
          </button>
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

      <section className="panel">
        <h2>Tracks & identity</h2>
        <p className="lede">Map camera-scoped tracks to roster players. Confirmed links override model proposals.</p>
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
      </section>

      <section className="panel">
        <h2>Event review</h2>
        <p className="lede">Accept / reject / edit. Labels: estimated · confirmed · rejected.</p>
        <ul className="list">
          {events.map((ev) => (
            <li key={ev.id}>
              <strong>
                {ev.event_type}
              </strong>{" "}
              @ {(ev.canonical_ts_ms / 1000).toFixed(1)}s · <span className="badge">{ev.confirmation}</span> · conf{" "}
              {ev.confidence?.toFixed(2) ?? "—"}
              <div className="inline" style={{ marginTop: "0.5rem" }}>
                <button
                  type="button"
                  className="primary"
                  onClick={async () => {
                    await api.reviewEvent(facilityId, matchId, ev.id, "accept");
                    setEvents(await api.listEvents(facilityId, matchId));
                    setStats(await api.getStats(facilityId, matchId));
                  }}
                >
                  Accept
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    await api.reviewEvent(facilityId, matchId, ev.id, "reject");
                    setEvents(await api.listEvents(facilityId, matchId));
                    setStats(await api.getStats(facilityId, matchId));
                  }}
                >
                  Reject
                </button>
                <select
                  onChange={async (e) => {
                    if (!e.target.value) return;
                    await api.reviewEvent(facilityId, matchId, ev.id, "edit", { player_id: e.target.value });
                    setEvents(await api.listEvents(facilityId, matchId));
                    setStats(await api.getStats(facilityId, matchId));
                  }}
                  defaultValue=""
                >
                  <option value="">Assign player…</option>
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
                    setMessage(`Clip ready: ${clip.playback_url}`);
                  }}
                >
                  Evidence clip
                </button>
              </div>
            </li>
          ))}
          {!events.length && <li>No events yet — run processing first.</li>}
        </ul>
      </section>

      <section className="panel">
        <h2>Match dashboard</h2>
        {stats ? (
          <>
            <p className="lede">
              Goals {stats.totals.goal ?? 0} · Shots {stats.totals.shot ?? 0} · Passes {stats.totals.pass ?? 0} · Confirmed{" "}
              {stats.totals.confirmed ?? 0} · Estimated {stats.totals.estimated ?? 0}
            </p>
            <ul className="list compact">
              {stats.by_player.map((row) => (
                <li key={row.player_id}>
                  {row.display_name}: G{row.goal} S{row.shot} P{row.pass}{" "}
                  <span className="badge">{row.attribution}</span>
                </li>
              ))}
            </ul>
            <button
              className="primary"
              type="button"
              onClick={async () => {
                await api.publishMatch(facilityId, matchId);
                setMessage("Match published");
                await refresh();
              }}
            >
              Publish results
            </button>
          </>
        ) : (
          <p className="lede">Stats appear after processing.</p>
        )}
        {message && <p className="notice">{message}</p>}
      </section>
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

function AppShell() {
  const { ready, authed, setAuthed } = useAuth();
  const navigate = useNavigate();
  if (!ready) return <div className="page">Loading…</div>;

  return (
    <Routes>
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
