import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { go } from '../App'
import { api, fmtTime, invert3, playerTag, toImage } from '../api'
import { Bars, Heatmap, ZoneSplit } from '../components/Charts'
import Court, { teamColor } from '../components/Court'
import type { Match, Player, Report, Shot } from '../types'

type Tab = 'players' | 'pairs' | 'rallies' | 'shots' | 'quality'

export default function ReportPage({ id }: { id: string }) {
  const [match, setMatch] = useState<Match | null>(null)
  const [rep, setRep] = useState<Report | null>(null)
  const [err, setErr] = useState('')
  const [tab, setTab] = useState<Tab>('players')
  const [now, setNow] = useState(0)
  const video = useRef<HTMLVideoElement>(null)

  const load = useCallback(() => {
    api.match(id).then(setMatch).catch((e) => setErr(e.message))
    api.report(id).then(setRep).catch((e) => setErr(e.message))
  }, [id])
  useEffect(load, [load])

  const seek = (t: number) => {
    if (video.current) {
      video.current.currentTime = Math.max(0, t)
      video.current.play().catch(() => undefined)
    }
  }
  const review = async (body: Record<string, Record<string, unknown>>) => {
    try {
      await api.review(id, body)
      setRep(await api.report(id))
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  if (err && !rep) return <div className="card err">{err}</div>
  if (!rep || !match) return <div className="muted">Loading…</div>
  const names = rep.players.map((p) => p.name)
  const idx = Math.min(rep.replay.players.length - 1, Math.max(0, Math.round(now * rep.replay.hz)))
  const m = rep.match

  return (
    <div className="stack">
      <div className="row">
        <div>
          <h1>{match.title}</h1>
          <div className="secondary small">
            {fmtTime(m.duration_s)} analysed · {m.rallies} rallies · all shot and stroke values are model estimates until confirmed
          </div>
        </div>
        <div className="spacer" />
        <div className="row small" style={{ gap: 6 }}>
          <span className="muted">Export</span>
          {['shots', 'players', 'pairs', 'rallies'].map((t) => (
            <a key={t} className="btn" href={api.exportUrl(id, t)} download>{t}.csv</a>
          ))}
          <a className="btn" href={api.reportUrl(id)} download={`padel_${id}_report.json`}>json</a>
        </div>
        <button onClick={() => go('/')}>All matches</button>
      </div>

      <div className="grid4">
        {m.score && m.points_decided ? (
          <Tile label={`Score${m.outcome_label === 'confirmed' ? '' : ' (est.)'}`}
            value={m.score.sets.map((g: number[]) => g.join('-')).join('  ')}
            sub={`${m.score.current_game ? `game ${m.score.current_game} · ` : ''}${m.points_decided} points`} />
        ) : null}
        <Tile label="Rallies" value={m.rallies} sub={m.rallies_confirmed ? `${m.rallies_confirmed} with winner set` : undefined} />
        <Tile label="Shots" value={m.hits} />
        <Tile label="Avg rally" value={m.avg_rally_shots ?? '–'} sub={m.avg_rally_s ? `shots · ${m.avg_rally_s} s` : 'shots'} />
        <Tile label="Longest rally" value={m.max_rally_shots ?? '–'} sub="shots" />
        <Tile label="Ball in play" value={m.time_in_play_pct != null ? `${m.time_in_play_pct}%` : '–'}
          sub={m.time_in_play_s != null ? `${fmtTime(m.time_in_play_s)} · ${m.shots_per_min_in_play ?? '–'} shots/min` : undefined} />
        {rep.teams.map((t) => (
          <Tile key={t.id} label={t.name} value={`${Math.round(Number(t.stats.distance_m ?? 0))} m`}
            sub={`${t.stats.shots} shots · ${t.stats.both_at_net_pct ?? 0}% both at net`} color={teamColor(t.id)} />
        ))}
      </div>

      <div className="player-layout">
        <div className="card stack">
          <VideoOverlay ref={video} id={id} rep={rep} idx={idx} onTime={setNow} />
          <Timeline rep={rep} now={now} seek={seek} />
        </div>
        <div className="card">
          <Court players={rep.replay.players[idx]} names={names} height={440} />
          <div className="row small" style={{ justifyContent: 'center', marginTop: 8 }}>
            {rep.teams.map((t) => (
              <span key={t.id} className="team-chip"><i style={{ background: teamColor(t.id) }} />{t.name}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="tabs" role="tablist">
        {(['players', 'pairs', 'rallies', 'shots', 'quality'] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)} role="tab">{t[0].toUpperCase() + t.slice(1)}</button>
        ))}
      </div>
      {err && <div className="err">{err}</div>}

      {tab === 'players' && (
        <div className="grid2">
          {rep.players.map((p) => <PlayerCard key={p.id} p={p} onRename={(name) => review({ players: { [p.id]: { name } } })} />)}
        </div>
      )}
      {tab === 'pairs' && (
        <div className="grid2">
          {rep.teams.map((t) => (
            <div className="card stack" key={t.id}>
              <div className="row">
                <span className="team-chip"><i style={{ background: teamColor(t.id) }} /></span>
                <input defaultValue={t.name} onBlur={(e) => e.target.value !== t.name && review({ teams: { [t.id]: { name: e.target.value } } })} />
                <span className="secondary small">{t.players.map((i) => names[i]).join(' & ')}</span>
              </div>
              <table>
                <tbody>
                  {[
                    ['Both at the net', `${t.stats.both_at_net_pct ?? '–'} %`],
                    ['Both back', `${t.stats.both_back_pct ?? '–'} %`],
                    ['Avg partner distance', `${t.stats.avg_partner_distance_m ?? '–'} m`],
                    ['Avg lateral spacing', `${t.stats.avg_lateral_spacing_m ?? '–'} m`],
                    ['Avg depth offset', `${t.stats.avg_depth_offset_m ?? '–'} m`],
                    ['Distance (pair)', `${t.stats.distance_m ?? '–'} m`],
                    ['Shots', t.stats.shots],
                    ['Hit the last shot of a rally', t.stats.last_shot_in_rallies],
                    ['Points won (est. unless confirmed)', t.stats.points_won != null ? `${t.stats.points_won} (${t.stats.points_won_pct ?? '–'} %)` : '–'],
                    ['Games won · held · breaks', `${t.stats.games_won ?? 0} · ${t.stats.service_games_held ?? 0} · ${t.stats.breaks ?? 0}`],
                    ['Serve / return points won', `${t.stats.serve_points_won_pct ?? '–'} % / ${t.stats.return_points_won_pct ?? '–'} %`],
                    ['Winners / errors', `${t.stats.winners ?? 0} / ${t.stats.errors ?? 0}`],
                    ...pairExtra(t.stats),
                  ].map(([k, v]) => (
                    <tr key={String(k)}><td className="secondary">{k as React.ReactNode}</td><td className="num" style={{ textAlign: "right" }}>{v as React.ReactNode}</td></tr>
                  ))}
                </tbody>
              </table>
              <h3>Where they hit from</h3>
              <Court height={300} marks={rep.shots.filter((s) => s.team === t.id && s.pos_m && !s.deleted).map((s) => ({
                p: s.pos_m!, team: s.team, title: `${fmtTime(s.t)} ${names[s.player]} · ${s.type}`, onClick: () => seek(s.t - 1),
              }))} />
            </div>
          ))}
        </div>
      )}
      {tab === 'rallies' && <Rallies rep={rep} names={names} seek={seek} review={review} />}
      {tab === 'shots' && <Shots rep={rep} names={names} seek={seek} review={review} now={now} />}
      {tab === 'quality' && (
        <div className="card">
          <table>
            <tbody>
              {Object.entries(rep.quality).map(([k, v]) => (
                <tr key={k}><td className="secondary">{k.replaceAll('_', ' ')}</td><td className="num">{String(v)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="small muted">
            Positions, distances and speeds are measured from the calibrated court. Hits, stroke types, volleys and shot speeds are
            model estimates; edits in the Shots and Rallies tabs are stored as confirmed and override the model.
          </p>
        </div>
      )}
    </div>
  )
}

function Tile({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="card tile">
      <div className="label">{color && <i style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: color, marginRight: 6 }} />}{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

const VideoOverlay = ({ ref, id, rep, idx, onTime }: {
  ref: React.RefObject<HTMLVideoElement | null>; id: string; rep: Report; idx: number; onTime: (t: number) => void
}) => {
  const canvas = useRef<HTMLCanvasElement>(null)
  const Hinv = useMemo(() => invert3(rep.calibration.H), [rep])
  useEffect(() => {
    let raf = 0
    const tick = () => {
      if (ref.current) onTime(ref.current.currentTime)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [ref, onTime])
  const draw = useCallback(() => {
    const c = canvas.current
    const v = ref.current
    if (!c || !v) return
    const [W, H] = rep.match.image_size
    // size from the rendered video every time: the element grows once metadata loads
    const dpr = devicePixelRatio
    c.width = v.clientWidth * dpr
    c.height = v.clientHeight * dpr
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    const sx = c.width / W
    const sy = c.height / H
    const style = getComputedStyle(document.documentElement)
    ctx.font = `600 ${13 * dpr}px system-ui`
    ctx.textAlign = 'center'
    rep.replay.players[idx]?.forEach((p, i) => {
      if (!p) return
      const [x, y] = toImage(rep.calibration, Hinv, p)
      const tag = playerTag(i)
      const w = ctx.measureText(tag).width + 10 * dpr
      ctx.fillStyle = style.getPropertyValue(i < 2 ? '--team-a' : '--team-b')
      ctx.beginPath()
      ctx.roundRect(x * sx - w / 2, y * sy + 4 * dpr, w, 18 * dpr, 4 * dpr)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.fillText(tag, x * sx, y * sy + 17 * dpr)
    })
    const b = rep.replay.ball_px[idx]
    if (b) {
      ctx.strokeStyle = '#ffeb3b'
      ctx.lineWidth = 2 * dpr
      ctx.beginPath()
      ctx.arc(b[0] * sx, b[1] * sy, 9 * dpr, 0, Math.PI * 2)
      ctx.stroke()
    }
  }, [idx, rep, Hinv, ref])
  useEffect(draw, [draw])
  useEffect(() => {
    const v = ref.current
    if (!v) return
    const ro = new ResizeObserver(() => draw())
    ro.observe(v)
    v.addEventListener('loadedmetadata', draw)
    return () => {
      ro.disconnect()
      v.removeEventListener('loadedmetadata', draw)
    }
  }, [draw, ref])
  return (
    <div className="video-box">
      <video ref={ref} src={api.videoUrl(id)} controls preload="metadata" />
      <canvas ref={canvas} />
    </div>
  )
}

function Timeline({ rep, now, seek }: { rep: Report; now: number; seek: (t: number) => void }) {
  const D = rep.match.duration_s || 1
  return (
    <div>
      <div className="timeline" onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        seek(((e.clientX - r.left) / r.width) * D)
      }}>
        {rep.rallies.map((r) => (
          <div key={r.id} className={`r ${r.winner_team === 0 ? 'won-a' : r.winner_team === 1 ? 'won-b' : ''}`}
            style={{ left: `${(r.start_t / D) * 100}%`, width: `${Math.max(0.3, ((r.end_t - r.start_t) / D) * 100)}%` }}
            title={`Rally ${r.id + 1} · ${r.n_shots} shots · ${fmtTime(r.start_t)}`}
            onClick={(e) => {
              e.stopPropagation()
              seek(r.start_t)
            }} />
        ))}
        <div className="cursor" style={{ left: `${(now / D) * 100}%` }} />
      </div>
      <div className="row small muted" style={{ justifyContent: 'space-between' }}>
        <span>Rallies (click to jump; coloured once the winner is set)</span>
        <span className="num">{fmtTime(now)} / {fmtTime(D)}</span>
      </div>
    </div>
  )
}

function PlayerCard({ p, onRename }: { p: Player; onRename: (name: string) => void }) {
  const s = p.stats
  return (
    <div className="card stack">
      <div className="row">
        <span className="team-chip"><i style={{ background: teamColor(p.team) }} /><b>{playerTag(p.id)}</b></span>
        <input defaultValue={p.name} onBlur={(e) => e.target.value !== p.name && onRename(e.target.value)} aria-label="Player name" />
      </div>
      <div className="grid4">
        <Tile label="Distance" value={`${Math.round(s.distance_m)} m`} sub={`${s.time_tracked_s >= 60 ? `${Math.round(s.time_tracked_s / 60)} min` : `${Math.round(s.time_tracked_s)} s`} tracked`} />
        <Tile label="Avg / max speed" value={`${s.avg_speed_kmh ?? '–'}`} sub={`max ${s.max_speed_kmh ?? '–'} km/h`} />
        <Tile label="Sprints" value={s.sprints} sub="> 4 m/s for 0.5 s" />
        <Tile label="Shots" value={s.shots} sub={`${s.volleys} volleys · ${s.overheads} overheads`} />
      </div>
      <div>
        <h3 style={{ marginBottom: 6 }}>Court position</h3>
        <ZoneSplit z={s.zone_pct} />
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: 24 }}>
        <div style={{ flex: 1, minWidth: 180 }}>
          <h3 style={{ marginBottom: 6 }}>Strokes</h3>
          <Bars data={s.by_type} />
          {s.avg_shot_speed_kmh && <div className="small muted" style={{ marginTop: 6 }}>Avg shot ground speed {s.avg_shot_speed_kmh} km/h (est.)</div>}
          {s.direction && (
            <>
              <h3 style={{ margin: '12px 0 6px' }}>Shot direction <span className="small muted">(est.)</span></h3>
              <Bars data={s.direction} />
            </>
          )}
          {s.padel_types && Object.keys(s.padel_types).length > 0 && (
            <>
              <h3 style={{ margin: '12px 0 6px' }}>Padel shots</h3>
              <Bars data={s.padel_types} />
            </>
          )}
        </div>
        <div>
          <h3 style={{ marginBottom: 6 }}>Own-half heatmap</h3>
          <Heatmap grid={p.heatmap} size={170} />
        </div>
      </div>
      {s.serves != null && (
        <table className="small">
          <tbody>
            {[
              ['Serves (left / right side)', `${s.serves} (${s.serve_side?.left ?? 0} / ${s.serve_side?.right ?? 0})`],
              ['Returns', s.returns],
              ['Shots from net / transition / back', `${s.shots_by_zone?.net ?? 0} / ${s.shots_by_zone?.transition ?? 0} / ${s.shots_by_zone?.back ?? 0}`],
              ['Shot depth deep / mid / short', `${s.depth?.deep ?? 0} / ${s.depth?.mid ?? 0} / ${s.depth?.short ?? 0}`],
              ['Winners / errors', `${s.winners ?? 0} / ${s.errors ?? 0}`],
              ['Serve points won', s.points_served ? `${s.serve_points_won ?? 0} of ${s.points_served}` : '–'],
              ['Smashes / bandejas & víboras', `${s.overhead_kind?.smash ?? 0} / ${s.overhead_kind?.['bandeja/víbora'] ?? 0}`],
              ['Lobs', s.lobs ?? 0],
              ['Off the back glass', s.off_wall ?? 0],
              ['Last shot of a rally', s.last_shot_of_rally],
              ['Rallies involved', `${s.rally_participation_pct ?? 0}%`],
              ['Net approaches', s.net_approaches],
              ['Distance while ball in play', `${s.distance_in_play_m ?? 0} m`],
              ['Work rate (in play)', s.work_rate_m_per_min != null ? `${s.work_rate_m_per_min} m/min` : '–'],
            ].map(([k, v]) => (
              <tr key={String(k)}><td className="secondary">{k}</td><td className="num" style={{ textAlign: 'right' }}>{v}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function pairExtra(st: Record<string, unknown>): [string, string | number][] {
  const f = st.formation_pct as Record<string, number> | undefined
  const d = st.direction as Record<string, number> | undefined
  const rows: [string, string | number][] = []
  if (f) {
    rows.push(['Attacking (both at net)', `${f.attack_both_net} %`])
    rows.push(['Defending (both back)', `${f.defence_both_back} %`])
    rows.push(['Split (one up, one back)', `${f.split_one_up_one_back} %`])
  }
  if (st.net_control_pct != null) rows.push(['Net control (≥ 1 player at net)', `${st.net_control_pct} %`])
  if (st.rallies_served != null) rows.push(['Rallies served', st.rallies_served as number])
  if (st.avg_time_to_net_after_serve_s != null) rows.push(['Time to net after own serve', `${st.avg_time_to_net_after_serve_s} s`])
  if (d) rows.push(['Crosscourt / down the line / centre', `${d.crosscourt} / ${d['down the line']} / ${d.centre}`])
  return rows
}

function Rallies({ rep, names, seek, review }: {
  rep: Report; names: string[]; seek: (t: number) => void; review: (b: Record<string, Record<string, unknown>>) => void
}) {
  return (
    <div className="card stack" style={{ overflowX: 'auto' }}>
      {rep.match.rally_length_distribution && (
        <div style={{ maxWidth: 420 }}>
          <h3 style={{ marginBottom: 6 }}>Rally length (shots)</h3>
          <Bars data={rep.match.rally_length_distribution} />
        </div>
      )}
      <table>
        <thead><tr><th>#</th><th>Start</th><th>Shots</th><th>Length</th><th>Opened with</th><th>Last shot</th><th>Winner</th></tr></thead>
        <tbody>
          {rep.rallies.map((r) => (
            <tr key={r.id} className="clickable" onClick={() => seek(r.start_t)}>
              <td className="num">{r.id + 1}</td>
              <td className="num">{fmtTime(r.start_t)}</td>
              <td className="num">{r.n_shots}</td>
              <td className="num">{r.duration_s} s</td>
              <td>{r.first_type ?? '–'}</td>
              <td>{r.last_player != null ? names[r.last_player] : '–'}</td>
              <td onClick={(e) => e.stopPropagation()}>
                <select value={r.winner_team ?? ''} onChange={(e) => review({ rallies: { [r.id]: { winner_team: e.target.value === '' ? null : Number(e.target.value) } } })}>
                  <option value="">–</option>
                  {rep.teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                </select>
                {r.winner_source === 'estimated' && <span className="small muted"> est. {Math.round((r.winner_conf ?? 0) * 100)}%</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Shots({ rep, names, seek, review, now }: {
  rep: Report; names: string[]; seek: (t: number) => void; review: (b: Record<string, Record<string, unknown>>) => void; now: number
}) {
  const [who, setWho] = useState<number | ''>('')
  const [kind, setKind] = useState('')
  const kinds = Object.keys(rep.match.type_mix)
  const list = rep.shots.filter((s: Shot) => !s.deleted && (who === '' || s.player === who) && (!kind || s.type === kind))
  return (
    <div className="card stack">
      <div className="row">
        <select value={who} onChange={(e) => setWho(e.target.value === '' ? '' : Number(e.target.value))}>
          <option value="">All players</option>
          {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
        </select>
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="">All strokes</option>
          {kinds.map((k) => <option key={k}>{k}</option>)}
        </select>
        <span className="muted small">{list.length} shots</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Time</th><th>Rally</th><th>Player</th><th>Stroke</th><th>Padel shot</th><th>Tags</th><th>Speed</th><th /></tr></thead>
          <tbody>
            {list.map((s) => (
              <tr key={s.id} className={`clickable ${Math.abs(s.t - now) < 0.4 ? 'active' : ''}`} onClick={() => seek(s.t - 1)}>
                <td className="num">{fmtTime(s.t)}</td>
                <td className="num">{s.rally + 1}</td>
                <td><span className="team-chip"><i style={{ background: teamColor(s.team) }} />{names[s.player]}</span></td>
                <td onClick={(e) => e.stopPropagation()}>
                  <select value={s.type} onChange={(e) => review({ shots: { [s.id]: { type: e.target.value } } })}>
                    {kinds.map((k) => <option key={k}>{k}</option>)}
                  </select>{' '}
                  <span className="small muted">{s.label === 'confirmed' ? '✓ confirmed' : `${Math.round(s.type_conf * 100)}%`}</span>
                </td>
                <td>{s.padel_type ? <>{s.padel_type} <span className="small muted">{Math.round((s.padel_type_conf ?? 0) * 100)}%</span></> : <span className="muted">–</span>}</td>
                <td>{s.volley && <span className="tag">volley</span>}{s.overhead && <span className="tag">overhead</span>}</td>
                <td className="num">{s.speed_kmh ? `${s.speed_kmh} km/h` : '–'}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button className="ghost small" title="Not a shot — remove" onClick={() => review({ shots: { [s.id]: { deleted: true } } })}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
