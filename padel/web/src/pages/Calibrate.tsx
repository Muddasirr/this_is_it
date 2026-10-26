import { useEffect, useMemo, useRef, useState } from 'react'
import { go } from '../App'
import { api, fmtTime } from '../api'
import Court from '../components/Court'
import type { CourtModel, Match, Pt } from '../types'

const LABEL: Record<string, string> = {
  far_left: 'Far corner · left',
  far_right: 'Far corner · right',
  far_service_left: 'Far service line · left end',
  far_service_t: 'Far service “T”',
  far_service_right: 'Far service line · right end',
  net_left: 'Net · left end (floor)',
  net_centre: 'Net · centre (floor)',
  net_right: 'Net · right end (floor)',
  near_service_left: 'Near service line · left end',
  near_service_t: 'Near service “T”',
  near_service_right: 'Near service line · right end',
  near_left: 'Near corner · left',
  near_right: 'Near corner · right',
}
// The four service-line ends are the easiest to see on most club cameras.
const START = ['far_service_left', 'far_service_right', 'near_service_right', 'near_service_left']

export default function Calibrate({ id }: { id: string }) {
  const [match, setMatch] = useState<Match | null>(null)
  const [court, setCourt] = useState<CourtModel | null>(null)
  const [t, setT] = useState(1)
  const [points, setPoints] = useState<Record<string, Pt>>({})
  const [active, setActive] = useState(START[0])
  const [overlay, setOverlay] = useState(false)
  const [snap, setSnap] = useState(false)
  const [camName, setCamName] = useState('')
  const [cams, setCams] = useState<{ id: string; name: string }[]>([])
  const [result, setResult] = useState<{ err: number | null; refine: Record<string, number> } | null>(null)
  const [msg, setMsg] = useState('')
  const [bust, setBust] = useState('')
  const img = useRef<HTMLImageElement>(null)

  useEffect(() => {
    api.court().then(setCourt)
    api.cameras().then(setCams)
    api.match(id).then((m) => {
      setMatch(m)
      if (m.calibration?.points_px) setPoints(m.calibration.points_px)
      if (m.calibration) setOverlay(true)
    })
  }, [id])

  const order = useMemo(() => (court ? [...START, ...Object.keys(court.landmarks).filter((k) => !START.includes(k))] : START), [court])
  const n = Object.keys(points).length
  // when saved points load, point the user at the first landmark still missing
  useEffect(() => {
    const first = order.find((k) => !(k in points))
    if (first && points[active]) setActive(first)
  }, [court, match?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  function click(e: React.MouseEvent<HTMLDivElement>) {
    const el = img.current
    if (!el || !match) return
    const r = el.getBoundingClientRect()
    const p: Pt = [((e.clientX - r.left) / r.width) * match.width, ((e.clientY - r.top) / r.height) * match.height]
    const next = { ...points, [active]: p }
    setPoints(next)
    setOverlay(false)
    const nextKey = order.find((k) => !(k in next))
    if (nextKey) setActive(nextKey)
  }

  async function fit() {
    setMsg('')
    try {
      const r = await api.calibrate(id, points, t, snap, camName || undefined)
      setMatch(r.match)
      setResult({ err: r.match.calibration?.reproj_m ?? null, refine: r.refine })
      setOverlay(true)
      setBust(String(Date.now()))
      if (camName) api.cameras().then(setCams)
    } catch (e) {
      setMsg((e as Error).message)
    }
  }

  async function useCam(cid: string) {
    try {
      const m = await api.useCamera(id, cid)
      setMatch(m)
      if (m.calibration?.points_px) setPoints(m.calibration.points_px)
      setOverlay(true)
      setBust(String(Date.now()))
    } catch (e) {
      setMsg((e as Error).message)
    }
  }

  if (!match || !court) return <div className="muted">Loading…</div>
  const dur = match.duration_s ?? 0

  return (
    <div className="stack">
      <div className="row">
        <div>
          <h1>{match.title}</h1>
          <div className="secondary">Click court landmarks on the frame. Four points work; more points are more accurate (one bad click is tolerated from five).</div>
        </div>
        <div className="spacer" />
        <button onClick={() => go('/')}>Back</button>
        <button className="primary" disabled={!match.calibration}
          onClick={() => api.analyze(id).then(() => go('/')).catch((e) => setMsg(e.message))}>
          Analyse match
        </button>
      </div>

      <div className="calib-layout">
        <div className="card stack">
          <div className="frame-box" onClick={click}>
            <img ref={img} src={api.frameUrl(id, t, overlay && !!match.calibration, bust)} alt="Video frame for calibration" draggable={false} />
            <svg viewBox={`0 0 ${match.width} ${match.height}`} preserveAspectRatio="none">
              {Object.entries(points).map(([k, p]) => (
                <g key={k}>
                  <circle cx={p[0]} cy={p[1]} r={match.width / 160} fill="none" stroke={k === active ? 'var(--team-b)' : '#fff'} strokeWidth={match.width / 640} />
                  <circle cx={p[0]} cy={p[1]} r={match.width / 900} fill={k === active ? 'var(--team-b)' : '#fff'} />
                </g>
              ))}
            </svg>
          </div>
          <div className="row">
            <span className="small secondary">Frame</span>
            <input type="range" min={0} max={Math.max(1, dur - 0.5)} step={0.5} value={t} style={{ flex: 1 }}
              onChange={(e) => setT(Number(e.target.value))} />
            <span className="num small">{fmtTime(t)}</span>
            <label className="row small" style={{ gap: 6 }}>
              <input type="checkbox" checked={overlay} disabled={!match.calibration} onChange={(e) => setOverlay(e.target.checked)} />
              Show fitted court
            </label>
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <Court landmarks={court.landmarks} activeLandmark={active} doneLandmarks={new Set(Object.keys(points))} height={240} />
            <div className="small" style={{ textAlign: 'center', marginTop: 6 }}>
              Now click: <b>{LABEL[active]}</b>
            </div>
          </div>
          <div className="card lm-list">
            {order.map((k) => (
              <button key={k} className={k === active ? 'on' : ''} onClick={() => setActive(k)}>
                <span>{LABEL[k]}</span>
                <span className="small">
                  {k in points ? (
                    <a onClick={(e) => {
                      e.stopPropagation()
                      const { [k]: _drop, ...rest } = points
                      void _drop
                      setPoints(rest)
                    }}>clear</a>
                  ) : <span className="muted">–</span>}
                </span>
              </button>
            ))}
          </div>
          <div className="card stack">
            <label className="row small" style={{ gap: 6 }}>
              <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
              Snap to painted lines (helps rough clicks)
            </label>
            <input placeholder="Save as camera (e.g. Court 3)" value={camName} onChange={(e) => setCamName(e.target.value)} />
            <button className="primary" disabled={n < 4} onClick={fit}>Fit court ({n} points)</button>
            {result && (
              <div className="small secondary">
                Fit residual: <b className="num">{result.err != null ? `${(result.err * 100).toFixed(1)} cm` : '–'}</b>
                {result.refine?.line_cost_after_px != null && (
                  <> · line distance {result.refine.line_cost_before_px} → {result.refine.line_cost_after_px} px</>
                )}
                <div>Check the yellow overlay sits on the court lines.</div>
              </div>
            )}
            {cams.length > 0 && (
              <select defaultValue="" onChange={(e) => e.target.value && useCam(e.target.value)}>
                <option value="">Use a saved camera…</option>
                {cams.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            )}
            {msg && <div className="err">{msg}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
