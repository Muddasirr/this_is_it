import { useEffect, useRef, useState } from 'react'
import { go } from '../App'
import { api, fmtTime } from '../api'
import type { Match } from '../types'

const STATUS_LABEL: Record<Match['status'], string> = {
  uploaded: 'Needs calibration',
  calibrated: 'Ready to analyse',
  queued: 'Queued',
  processing: 'Analysing',
  done: 'Report ready',
  failed: 'Failed',
}

export default function Matches() {
  const [matches, setMatches] = useState<Match[]>([])
  const [err, setErr] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [up, setUp] = useState<number | null>(null)
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const load = () => api.matches().then(setMatches).catch((e) => setErr(String(e.message ?? e)))
  useEffect(() => {
    load()
  }, [])
  const busy = matches.some((m) => m.status === 'queued' || m.status === 'processing')
  useEffect(() => {
    if (!busy) return
    const t = setInterval(load, 2000)
    return () => clearInterval(t)
  }, [busy])

  async function upload() {
    if (!file) return
    setErr('')
    setUp(0)
    try {
      const m = await api.upload(file, title, setUp)
      setFile(null)
      setTitle('')
      go(`/m/${m.id}/calibrate`)
    } catch (e) {
      setErr(String((e as Error).message))
    } finally {
      setUp(null)
    }
  }

  return (
    <div className="stack">
      <div>
        <h1>Matches</h1>
        <p className="secondary" style={{ margin: '4px 0 0' }}>
          Upload a match from a fixed court camera, mark the court once, and get player, pair and rally stats.
        </p>
      </div>

      <div className="card stack">
        <h2>New match</h2>
        <div
          className={`drop ${over ? 'over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
          }}
          onClick={() => input.current?.click()}
          style={{ cursor: 'pointer' }}
        >
          {file ? <b>{file.name}</b> : 'Drop a match video here, or click to choose (mp4, mov)'}
          {file && <div className="small muted">{(file.size / 1e6).toFixed(0)} MB</div>}
          <input ref={input} type="file" accept="video/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </div>
        <div className="row">
          <input placeholder="Title (optional)" value={title} onChange={(e) => setTitle(e.target.value)} style={{ flex: 1, minWidth: 200 }} />
          <button className="primary" disabled={!file || up !== null} onClick={upload}>
            {up !== null ? `Uploading ${Math.round(up * 100)}%` : 'Upload'}
          </button>
        </div>
        {err && <div className="err">{err}</div>}
      </div>

      <div className="card">
        {matches.length === 0 ? (
          <div className="muted">No matches yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Match</th>
                <th>Length</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => (
                <tr key={m.id}>
                  <td>
                    <div style={{ fontWeight: 500 }}>{m.title}</div>
                    <div className="small muted">{new Date(m.created_at).toLocaleString()} · {m.width}×{m.height}</div>
                  </td>
                  <td className="num">{m.duration_s ? fmtTime(m.duration_s) : '–'}</td>
                  <td>
                    <span className={`status ${m.status}`}><span className="dot" />{STATUS_LABEL[m.status]}</span>
                    {(m.status === 'processing' || m.status === 'queued') && (
                      <div className="row small muted" style={{ marginTop: 4 }}>
                        <div className="progress"><div style={{ width: `${m.progress * 100}%` }} /></div>
                        {Math.round(m.progress * 100)}% {m.stage ?? ''}
                      </div>
                    )}
                    {m.status === 'failed' && <div className="err">{m.error?.split('\n')[0]}</div>}
                  </td>
                  <td>
                    <div className="row" style={{ justifyContent: 'flex-end' }}>
                      <button onClick={() => go(`/m/${m.id}/calibrate`)}>{m.calibration ? 'Recalibrate' : 'Calibrate'}</button>
                      {m.calibration && m.status !== 'processing' && m.status !== 'queued' && (
                        <button onClick={() => api.analyze(m.id).then(load).catch((e) => setErr(e.message))}>
                          {m.status === 'done' ? 'Re-run' : 'Analyse'}
                        </button>
                      )}
                      {m.status === 'done' && <button className="primary" onClick={() => go(`/m/${m.id}`)}>Report</button>}
                      <button className="ghost danger" title="Delete match"
                        onClick={() => confirm(`Delete “${m.title}”?`) && api.remove(m.id).then(load).catch((e) => setErr(e.message))}>
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
