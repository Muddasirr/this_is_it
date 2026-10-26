/** Small single-series charts: labelled bars, zone split, folded heatmap. Text uses ink tokens;
 *  the only colour carriers are the marks. */

export function Bars({ data, unit = '' }: { data: Record<string, number>; unit?: string }) {
  const max = Math.max(1, ...Object.values(data))
  return (
    <div className="bars" role="table">
      {Object.entries(data).map(([k, v]) => (
        <div className="bar-row" key={k} role="row" title={`${k}: ${v}${unit}`}>
          <span className="secondary" role="cell">{k}</span>
          <div className="track"><div className="fill" style={{ width: `${(v / max) * 100}%`, minWidth: v ? 4 : 0 }} /></div>
          <span className="num" role="cell" style={{ textAlign: 'right' }}>{v}{unit}</span>
        </div>
      ))}
    </div>
  )
}

const ZONES: [keyof Zones, string, string][] = [
  ['net', 'Net', 'var(--seq-1)'],
  ['transition', 'Transition', 'var(--accent)'],
  ['back', 'Back', 'var(--seq-0)'],
]
interface Zones { net: number | null; transition: number | null; back: number | null }

/** Ordered 3-step split (net -> back) with direct labels so the steps never rely on colour alone. */
export function ZoneSplit({ z }: { z: Zones }) {
  return (
    <div>
      <div className="zone-bar">
        {ZONES.map(([k, label, c]) => (
          <span key={k} style={{ width: `${z[k] ?? 0}%`, background: c }} title={`${label}: ${z[k] ?? 0}%`} />
        ))}
      </div>
      <div className="row small secondary" style={{ marginTop: 4, gap: 10 }}>
        {ZONES.map(([k, label]) => (
          <span key={k}>{label} <b className="num" style={{ color: 'var(--text-primary)' }}>{z[k] ?? 0}%</b></span>
        ))}
      </div>
    </div>
  )
}

/** Occupancy on the player's own half: net at the top, back wall at the bottom, seen from behind the player. */
export function Heatmap({ grid, size = 200 }: { grid: number[][]; size?: number }) {
  if (!grid.length) return <div className="muted small">No position data</div>
  const rows = grid.length
  const cols = grid[0].length
  const max = Math.max(...grid.flat())
  return (
    <svg viewBox={`0 0 ${cols} ${rows + 1.4}`} style={{ width: size, height: (size * (rows + 1.4)) / cols }} role="img"
      aria-label="Court occupancy heatmap">
      <rect width={cols} height={rows} fill="var(--surface-2)" />
      {grid.map((r, i) =>
        r.map((v, j) =>
          v > 0 ? (
            <rect key={`${i}-${j}`} x={j} y={i} width={1} height={1} fill="var(--accent)" fillOpacity={0.12 + 0.88 * Math.sqrt(v / max)}>
              <title>{`${(i * 0.5).toFixed(1)}–${(i * 0.5 + 0.5).toFixed(1)} m from net: ${(v * 100).toFixed(1)}% of time`}</title>
            </rect>
          ) : null,
        ),
      )}
      <line x1={0} y1={0} x2={cols} y2={0} stroke="var(--text-primary)" strokeWidth={0.3} />
      <line x1={0} y1={rows * (6.95 / 10)} x2={cols} y2={rows * (6.95 / 10)} stroke="var(--axis)" strokeWidth={0.15} />
      <line x1={cols / 2} y1={0} x2={cols / 2} y2={rows * (6.95 / 10)} stroke="var(--axis)" strokeWidth={0.15} />
      <text x={0} y={rows + 1.1} fontSize={1} fill="var(--text-muted)">net ↑ · back wall ↓</text>
    </svg>
  )
}
