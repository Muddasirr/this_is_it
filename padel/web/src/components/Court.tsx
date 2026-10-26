import type { ReactNode } from 'react'
import { playerTag } from '../api'
import type { Pt } from '../types'

const W = 10
const L = 20
const SERVICE = [3.05, 16.95]

export const teamColor = (team: number) => (team === 0 ? 'var(--team-a)' : 'var(--team-b)')

interface Props {
  players?: (Pt | null)[]
  names?: string[]
  highlight?: Pt | null
  marks?: { p: Pt; team: number; title: string; active?: boolean; onClick?: () => void }[]
  landmarks?: Record<string, Pt>
  activeLandmark?: string
  doneLandmarks?: Set<string>
  children?: ReactNode
  height?: number
}

/** Top-down court in metres: far wall at the top, net across the middle (same frame as the analysis). */
export default function Court({ players, names, highlight, marks, landmarks, activeLandmark, doneLandmarks, height = 420 }: Props) {
  return (
    <svg viewBox="-1.2 -1.2 12.4 22.4" style={{ height, width: 'auto', maxWidth: '100%', display: 'block', margin: '0 auto' }}
      role="img" aria-label="Padel court, top-down">
      <rect x={-1.2} y={-1.2} width={12.4} height={22.4} rx={0.4} fill="var(--surface-2)" />
      <rect x={0} y={0} width={W} height={L} fill="var(--court)" />
      <g stroke="var(--court-line)" strokeWidth={0.06} fill="none">
        <line x1={0} y1={SERVICE[0]} x2={W} y2={SERVICE[0]} />
        <line x1={0} y1={SERVICE[1]} x2={W} y2={SERVICE[1]} />
        <line x1={W / 2} y1={SERVICE[0]} x2={W / 2} y2={SERVICE[1]} />
        <rect x={0} y={0} width={W} height={L} strokeOpacity={0.5} />
      </g>
      <line x1={-0.4} y1={10} x2={W + 0.4} y2={10} stroke="var(--court-line)" strokeWidth={0.14} strokeDasharray="0.25 0.12" />
      {landmarks &&
        Object.entries(landmarks).map(([k, p]) => (
          <circle key={k} cx={p[0]} cy={p[1]} r={k === activeLandmark ? 0.45 : 0.28}
            fill={k === activeLandmark ? 'var(--team-b)' : doneLandmarks?.has(k) ? 'var(--good)' : 'var(--surface-1)'}
            stroke="var(--text-primary)" strokeWidth={0.05}>
            <title>{k.replaceAll('_', ' ')}</title>
          </circle>
        ))}
      {marks?.map((m, i) => (
        <circle key={i} cx={m.p[0]} cy={m.p[1]} r={m.active ? 0.42 : 0.3} fill={teamColor(m.team)}
          stroke="var(--surface-1)" strokeWidth={0.08} style={{ cursor: m.onClick ? 'pointer' : undefined, pointerEvents: 'all' }}
          onClick={m.onClick}>
          <title>{m.title}</title>
        </circle>
      ))}
      {highlight && <circle cx={highlight[0]} cy={highlight[1]} r={0.5} fill="none" stroke="var(--text-primary)" strokeWidth={0.08} />}
      {players?.map((p, i) =>
        p ? (
          <g key={i} transform={`translate(${p[0]} ${p[1]})`}>
            <circle r={0.62} fill={teamColor(i < 2 ? 0 : 1)} stroke="var(--surface-1)" strokeWidth={0.08} />
            <text y={0.2} textAnchor="middle" fontSize={0.56} fontWeight={700} fill="#fff">{playerTag(i)}</text>
            <title>{names?.[i] ?? playerTag(i)}</title>
          </g>
        ) : null,
      )}
    </svg>
  )
}
