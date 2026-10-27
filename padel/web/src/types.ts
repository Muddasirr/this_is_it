export type Pt = [number, number]

export interface Calibration {
  H: number[][]
  image_size: [number, number]
  points_px: Record<string, Pt> | null
  reproj_m: number | null
  k1?: number
}

export interface Match {
  id: string
  title: string
  filename: string
  created_at: string
  status: 'uploaded' | 'calibrated' | 'queued' | 'processing' | 'done' | 'failed'
  stage: string | null
  progress: number
  error: string | null
  fps: number
  n_frames: number
  width: number
  height: number
  duration_s: number | null
  calibration: Calibration | null
  camera_id: string | null
}

export interface CourtModel {
  width: number
  length: number
  net_y: number
  service_y: [number, number]
  net_zone_m: number
  landmarks: Record<string, Pt>
  lines: [Pt, Pt][]
  stroke_types: string[]
}

export interface PlayerStats {
  time_tracked_s: number
  distance_m: number
  avg_speed_kmh: number | null
  max_speed_kmh: number | null
  sprints: number
  zone_pct: { net: number | null; transition: number | null; back: number | null }
  shots: number
  by_type: Record<string, number>
  volleys: number
  overheads: number
  avg_shot_speed_kmh: number | null
  padel_types?: Record<string, number>
  serves?: number
  serve_side?: Record<string, number>
  returns?: number
  shots_by_zone?: Record<string, number>
  direction?: Record<string, number>
  depth?: Record<string, number>
  last_shot_of_rally?: number
  rally_participation_pct?: number
  net_approaches?: number
  distance_in_play_m?: number
  work_rate_m_per_min?: number | null
}

export interface Player {
  id: number
  name: string
  team: number
  stats: PlayerStats
  heatmap: number[][]
}

export interface Team {
  id: number
  name: string
  players: number[]
  stats: Record<string, number | null | Record<string, number>>
}

export interface Shot {
  id: number
  frame: number
  t: number
  rally: number
  player: number
  team: number
  type: string
  type_conf: number
  type_probs: Record<string, number>
  overhead: boolean
  volley: boolean
  speed_kmh: number | null
  pos_m: Pt | null
  ball_px: Pt | null
  label: 'estimated' | 'confirmed'
  deleted?: boolean
  padel_type?: string | null
  padel_type_conf?: number
}

export interface Rally {
  id: number
  start_t: number
  end_t: number
  shots: number[]
  n_shots: number
  duration_s: number
  first_type: string | null
  last_player: number | null
  last_team: number | null
  winner_team: number | null
}

export interface Report {
  match: {
    fps: number
    n_frames: number
    duration_s: number
    image_size: [number, number]
    rallies: number
    avg_rally_shots: number | null
    max_rally_shots: number | null
    avg_rally_s: number | null
    hits: number
    type_mix: Record<string, number>
    rallies_confirmed?: number
    time_in_play_s?: number
    time_in_play_pct?: number
    shots_per_min_in_play?: number | null
    rally_length_distribution?: Record<string, number>
    direction_mix?: Record<string, number>
  }
  players: Player[]
  teams: Team[]
  rallies: Rally[]
  shots: Shot[]
  quality: Record<string, number | boolean | null>
  replay: { hz: number; players: (Pt | null)[][]; ball_px: (Pt | null)[] }
  calibration: Calibration
}
