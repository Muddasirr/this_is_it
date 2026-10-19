const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type Facility = { id: string; name: string; timezone: string; created_at: string };
export type Team = { id: string; facility_id: string; name: string; primary_color: string | null };
export type Player = {
  id: string;
  facility_id: string;
  team_id: string | null;
  display_name: string;
  default_shirt_number: number | null;
  avatar_key?: string | null;
  avatar_url?: string | null;
};
export type Match = {
  id: string;
  facility_id: string;
  title: string;
  status: string;
  scheduled_date: string | null;
  notes: string | null;
  created_at: string;
};
export type VideoAsset = {
  id: string;
  kind: string;
  object_key: string;
  validation_status: string;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  frame_rate: number | null;
  codec: string | null;
};
export type ProcessingJob = {
  id: string;
  status: string;
  stage: string | null;
  progress_pct: number;
  error_message: string | null;
};
export type CameraSync = {
  id: string;
  offset_ms: number;
  confidence: number | null;
  method: string;
  is_manual: boolean;
  notes: string | null;
};
export type Track = {
  id: string;
  camera_label: string;
  track_key: string;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
  class_name: string | null;
  identity: {
    link_id: string;
    player_id: string | null;
    state: string;
    confidence: number | null;
    source: string;
  } | null;
};
export type MatchEvent = {
  id: string;
  event_type: string;
  canonical_ts_ms: number;
  confirmation: string;
  confidence: number | null;
  participants: { player_id: string | null; track_id: string | null; role: string }[];
  observations: { camera_label: string; camera_ts_ms: number; evidence_key: string | null }[];
};
export type MatchStats = {
  match_id: string;
  status: string;
  totals: Record<string, number>;
  by_player: {
    player_id: string;
    display_name: string;
    shirt_number?: number | null;
    team_id?: string | null;
    avatar_url?: string | null;
    goal: number;
    shot: number;
    pass: number;
    assist?: number;
    possession_pct?: number;
    touches?: number;
    distance?: number;
    heatmap?: number[][];
    attribution: string;
  }[];
  heatmap?: {
    width: number;
    height: number;
    match: number[][];
    unit?: string;
    note?: string;
  };
  labels: Record<string, string>;
};

function authHeaders(json = true): HeadersInit {
  const token = localStorage.getItem("token");
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { ...authHeaders(!(init?.body instanceof FormData)), ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  register: (email: string, full_name: string, password: string) =>
    request("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, full_name, password }),
    }),
  login: async (email: string, password: string) => {
    const data = await request<{ access_token: string }>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem("token", data.access_token);
    return data;
  },
  me: () => request<{ id: string; email: string; full_name: string }>("/api/v1/auth/me"),
  listFacilities: () => request<Facility[]>("/api/v1/facilities"),
  createFacility: (name: string) =>
    request<Facility>("/api/v1/facilities", { method: "POST", body: JSON.stringify({ name }) }),
  listTeams: (facilityId: string) => request<Team[]>(`/api/v1/facilities/${facilityId}/teams`),
  createTeam: (facilityId: string, name: string) =>
    request<Team>(`/api/v1/facilities/${facilityId}/teams`, {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  listPlayers: (facilityId: string) => request<Player[]>(`/api/v1/facilities/${facilityId}/players`),
  createPlayer: (facilityId: string, display_name: string, default_shirt_number?: number, team_id?: string) =>
    request<Player>(`/api/v1/facilities/${facilityId}/players`, {
      method: "POST",
      body: JSON.stringify({ display_name, default_shirt_number, team_id }),
    }),
  listMatches: (facilityId: string) => request<Match[]>(`/api/v1/facilities/${facilityId}/matches`),
  createMatch: (facilityId: string, title: string) =>
    request<Match>(`/api/v1/facilities/${facilityId}/matches`, {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  getMatch: (facilityId: string, matchId: string) =>
    request<Match>(`/api/v1/facilities/${facilityId}/matches/${matchId}`),
  addRoster: (facilityId: string, matchId: string, player_id: string, shirt_number: number) =>
    request(`/api/v1/facilities/${facilityId}/matches/${matchId}/roster`, {
      method: "POST",
      body: JSON.stringify({ player_id, shirt_number }),
    }),
  getUploadUrl: (facilityId: string, matchId: string, camera_label: "A" | "B", filename: string, byte_size: number) =>
    request<{ video_asset_id: string; upload_url: string }>(
      `/api/v1/facilities/${facilityId}/matches/${matchId}/videos/upload-url`,
      {
        method: "POST",
        body: JSON.stringify({ camera_label, filename, content_type: "video/mp4", byte_size }),
      },
    ),
  listVideos: (facilityId: string, matchId: string) =>
    request<VideoAsset[]>(`/api/v1/facilities/${facilityId}/matches/${matchId}/videos`),
  getSync: (facilityId: string, matchId: string) =>
    request<CameraSync | null>(`/api/v1/facilities/${facilityId}/matches/${matchId}/sync`),
  updateSync: (facilityId: string, matchId: string, offset_ms: number, notes?: string) =>
    request<CameraSync>(`/api/v1/facilities/${facilityId}/matches/${matchId}/sync`, {
      method: "PUT",
      body: JSON.stringify({ offset_ms, notes }),
    }),
  startProcessing: (facilityId: string, matchId: string) =>
    request<ProcessingJob>(`/api/v1/facilities/${facilityId}/matches/${matchId}/process`, { method: "POST" }),
  listJobs: (facilityId: string, matchId: string) =>
    request<ProcessingJob[]>(`/api/v1/facilities/${facilityId}/matches/${matchId}/jobs`),
  cancelJob: (facilityId: string, matchId: string, jobId: string) =>
    request<ProcessingJob>(`/api/v1/facilities/${facilityId}/matches/${matchId}/jobs/${jobId}/cancel`, {
      method: "POST",
    }),
  retryJob: (facilityId: string, matchId: string) =>
    request<ProcessingJob>(`/api/v1/facilities/${facilityId}/matches/${matchId}/jobs/retry`, { method: "POST" }),
  listTracks: (facilityId: string, matchId: string) =>
    request<Track[]>(`/api/v1/facilities/${facilityId}/matches/${matchId}/tracks`),
  setTrackIdentity: (facilityId: string, matchId: string, trackId: string, player_id: string | null) =>
    request(`/api/v1/facilities/${facilityId}/matches/${matchId}/tracks/${trackId}/identity`, {
      method: "PUT",
      body: JSON.stringify({ player_id, state: player_id ? "confirmed" : "unknown" }),
    }),
  listEvents: (facilityId: string, matchId: string) =>
    request<MatchEvent[]>(`/api/v1/facilities/${facilityId}/matches/${matchId}/events`),
  reviewEvent: (
    facilityId: string,
    matchId: string,
    eventId: string,
    action: "accept" | "reject" | "edit",
    extra?: { player_id?: string; notes?: string },
  ) =>
    request<MatchEvent>(`/api/v1/facilities/${facilityId}/matches/${matchId}/events/${eventId}/review`, {
      method: "POST",
      body: JSON.stringify({ action, ...extra }),
    }),
  createClip: (facilityId: string, matchId: string, eventId: string) =>
    request<{ playback_url: string; share_token: string }>(
      `/api/v1/facilities/${facilityId}/matches/${matchId}/events/${eventId}/clip`,
      { method: "POST" },
    ),
  getStats: (facilityId: string, matchId: string) =>
    request<MatchStats>(`/api/v1/facilities/${facilityId}/matches/${matchId}/stats`),
  publishMatch: (facilityId: string, matchId: string) =>
    request<{
      match_id: string;
      status: string;
      share_token?: string;
      report_path?: string;
      api_report?: string;
    }>(`/api/v1/facilities/${facilityId}/matches/${matchId}/publish`, { method: "POST" }),
  renamePlayer: (facilityId: string, playerId: string, display_name: string) =>
    request<Player>(`/api/v1/facilities/${facilityId}/players/${playerId}`, {
      method: "PATCH",
      body: JSON.stringify({ display_name }),
    }),
  getPublicReport: (shareToken: string) =>
    request<{
      title: string;
      facility_name: string;
      status: string;
      totals: Record<string, number>;
      by_player: {
        display_name: string;
        goal: number;
        shot: number;
        pass: number;
        possession_pct: number;
        touches: number;
        distance: number;
        attribution: string;
      }[];
      heatmap: { width: number; height: number; match: number[][]; note?: string };
      labels: Record<string, string>;
    }>(`/api/v1/share/matches/${shareToken}`),
  runFeasibility: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(`${API_URL}/api/v1/phase0/feasibility`, {
      method: "POST",
      headers: authHeaders(false),
      body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  },
};

export async function uploadFile(uploadUrl: string, file: File) {
  const res = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "Content-Type": file.type || "video/mp4" },
    body: file,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
}
