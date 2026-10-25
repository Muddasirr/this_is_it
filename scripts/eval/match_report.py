#!/usr/bin/env python3
"""Print match + team + player stats for matches in the local app (same numbers as the UI).

  python scripts/eval/match_report.py --title gpu_eval_v12 [--facility-name "CourtLens Kaggle eval"]
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def _req(api: str, path: str, tok: str | None = None, data: dict | None = None):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {tok}"} if tok else {})}
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(api + path, data=body, headers=headers)) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="", help="only matches whose title contains this")
    ap.add_argument("--facility-name", default="CourtLens Kaggle eval")
    ap.add_argument("--email", default="corpus179@example.com")
    ap.add_argument("--password", default="Testpass123!")
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--ui", default="http://localhost:5174")
    args = ap.parse_args()
    api = args.api.rstrip("/") + "/api/v1"
    tok = _req(api, "/auth/login", data={"email": args.email, "password": args.password})["access_token"]
    fac = next(f for f in _req(api, "/facilities", tok) if f["name"] == args.facility_name)
    teams = {t["id"]: t["name"] for t in _req(api, f"/facilities/{fac['id']}/teams", tok)}
    for m in _req(api, f"/facilities/{fac['id']}/matches", tok):
        title = m.get("title") or ""
        if args.title not in title:
            continue
        s = _req(api, f"/facilities/{fac['id']}/matches/{m['id']}/stats", tok)
        t = s["totals"]
        print(f"\n### {title}\n{args.ui}/facilities/{fac['id']}/matches/{m['id']}")
        print("  totals: " + ", ".join(f"{k} {t.get(k)}" for k in
                                       ["pass", "dribble", "shot", "goal", "assist", "players", "teams"]))
        for i, tm in enumerate(s.get("by_team") or []):
            name = teams.get(tm.get("team_id"), f"Team {i + 1}")
            print(f"  {name}: possession {tm['possession_pct']}% · passes {tm['pass']} · dribbles {tm['dribble']}"
                  f" · shots {tm['shot']} · goals {tm['goal']} · players {tm['players']}")
        print("  player | min | poss% | touches | pass | rec | drib | shot | goal | move")
        for p in sorted(s["by_player"], key=lambda p: -(p.get("touches") or 0)):
            move = f"{p['distance_m']:.0f} m" if p.get("distance_m") is not None else f"{p.get('distance') or 0:.0f}"
            print(f"  {p['display_name'][:16]:16s} | {p.get('minutes') or 0:.1f} | {p.get('possession_pct') or 0:.0f} |"
                  f" {p.get('touches') or 0} | {p['pass']} | {p.get('pass_received') or 0} | {p.get('dribble') or 0} |"
                  f" {p['shot']} | {p['goal']} | {move}")


if __name__ == "__main__":
    main()
