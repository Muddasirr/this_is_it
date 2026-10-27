# PadelLens — what we track

Every number in a report carries a label: **measured** (from calibrated positions), **estimated** (model output), **confirmed** (edited / accepted in the review UI). Accuracy figures are held-out results from `EVAL.md` (PadelTracker100 broadcast camera, model trained on the other final) unless marked otherwise.

## Events (timeline items)

| Event | What it is | How | Validation |
| --- | --- | --- | --- |
| **Hit / shot** | racket contact: time, hitter, team, court position | **audio (racket-impact sound CRNN) + video** (ball trajectory change, wrist proximity) → GBM → peaks | F1 **0.94** fused on 11 unseen tournaments (video-only 0.83); PT100 video-only 0.88 / 0.87; hitter correct 89.7 % (team 93.6 %) |
| **Stroke type** | serve · forehand · backhand · smash · other | RGB clip model on the hitter crop (sees the racket) fused with a pose model | 0.79 / 0.78 accuracy |
| **Volley** tag | contact made inside the net zone (≤ 4 m from the net) | position rule | estimated, not yet validated |
| **Overhead** tag | contact with the wrist above the head (bandeja / víbora / smash family) | pose rule | estimated, not yet validated |
| **Rally** | sequence of hits; start, end, length, duration, who opened it, who hit last | hits grouped by gap (4 s) + scene cuts | count within +3 % / −8 % |
| **Rally winner** | team that won the point | set in the review UI | confirmed only (no model yet) |

## Per player

| Stat | Definition | Label | Validation |
| --- | --- | --- | --- |
| Distance covered (m) | path length of the smoothed floor track | measured | pair totals within ≤ 2.1 % |
| Avg / max speed (km/h) | distance / time tracked; 99.5th-percentile speed | measured | from the same tracks |
| Sprints | runs > 4 m/s lasting ≥ 0.5 s | measured | |
| Time in zone (%) | net (≤ 4 m from net) · transition (4 m–service line) · back (behind service line) | measured | position error 5 cm median |
| Own-half heatmap | 0.5 m occupancy grid, folded so every player is seen from behind | measured | |
| Shots, by stroke type | counts of the hits above | estimated | |
| Volleys, overheads | counts of the tags above | estimated | |
| Avg shot ground speed (km/h) | hitter → next hitter distance / flight time | estimated (ball height unknown from one camera) | |
| Time tracked | how long the player was seen | measured | |

### Tactical (per player, estimated where they depend on detected shots)

| Stat | Definition |
| --- | --- |
| Serves, serve side | first shot of each rally; side of the court it was hit from (left / right as the player faces the net) |
| Returns | second shot of each rally |
| Shots by zone | net / transition / back position at contact |
| Shot direction | crosscourt / down the line / centre, from where the receiving opponent had to play it |
| Shot depth | deep / mid / short, from the receiving opponent's position |
| Last shot of a rally | how many rallies ended on this player's shot (winner *or* error — confirm in review) |
| Rallies involved | % of rallies with at least one shot by the player |
| Net approaches | moves from behind the service line into the net zone, staying ≥ 1 s (measured) |
| Distance in play, work rate | metres while the ball is in play, and per minute of play (measured) |

## Per pair

| Stat | Definition | Label |
| --- | --- | --- |
| Both at the net (%) | both partners in the net zone | measured |
| Both back (%) | both behind the service line | measured |
| Avg partner distance, lateral spacing, depth offset (m) | how the pair holds its shape | measured |
| Pair distance (m) | sum of both players | measured |
| Shots, last shot of rally | team totals | estimated |
| Rallies won | from confirmed rally winners | confirmed |
| Formation | % attacking (both at net), defending (both back), split | measured |
| Net control | % of time with at least one player at the net | measured |
| Rallies served, time to net after own serve | how fast the serving pair takes the net | estimated |
| Shot direction mix | crosscourt / down the line / centre | estimated |

## Per match

Rallies, shots, avg / longest rally (shots), avg rally duration, rally-length distribution (1–3, 4–6, 7–9, 10+), ball-in-play time and %, shots per minute in play, stroke-type and direction mix, data quality (ball seen %, players tracked %, audio used, scene cuts, calibration error, models used). Exports: shots / players / pairs / rallies as CSV, full report as JSON.

## In the app

Video with player tags + ball circle overlay, live top-down mini-court replay (5 Hz), rally timeline (click to jump, coloured by confirmed winner), shot list (filter, jump, fix stroke type, delete false hits), player cards, pair cards with "where they hit from" shot maps, editable player / team names.

## Club cameras (measured 2026-10-26 on a YouTube club match, KLICK indoor court, 720p60 wide-angle)

- Calibration needs the lens-distortion fit (k1 ≈ −0.45): residual 0.09 m with 6 landmarks.
- Player movement works (all 4 tracked 88 % of the time; near players whose feet leave the frame are lost).
- **Ball and hits do not transfer yet**: the ball was found in 8 % of frames → only 3 hits in 6 minutes. The detectors learned the WPT broadcast camera only. Venue retraining with public Roboflow padel datasets is running (`padel-venues`).

## Not tracked yet (roadmap)

Point winner / score automatically, bounces (floor vs glass), lobs, bandeja vs víbora vs smash, serve side and return, ball speed in the air, error / winner classification. A detector trained on the public padelcnn shot dataset (bandeja, volleys, lob, off-the-wall …) did not transfer to match footage (0 % coverage) and is not used; bandeja / víbora / lob need labelled match footage.
