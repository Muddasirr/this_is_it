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
| **Pair identity** | which pair is which across camera cuts and changes of ends | DINOv2 appearance + joint linker over all segments | 113 / 113 segments on both PT100 finals (EVAL §11) |
| **Rally winner** | team that won the point | model: P(last hitter's pair won) from the last shot, the ball after it and the opponents' reaction; or set in review | **72 %** of points on matches not trained on (guessing 58 %), labels from broadcast scoreboards |
| **Dead ball** | between-point touches (≤ 3 hits, one pair, no exchange) are not rallies | rule | Kuwait 2025: 27 rallies for 25 points, none missed |

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
| Time tracked | how long the player was seen | measured | |
| Winners / errors | last shot of a point the pair won / lost | estimated (point winner model) or confirmed | |
| Serve points won | points won by the pair when this player served | estimated or confirmed | |
| Ball speed to bounce (km/h) | average ball speed from the racket to its first bounce, per shot; per player avg / max / median by stroke | estimated | serves land in the service box 119/122; detected vs labelled ball: median 0.3 km/h apart; no radar truth (EVAL §10) |
| Winners / errors by stroke | the above split by the stroke type of the last shot | estimated or confirmed | |
| Smashes vs bandejas / víboras | overhead at the net = smash (remate), from further back = bandeja / víbora | estimated (position rule) | not validated: no labelled bandeja / víbora footage exists |
| Lobs | hit while both opponents are at the net and answered from behind the service line | estimated | not validated |
| Off the back glass | contact ≤ 1 m from the back glass | estimated (position rule) | not validated |

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
| Points won (%), winners, errors | from rally winners | estimated or confirmed |
| Serve / return points won (%) | by the serving pair of each game | estimated or confirmed |
| Games won, service games held, breaks | padel scoring replayed over the points | **confirmed winners only** |
| Break points won / saved, golden points won | from the replayed score (receiver at game point; 40-40 under golden-point rules) | **confirmed winners only** |
| Points finished at the net (won / lost) | points ended by this pair's shot from the net zone | estimated or confirmed |
| Formation | % attacking (both at net), defending (both back), split | measured |
| Net control | % of time with at least one player at the net | measured |
| Rallies served, time to net after own serve | how fast the serving pair takes the net | estimated |
| Shot direction mix | crosscourt / down the line / centre | estimated |

## Per match

Score (sets, current game; from confirmed winners), points decided, rallies, shots, avg / longest rally (shots), avg rally duration, rally-length distribution (1–3, 4–6, 7–9, 10+), ball-in-play time and %, shots per minute in play, stroke-type and direction mix, data quality (ball seen %, players tracked %, audio used, scene cuts, calibration error, models used). Exports: shots / players / pairs / rallies as CSV, full report as JSON.

## In the app

Video with player tags + ball circle overlay, live top-down mini-court replay (5 Hz), rally timeline (click to jump, coloured by confirmed winner), shot list (filter, jump, fix stroke type, delete false hits), player cards, pair cards with "where they hit from" shot maps, editable player / team names.

## Club cameras (measured 2026-10-26 on a YouTube club match, KLICK indoor court, 720p60 wide-angle)

- Calibration needs the lens-distortion fit (k1 ≈ −0.45): residual 0.09 m with 6 landmarks.
- Player movement works (all 4 tracked 88 % of the time; near players whose feet leave the frame are lost).
- **Ball and hits do not transfer yet**: the ball was found in 8 % of frames → only 3 hits in 6 minutes. The detectors learned the WPT broadcast camera only. Venue retraining with public Roboflow padel datasets is running (`padel-venues`).

## Not tracked yet (roadmap)

Automatic game score (needs > 90 % point-winner accuracy; today 72 % → confirm winners in review), bounces (floor vs glass), bandeja vs víbora (only told apart from smashes by position), forced vs unforced errors, launch (racket) speed — one camera cannot see depth along its viewing direction; we report the average speed to the bounce instead. A detector trained on the public padelcnn shot dataset (bandeja, volleys, lob, off-the-wall …) did not transfer to match footage (0 % coverage) and is not used; bandeja / víbora / lob need labelled match footage.
