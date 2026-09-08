# Integration: this server and the robot's ROS 2 workspace

The two halves of the Deep3R seeking system were built and, until 2026-09-08,
**not connected**. They are now: the robot speaks protocol 2, the server
produces the verdict the robot's exploration runs on, and the four contract
mismatches that would still have bitten afterwards are resolved. One item — 8,
whether the SEEKING circuit steers T1 — is deliberately left open, because it is
a research decision rather than an engineering one.

This document records what each end expects of the other, where those
expectations disagreed, and what was done about it. It is the coordination
contract between four repositories:

| Repository | Role |
| --- | --- |
| `RoboCamStreamProcessing` (this one) | the server: reconstruction, comparison, the T2 decision stage |
| `mecanumbot_ws/src/mecanumbot_server` | the robot's end of the link (`mecanumbot_deep3r`) |
| `mecanumbot_ws/src/mecanumbot` | `mecanumbot_custom_nav2`: exploration, the robot's half of the comparison |
| `mecanumbot_ws/src/mecanumbot_behaviours` | `mecanumbot_seek`: the T2 behaviour tree |

The system is `docs/system_diagram_v2.drawio.png`. Read [The map loop](../README.md#the-map-loop)
first; this document assumes it.

---

## The headline

**The server spoke protocol v2. The ROS client spoke v1.** *(Fixed 2026-09-08.
Kept here because it is the shape of the problem, and the first thing to check
if the loop ever goes quiet again.)*

`deep3r_client_node.py` constructed `RoboCamClient` with a camera source and
nothing else — no lidar, odometry, grid or exits source, and none of the
announcement callbacks:

```python
self.client = client_module.RoboCamClient(
    server=..., client_id=..., max_inflight=..., on_result=self._on_result,
)
```

So `scan`, `odom`, `map` and `phase` were never sent, and `map_update`,
`pose_hint` and `found` were never received. Searching the whole ROS workspace
for `map_update`, `pose_hint`, `on_found`, `t1_exit` and `exits_result` returned
**zero hits**.

Everything in the map loop and the whole T2 decision stage was therefore dead
code from the robot's point of view — not because either end was wrong, but
because nothing carried the messages between them. `exits` is still not sent;
the robot judges its own frontiers, and the server's ranking of them is
advisory (see item 5).

---

## The mismatches

Ordered by what blocks what. Status is updated as each is closed.

### 1. Three topics have no publisher — BLOCKING

| Topic | Subscribed by | Published by |
| --- | --- | --- |
| `/mecanumbot/deep3r/map_agreement` | `mecanumbot_map_agreement`, `mecanumbot_frontier_explorer` | **nothing** |
| `/mecanumbot/seek/target` | `AcquireSeekTarget` | **nothing** |
| `/mecanumbot/seek/detections` | `WatchForObject` | **nothing** |

Consequences, each of which looks like a different bug from the robot:

- The explorer's `CLOUD` exit criterion can never be satisfied, so with
  `require_cloud: true` **T1 never ends**. Only the `require_cloud:=false`
  dry-run path completes.
- `AcquireSeekTarget` always reaches `seek_target_timeout` and fails, so every
  seeking episode is abandoned before it begins.
- `WatchForObject` — the parallel branch the entire two-branch search design
  exists for — has no perception source at all. The only onboard detectors are
  `mecanumbot_sensorprocess_smart` (people) and `mecanumbot_detect_tennis.py`;
  neither publishes `vision_msgs/Detection3DArray`.

**Status: CLOSED (2026-09-08).** `mecanumbot_deep3r` now publishes all three. `agreement` and `found` are routed by `bridge.py`; `found` splits on `basis` so `live` goes to `seek/detections` and `memory` to `seek/target`, which is the tree's two-branch distinction. The server's decision stage is the only object detector in this system, so it supplies both.

### 2. `MapCloudAgreement` cannot be filled from what this server computes — BLOCKING

Nine of the message's fields do not exist anywhere in `robocam/`:
`grid_coverage`, `cloud_coverage`, `compared_cells`, `conflicting_cells`,
`uncertain_points`, `uncertain_scores`, `uncertain_kinds`, `uncertain_heights`,
`uncertain_radii`.

`compare()` reports `agreement` and emits a **cell patch**; the robot expects a
**region list** — centres, radii, kinds and heights. The clustering step that
turns one into the other is unimplemented on both ends. Writing the bridge node
does not fix this: the data is not produced yet.

**Status: CLOSED (2026-09-08).** `robocam/regions.py` clusters the disagreement into regions (centre, radius, kind, height) and computes both coverage numbers; they travel in a new `agreement` announcement. It is a separate message from `map_update` on purpose: the most important verdict this server sends is "the cloud agrees with nothing", and that one produces no patch.

### 3. `map_id` means opposite things on the two ends — CORRECTNESS

| Where | Meaning |
| --- | --- |
| `wire.map_update`, `wire.found` | **the robot's** SLAM map, echoed back |
| `seek.Placement.cloud_map_id` | CUT3R's reconstruction session (an int) |
| `MapCloudAgreement.msg` | declares `string map_id`, documents it as *the cloud's* |
| `deep3r_client_node._publish` | warns on `data["map_id"]`, which is CUT3R's |

Two different identities under one name, in a field whose entire purpose is to
say "everything you accumulated is now void". Keyed on the wrong one, the robot
discards state on the wrong event and keeps it across the event that actually
invalidates it — while looking correct.

**Status: CLOSED (2026-09-08).** `MapCloudAgreement` now carries both: `map_id` (the robot's SLAM map, matching the server's wire convention) and `cloud_map_id` (CUT3R's session). The split exposed a real bug — `agreement.py` reset its accumulated keepouts on the *cloud's* id, but regions are map-frame facts that survive a reconstruction restart and die with a SLAM restart. It now keys on the robot's; `exit_criteria.py`, which measures cloud growth, correctly keys on the cloud's.

### 4. The grasp band disagrees with itself — CORRECTNESS

Both ends cite the same hardware: the grabber shafts sit at z ≈ 0.034 m with a
0.116 m clear gap, so the band is 0.034 … 0.150 m.

| | server (`config/server.yaml`) | robot (`seek_setting_constants.yaml`) |
| --- | --- | --- |
| band max | `grasp_z_max: 0.12` | `seek_grasp_height_max: 0.15` |
| band min | `grasp_z_min: -0.05` | `seek_grasp_height_min: 0.03` |
| standoff | `reach_radius_m: 0.45` | `seek_grasp_distance: 0.30` |

Two dead zones where the two ends give opposite answers about the same object:

- **z ∈ [-0.05, 0.03)** — the server says `reachable`, the tree says `too_low`
  and goes to alert a person about something it was told it could pick up.
- **z ∈ (0.12, 0.15]** — the server says `too_high`, but the grabbers would
  close on it. A retrievable object is reported as needing a human.

And the approach pose is computed at 0.45 m from the target, which the tree's
0.30 m grasp check then rejects as too far to try.

**Status: CLOSED (2026-09-08).** The server moved to the robot's hardware-derived numbers: `grasp_z_max` 0.12 -> 0.15, `grasp_z_min` -0.05 -> 0.03, `reach_radius_m` 0.45 -> 0.30, `approach_standoff_m` 0.8 -> 0.55. The conflated below-floor test moved to its own `floor_tolerance_m`, and the server gained the `too_low` verdict it never had — so an object in a recess is now reported as something to tell a person about rather than as a reconstruction error. Both config files cross-reference each other.

### 5. T1 exit is implemented twice, and the two share no test — DESIGN

| Server (`seek.t1_exit_criteria`) | Robot (`custom_nav2.exit_criteria`) |
| --- | --- |
| `explored` ≥ 0.80 | — |
| `no_open_exits` (needs `exits` the robot never sends) | `FRONTIERS` (computed locally) |
| `not_growing` | `STABLE` (plus loop-closure settling) |
| — | `GAIN` (cells per **metre driven**) |
| `placed` (mean agreement ≥ 0.15) | `CLOUD` (needs the message that never arrives) |
| floors: `t1_min_frames`, `t1_min_runtime_s` | `BUDGET` (time, distance, battery) |

Neither can currently run to completion. Worse, nothing joins the robot's
`exploration/finished` latch to a `phase` message, so even a perfect T1 never
tells the server that T2 has begun — and this server never changes phase on its
own, by design.

**The resolution: the robot decides, the server advises.** The robot owns the
latch, owns the frontier list (only it knows which frontiers it has already
tried), and owns the battery. The server owns the one thing the robot cannot
know — whether the cloud is placed and covering. So:

```
finished = (FRONTIERS or GAIN) and STABLE and CLOUD, or BUDGET
CLOUD    = placed (mean agreement >= 0.15)
           and grid_coverage >= threshold
           and the cloud has stopped growing
           and no CUT3R reset within the settle window
```

`placed` carries the most weight: a fully explored grid over a cloud that was
never correctly placed is a failed T1 that looks exactly like a successful one,
and it surfaces an hour later as "the detector never sees anything". The
server's `explored` / `no_open_exits` / `not_growing` become **reported
diagnostics** rather than decisions — they duplicate tests the robot makes with
better information.

Two things are added rather than reconciled:

- **The server's floors move to the robot.** `BUDGET` has no *minimum*: a first
  grid arriving before the robot has moved satisfies "not growing" trivially.
- **A reset-settle window, which neither end has today.** T1 must not finish
  within N seconds of a CUT3R reset, for exactly the reason the robot will not
  finish within `loop_closure_settle` of a SLAM loop closure — a late reset
  means the cloud T2 is about to search was built from a fraction of the drive.

**Status: CLOSED (2026-09-08).** The robot decides, the server advises. The robot gained the `placed` test it was missing entirely — it logged `agreement` without ever testing it — plus the reset-settle window and a runtime floor. The server's block is marked advisory and its `explored`/`no_open_exits`/`not_growing` are now documented as diagnostics.

### 6. In T1 the robot cannot satisfy this server's pose requirement — BLOCKING

`odom.expect_frame` defaults to `map` and the server **refuses** to compare
across frames rather than absorbing a mismatch. The README says to use
`--odom-topic /amcl_pose`.

But T1 runs under slam_toolbox, which publishes no `/amcl_pose` — which is
precisely why `mecanumbot_frontier_explorer` defaults to `pose_source: tf`.

A bridge that sends `/odom` has every pose rejected as `bad_odom`, `compare`
never runs, `mean_agreement` stays `null`, and that in turn blocks the server's
own `placed` test. The failure mode is correct and the cause is invisible. The
bridge must synthesise the pose from the `map → mecanumbot/base_link`
transform, exactly as the explorer does.

**Status: CLOSED (2026-09-08).** `sources.TfOdomSource` samples `map -> mecanumbot/base_link` at 10 Hz. A failed lookup is skipped rather than substituted: sending the last known pose while TF is out would place every cloud where the robot used to be, with full confidence.

### 7. The target is free text here and a class label there — DESIGN

This server takes the target as free text and hands it to OWLv2 verbatim,
because "the red mug on the desk" beats "mug" and the description is half of
what makes the object findable. `/mecanumbot/seek/request` is documented as "a
class label", and it lands in `SeekingState.object_class` and
`SeekAlert.object_class` — the trial record.

Free text wins (it is the thing that makes an open-vocabulary detector worth
having), so the two message comments and the record fields have to say so.

**Status: CLOSED (2026-09-08).** Free text wins. `ros_interfaces.py`, the seek README and both `object_class` message comments now say so, and the bridge forwards `seek/request` to the server's mission target verbatim.

### 8. The SEEKING circuit does not drive T1 — RESEARCH

`mecanumbot_seek`'s README states that with no object named, the circuit sits at
baseline in the `undirected` phase and *"the autonomous exploration in
`mecanumbot_custom_nav2` is what it drives"*.

`mecanumbot_custom_nav2` contains **zero** references to `seeking`, `arousal` or
`expectancy`. Its goal scoring is entirely its own, and no `SeekingState`
crosses between the two packages. The Panksepp grounding currently holds for T2
only.

This is a research decision, not a bug: either the circuit modulates
exploration (and the claim becomes true), or the claim is restated to cover T2
alone. It must not stay as written.

**Status: OPEN — deliberately (2026-09-08).** The claim has been corrected in `mecanumbot_seek`'s README, `seeking.py` and `mecanumbot_custom_nav2`'s README rather than made true by wiring. What an undirected circuit is entitled to change during exploration — whether falling expectancy widens the frontier search, whether extinction may end T1 — is a thesis question about how far the model reaches, and wiring it would answer it by accident. **This one is Csenge's to decide.**

### 9. "The second scan updates the first" is not implemented — DESIGN

The requirement is that T2's scan updates T1's reconstruction. As built,
`reset_on_gap` and `reset_every` *start a new map* on discontinuity, the client
is told to treat a new `map_id` as a new world frame, and nothing merges T2's
reconstruction into T1's cloud. `recall()` retrieves T1 keyframes but does not
update the cloud they came from.

So a phase change plus any gap currently yields a fresh, unrelated coordinate
frame — the opposite of the requirement.

**Status: CLOSED, and the finding was partly wrong (2026-09-08).** The recurrent state already survives the phase change: nothing resets on `t1 -> t2`, so T2 folds its frames into the same state, same `map_id`, same world frame. That is what "the second scan updates the first" means in practice, and it is now pinned by a test. What was missing was any guarantee: a reset during T2 silently ends the continuity and voids every coordinate the robot was given from the T1 cloud, so it now logs loudly, and `reset_every` must stay 0 for a real mission.

### Minor

- `config/deep3r.yaml` sets `client_path: ~/robocam_client.py`; the node's
  declared default is `~/server/RoboCamStreamProcessing/link/robocam_client.py`.
- `mecanumbot_deep3r/README.md` still describes the robot as pinned to nipg36,
  which [this README](../README.md#serving-it) explicitly retracted.

### 10. The robot merged regions across kinds — CORRECTNESS

Found by running the loop end to end rather than by reading either end.
`AgreementModel._absorb` merged any two regions within `merge_radius`
regardless of kind, so a table pushed against a glass partition — one
`cloud_only` and one `map_only`, routinely within a merge radius of each other —
became a single region whose kind was decided by whichever verdict arrived last.
The same two surfaces therefore produced a keepout or a collision depending on
message order. The server clusters within a kind for exactly this reason
(`robocam/regions.py`); this end undid it.

**Status: CLOSED (2026-09-08).** `_absorb` now only merges within a kind.

---

## Checking the loop end to end

Neither repository's test suite can cover the join, so this is the check that
does. It runs the server's clustering, puts the result on the wire, reads it
back through the robot's bridge, and asks the robot's agreement model what it
would do about it. No ROS graph, no server, no GPU:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 - <<'EOF'
import sys, numpy as np
for p in ("/home/csenge/Documents/mecanumbot_directories/RoboCamStreamProcessing",
          "/home/csenge/Documents/mecanumbot_ws/src/mecanumbot_server/mecanumbot_deep3r",
          "/home/csenge/Documents/mecanumbot_ws/src/mecanumbot/mecanumbot_custom_nav2"):
    sys.path.insert(0, p)
from robocam import regions, wire
from robocam.occupancy import Grid
from mecanumbot_deep3r import bridge
from mecanumbot_custom_nav2.agreement import AgreementModel

cells = np.zeros((60, 60), np.int8)
cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
grid = Grid(cells=cells, resolution=0.05, origin=(0., 0., 0.),
            frame="map", map_id="m")
cloud = np.zeros((60, 60), bool); cloud[20:30, 20:30] = True
h = np.full((60, 60), np.nan, np.float32); h[20:30, 20:30] = 0.30   # a step

found = regions.extract_regions(cloud, grid, h, block_m=0.5)
cover = regions.coverage(cloud, grid)
pl = regions.agreement_payload({"agreement": 0.42, "cloud_cells": 100,
                                "missing": 10, "new_cells": 90}, found, cover)
hdr = wire.agreement(seq=1, frame="map", map_id="m", cloud_map_id=3,
                     grid_coverage=pl["grid_coverage"],
                     cloud_coverage=pl["cloud_coverage"], agreement_value=0.42,
                     compared_cells=pl["compared_cells"],
                     conflicting_cells=pl["conflicting_cells"], regions=pl)
parsed = bridge.agreement_regions(hdr)
m = AgreementModel()
m.update(0.0, [(r["x"], r["y"]) for r in parsed], [r["score"] for r in parsed],
         kinds=[r["kind"] for r in parsed],
         heights=[(np.nan if r["height"] is None else r["height"]) for r in parsed],
         radii=[r["radius"] for r in parsed], map_id=hdr["map_id"])
print(f"server {len(found)} -> wire -> parsed {len(parsed)} -> held {len(m.regions)}")
for r in m.regions:
    print(f"  {r.kind:12s} -> {m.action_for(r)}")
EOF
```

Expected: three regions in, three held, and the `cloud_only` one at 0.30 m
becoming a **keepout** — a step the lidar plane passed over, now lethal in the
costmap. If the count drops on the way through, the two ends disagree about the
region contract; that is what caught item 10.

---

## Feeding back into the reconstruction

The diagram's arrow from `Compare` back into `Deep3r`. The question is whether
the robot can improve the cloud rather than only consume it.

**Not by editing the cloud.** CUT3R is feed-forward with recurrent state; there
is no write path into that state, and its one "imagine a viewpoint" entry point
(`inference_step`) is a probe that discards its own update. Three channels that
do work, in order of value against cost:

**(a) View planning.** The *"non-full lines"* — a wall the lidar drew as a
broken line — are `unobserved` or `disagreement` regions: places to go and look
from, not corrections to either map. More frames from a better viewpoint is the
only thing that reliably improves a feed-forward reconstruction. The robot half
is built (`revisit_regions`, interleaved with frontier goals so the explorer
does not stop exploring); the server half is the region extraction of item 2.

**(b) Scale and rigid anchoring — the highest-value one, and not built.**
`data.scale_check` already computes the ratio between the cloud's near depth and
the lidar's forward range. Nothing consumes a correction: there is no
scale-correction path anywhere on this server. Promoting that ratio to a
published running estimate, and letting the server apply a per-session scale and
a rigid alignment onto the map frame, is genuine feedback into the
reconstruction, needs no change to the model, and fixes the drift that would
otherwise make T2 search a misaligned cloud. **This is where the work after the
plan below should go.**

**(c) Conditioning CUT3R on the 2D occupancy.** A research project of its own,
not a next step.

One direction trap worth stating: `pose_hint` runs cloud → SLAM. It is not
feedback into the cloud, and counting it as such hides the fact that (b) is
missing.

### Which source is more reliable

They are reliable about **different things**, and the architecture depends on
not collapsing them:

- The **2D lidar map** is better on geometry and global consistency — metric by
  construction, loop-closed, does not drift. But it is one horizontal slice,
  blind above and below, and cannot tell a table from a wall.
- The **cloud** is better on coverage and semantics — it sees the volume and
  carries colour, which is what an object can be located in. But it anchors its
  world frame on the first frame of a `map_id`, drifts independently of
  odometry, and is only approximately metric.

So: **the 2D map is the frame of record for navigation; the cloud is the frame
of record for what is where.** The comparison is not an adjudication between
them — it is the transform plus a disagreement map. That is why
`MapCloudAgreement` carries its points in the *map* frame: the server owns the
alignment, and the robot never reasons in a frame that drifts under it.

The height axis is where the split pays. From the lidar alone, a mug on a table
and a mug on the floor behind a chair are the same fact — "something is at
(x, y) and the robot cannot get to it". With a height they become two different
problems with two different answers: drive around it, or go and tell somebody.

---

## The plan

Items 1–3 stand between the current state and a system that runs end to end.
Items 4–7 stand between it running and it being right.

All seven were worked through on 2026-09-08; item 7 was answered by correcting
the claim rather than by building.

| # | Item | Repos touched |
| --- | --- | --- |
| 1 | The bridge node: sensor sources up, announcements down, TF-derived pose, `exploration/finished` → `phase` | `mecanumbot_server` |
| 2 | Region extraction and coverage metrics, so `MapCloudAgreement` can be filled | this one |
| 3 | Split `map_id` into two named identities everywhere | both |
| 4 | One grasp band, derived once from the hardware | both |
| 5 | One T1 exit rule: robot decides, server advises | both |
| 6 | Free text vs class label; cross-phase cloud continuity | both |
| 7 | Whether the SEEKING circuit drives T1, or the claim is restated | `mecanumbot_behaviours`, `mecanumbot` |
