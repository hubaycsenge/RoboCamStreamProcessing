"""Configuration loading.

Everything the server does is driven by a YAML file (see ``config/server.yaml``).
Values are plain dataclasses so that a typo in the YAML fails loudly at startup
rather than silently at frame 10000.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    pass


@dataclass
class ServerConfig:
    # ZeroMQ endpoint to bind.  0.0.0.0 so the robot on the LAN can reach it.
    bind: str = "tcp://0.0.0.0:5555"
    # How long the IO loop blocks in poll() before doing housekeeping.
    io_poll_ms: int = 20
    # A session with no traffic for this long is reaped.
    session_timeout_s: float = 20.0
    # Refuse payloads larger than this.  Guards against a corrupt length field
    # turning into a multi-GB allocation.
    max_payload_bytes: int = 32 * 1024 * 1024
    # ZeroMQ receive high-water mark, in messages.
    rcvhwm: int = 64
    sndhwm: int = 64


@dataclass
class QueueConfig:
    # Frames buffered between the IO thread and the workers.  Keep this small:
    # for a robot, a fresh frame is worth more than a complete history.
    max_depth: int = 2
    # Which frame to discard when the queue is full: "oldest" keeps latency low,
    # "newest" preserves ordering at the cost of staleness.
    drop_policy: str = "oldest"

    def __post_init__(self) -> None:
        if self.drop_policy not in ("oldest", "newest"):
            raise ConfigError(f"queue.drop_policy must be 'oldest' or 'newest', got {self.drop_policy!r}")
        if self.max_depth < 1:
            raise ConfigError("queue.max_depth must be >= 1")


@dataclass
class ProcessorConfig:
    # Name registered in robocam.processors.REGISTRY.
    name: str = "stats"
    # Worker threads pulling from the queue.  One is right for a GPU model;
    # more only helps for genuinely parallel CPU work.
    workers: int = 1
    # Passed verbatim to the processor constructor.
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ConfigError("processor.workers must be >= 1")


@dataclass
class LidarConfig:
    """How to interpret the robot's LDS-02 scans.

    The defaults describe an LDS-02 mounted looking the same way as the camera.
    Two of these are mounting facts about *your* robot rather than tastes, and
    getting them wrong makes every fused number quietly wrong rather than
    obviously wrong: ``mount_yaw_deg`` and ``camera_hfov_deg``.  See the README
    for how to check them in one minute with a snapshot.
    """

    enabled: bool = True
    # Bearing, in the LiDAR's own frame, that the camera looks along.  0 means
    # the scanner's zero and the optical axis point the same way; 180 means the
    # scanner is mounted backwards, which is easy to do and easy to miss.
    mount_yaw_deg: float = 0.0
    # Camera horizontal field of view.  Only used to map bearings to columns.
    camera_hfov_deg: float = 70.0
    # Sectors in the clearance summary.  12 gives 30° each, which is about the
    # granularity a differential-drive robot can act on.
    sectors: int = 12
    # A return closer than this inside the front arc raises data.lidar.obstacle.
    obstacle_m: float = 0.5
    # A direction is "free" at or beyond this range, for the free-direction search.
    clear_m: float = 1.0
    # Narrowest gap reported as a free direction.  Stops a single long reading
    # between two walls from looking like a doorway, and stands in for the fact
    # that the robot has a width.
    min_free_deg: float = 15.0
    # Width of the arc treated as "in front" for obstacle/front_min reporting.
    front_deg: float = 60.0
    # Horizontal slices of the image that get a range in fused results.
    fov_bins: int = 32
    # A scan older than this is not attached to a frame.  At 5 Hz a scan is up
    # to 200 ms old before its successor exists, so anything under ~250 ms means
    # most frames get nothing; too high and the robot fuses a stale world.
    stale_after_ms: float = 400.0

    def __post_init__(self) -> None:
        if self.sectors < 1:
            raise ConfigError("lidar.sectors must be >= 1")
        if self.fov_bins < 1:
            raise ConfigError("lidar.fov_bins must be >= 1")
        if not 0.0 < self.camera_hfov_deg < 180.0:
            raise ConfigError("lidar.camera_hfov_deg must be in (0, 180)")
        if self.front_deg <= 0.0 or self.front_deg > 360.0:
            raise ConfigError("lidar.front_deg must be in (0, 360]")


@dataclass
class ImuConfig:
    """How to interpret the bursts coming off the robot's OpenCR board.

    Unlike the LiDAR section there is no mounting geometry here, because the IMU
    is bolted to the chassis and reports the chassis's own motion: nothing has to
    be related to the camera.  What is here is thresholds — the points at which
    "the robot is turning", "something hit it" and "it is leaning over" become
    true — and they are robot-specific in a way the defaults cannot be.
    """

    enabled: bool = True
    # Angular rate above which the robot counts as turning.  Just above the
    # noise floor of a MEMS gyro at rest, which is a degree or two per second.
    still_gyro_dps: float = 2.0
    # Spread of specific force above which it counts as shaking.  A robot
    # driving over a hard floor sits around 0.2-0.5 m/s²; a stationary one is
    # an order of magnitude below that.
    still_accel_ms2: float = 0.35
    # Tilt beyond which the summary says so.  15° is a slope this robot should
    # not be on, not the angle it tips over at.
    tilt_warn_deg: float = 15.0
    # Peak specific force treated as an impact.  2.5 g — well clear of driving
    # over a cable, well below what a fall produces.
    shock_ms2: float = 25.0
    # How far the mean specific force may sit from gravity before the burst is
    # flagged implausible.  Catches a units mistake or a dead axis, which are
    # the two failures that otherwise produce confident nonsense.
    gravity_tolerance_ms2: float = 2.0
    # A burst older than this is not attached to a frame.  Much tighter than the
    # LiDAR's window because the IMU runs 20x faster: if the newest inertial
    # data is 150 ms old, something is wrong rather than merely slow.
    stale_after_ms: float = 150.0

    def __post_init__(self) -> None:
        if self.still_gyro_dps < 0.0:
            raise ConfigError("imu.still_gyro_dps must be >= 0")
        if self.still_accel_ms2 < 0.0:
            raise ConfigError("imu.still_accel_ms2 must be >= 0")
        if not 0.0 < self.tilt_warn_deg <= 180.0:
            raise ConfigError("imu.tilt_warn_deg must be in (0, 180]")
        if self.shock_ms2 <= 0.0:
            raise ConfigError("imu.shock_ms2 must be > 0")


@dataclass
class OdomConfig:
    """How the server treats the robot's odometry stream.

    There is almost nothing to configure, and that is the point: a pose is not
    interpreted, it is *used* — to place a reconstruction in the frame the map is
    drawn in.  What is here is the freshness window and the one threshold that
    decides whether the robot counts as standing still.
    """

    enabled: bool = True
    # A pose older than this is not attached to a frame.  Between the LiDAR's
    # window and the IMU's: odometry updates at roughly the frame rate, and a
    # 200 ms old pose belongs to a robot that has moved ~6 cm at this robot's
    # cruising speed -- about one map cell, which is the error budget.
    stale_after_ms: float = 200.0
    # Below this speed and yaw rate the robot counts as still.  Only reported,
    # never acted on here; it is what tells a reader of the log that a frame
    # gave the reconstruction no new parallax.
    still_speed_ms: float = 0.02
    still_yaw_rate_dps: float = 2.0
    # The frame odometry is expected to arrive in.  "map" is what makes a
    # comparison against the SLAM grid meaningful; a robot sending "odom" is
    # sending dead reckoning, and the mismatch is refused rather than absorbed.
    # Empty accepts whatever arrives and leaves the check to compare.py.
    expect_frame: str = "map"

    def __post_init__(self) -> None:
        if self.stale_after_ms < 0.0:
            raise ConfigError("odom.stale_after_ms must be >= 0")


@dataclass
class MapConfig:
    """How the server treats the robot's 2D occupancy grid."""

    enabled: bool = True
    # Refuse a grid larger than this.  4 million cells is a 1 km square at 5 cm,
    # or a 100 m square at 5 mm: past anything this robot maps, and a bound on
    # what a corrupt width field can allocate.
    max_cells: int = 4_000_000
    # A grid older than this is stale for comparison purposes.  Much longer than
    # the sensor windows because a map is not perishable in the same way: a
    # room does not change in ten seconds, whereas the robot's pose in it does.
    stale_after_ms: float = 10_000.0
    # Send map_update patches back to the robot.  Turning this off leaves the
    # comparison running and reported -- useful for checking what the server
    # *would* send before letting it write to the robot's costmap.
    send_updates: bool = True
    # Smallest patch worth a message, in cells.  Below this the news is single
    # frame reconstruction noise more often than it is furniture.
    min_update_cells: int = 4
    # Value written into cells the comparison newly calls occupied.  100 is
    # "certainly occupied" in the ROS scale; lower it to let the robot's own
    # costmap inflation treat these as weaker evidence than its scanner's.
    occupied_value: int = 100
    # How the robot should merge a patch: "max" (add obstacles, never clear) or
    # "replace".  See robocam/occupancy.py for why max is the default and why
    # replace needs a reason.
    merge: str = "max"
    # No more than one patch per this many milliseconds.  The reconstruction
    # produces a cloud several times a second and the robot's costmap does not
    # need to be rewritten at that rate; the cost of the message is small, the
    # cost of the merge on the robot is not.
    min_update_interval_ms: float = 500.0

    def __post_init__(self) -> None:
        if self.max_cells < 1:
            raise ConfigError("map.max_cells must be >= 1")
        if self.merge not in ("max", "replace"):
            raise ConfigError(f"map.merge must be 'max' or 'replace', got {self.merge!r}")
        if not 0 <= self.occupied_value <= 100:
            raise ConfigError("map.occupied_value must be in [0, 100]")


@dataclass
class CompareConfig:
    """The comparison of the reconstruction against the robot's grid.

    The height slice and the camera mount are the two groups that describe
    physical facts, and both are worth measuring rather than guessing: the slice
    decides which surfaces count as obstacles, and the mount decides where the
    cloud lands.  A wrong mount produces a comparison that runs perfectly and
    reports agreement near zero -- which is why compare.py makes that number the
    first one in its output.
    """

    enabled: bool = True

    # --- the height slice, metres above the map's floor plane ---
    # Above the floor, so the floor itself is not an obstacle.  5 cm is roughly
    # the lip this robot can drive over.
    z_min: float = 0.05
    # Below the ceiling, and below anything the robot passes under.  1.6 m is
    # well above the Mecanumbot with its mast; lower it if the robot is shorter
    # than the things it is allowed to drive beneath.
    z_max: float = 1.6
    # Points needed in a cell before it counts as a surface.  One is a pixel of
    # a monocular depth estimate; three in a 5 cm cell is a patch of something.
    min_points: int = 3

    # --- where the camera is bolted, in the base frame (metres, radians) ---
    # REP-103: x forward, y left, z up.  pitch is positive nose-down.  On this
    # robot the camera is on a neck motor, so pitch is whatever the neck is
    # commanded to -- these defaults describe it looking level.
    camera_x: float = 0.10
    camera_y: float = 0.0
    camera_z: float = 0.45
    camera_roll: float = 0.0
    camera_pitch: float = 0.0
    camera_yaw: float = 0.0

    # --- the pose hint ---
    # Offer SLAM a correction when the cloud and the map agree better at an
    # offset than at zero.  Off by default: it is an inference from a histogram
    # against a robot that has a pose graph, and it should be switched on
    # deliberately once agreement has been seen to be healthy.
    send_hints: bool = False
    # Half-width of the offset search, in cells.  4 cells at 5 cm is +/-20 cm,
    # which is the drift this is meant to catch; a wider search mostly buys
    # opportunities to lock onto a corridor wall at the wrong offset.
    search_cells: int = 4
    # No more than one hint per this many milliseconds.  A correction the robot
    # has not had time to apply is a correction that will be measured again.
    min_hint_interval_ms: float = 2000.0

    # --- the agreement verdict (the `agreement` announcement) ---
    # Send it at all.  Off means the comparison still runs and is logged, but
    # the robot never hears the verdict -- which also means its T1 `CLOUD` exit
    # criterion can never be satisfied.  See docs/INTEGRATION.md.
    send_agreement: bool = True
    # One verdict per this many milliseconds.  The reconstruction runs at ~6 Hz;
    # the robot's exploration decides things at walking pace, and a verdict it
    # has not acted on yet is one that will be measured again.
    # 1 Hz was the first guess and it was too fast: the robot's exploration
    # decides where to drive at walking pace, and each verdict makes it
    # re-evaluate its keepouts and its revisit list. 3 s is still several
    # verdicts per goal.
    min_agreement_interval_ms: float = 3000.0
    # Side of the square the disagreement is pooled into before clustering, in
    # metres.  Regions are places to drive to, so this is roughly the smallest
    # thing worth a detour; below about 0.3 m the list fills with reconstruction
    # noise and the robot spends T1 visiting specks.
    region_block_m: float = 0.5
    # Disagreeing cells a block needs before it counts.  The same speck filter
    # `min_new_cells` applies to the patch, at block scale.
    region_min_cells: int = 3
    # Cap on the region list.  It rides in an announcement on the same socket as
    # the frames, and the lowest-scoring are what the cap drops.
    # 32 was a cap, not a considered number. The robot interleaves one revisit
    # goal per `uncertain_every` frontier goals, so a list far longer than the
    # goals it will ever service is churn with no benefit -- and the low-scoring
    # tail is exactly what changes between verdicts.
    max_regions: int = 12

    def __post_init__(self) -> None:
        if self.z_min >= self.z_max:
            raise ConfigError("compare.z_min must be below compare.z_max")
        if self.min_points < 1:
            raise ConfigError("compare.min_points must be >= 1")
        if self.search_cells < 0:
            raise ConfigError("compare.search_cells must be >= 0")
        if self.region_block_m <= 0.0:
            raise ConfigError("compare.region_block_m must be > 0")
        if self.region_min_cells < 1:
            raise ConfigError("compare.region_min_cells must be >= 1")
        if self.max_regions < 0:
            raise ConfigError("compare.max_regions must be >= 0")


@dataclass
class MissionConfig:
    """The phases, the exits, and what T2 is looking for."""

    # Which row of the system diagram a session starts on when its hello does
    # not say.  t1 is explore-and-map; t2 is seek.
    phase: str = "t1"
    # Free text handed to whatever runs the decision stage in t2.  Empty means
    # the robot is expected to declare it in its hello, which is the normal case
    # -- the mission belongs to the run, not to the server's config file.
    target: str = ""
    # Rank exit candidates and answer them.  Off leaves the robot's own ordering
    # untouched, which is what you want while the ranking is being developed.
    rank_exits: bool = True
    # Narrowest gap the ranking will recommend driving through.  The Mecanumbot
    # is 28 cm across the wheels; 60 cm leaves room for the fact that a frontier
    # width is measured on a grid, not with calipers.
    min_exit_width_m: float = 0.6
    # Metres per radian charged against an exit for having to turn to face it.
    # A tie-breaker at this robot's driving and turning speeds, not a model.
    turn_cost_m_per_rad: float = 0.5
    # Standoff for the approach pose in a found announcement: how far short of
    # the target the robot should stop.  Its reach plus its radius.
    approach_standoff_m: float = 0.8

    # -- when T1 is finished ------------------------------------------------
    # Evaluated on every uploaded grid and reported in the map reply; the server
    # never changes phase on its own.  See robocam.seek.t1_exit_criteria for
    # what each test is for and why no one of them suffices alone.
    #
    # Report the verdict at all.  Off makes the whole evaluation a no-op, for a
    # run whose phases are being driven by hand.
    t1_exit_report: bool = True
    # Share of the grid that must have been observed.  Never 1.0: the grid is a
    # rectangle and rooms are not, so some of it is permanently behind a wall.
    t1_min_explored: float = 0.80
    # Open frontier candidates, at least min_exit_width_m wide, still to visit.
    t1_max_open_exits: int = 0
    # Newly observed cells per second below which the map has stopped growing.
    t1_stall_cells_per_s: float = 15.0
    # How long that rate must have been measurable before it counts.  Shorter
    # than this and a robot pausing to turn looks like a finished map.
    t1_min_stall_window_s: float = 30.0
    # Mean cloud/grid agreement.  T1's product is the cloud, so finishing with a
    # good map and a cloud that was never correctly placed leaves T2 nothing to
    # search -- which surfaces an hour later as "the detector never sees
    # anything" and is very hard to trace back to here.
    t1_min_agreement: float = 0.15
    # Floors, so that a first grid arriving before the robot has moved cannot
    # satisfy every other test at once.
    t1_min_frames: int = 300
    t1_min_runtime_s: float = 60.0

    def __post_init__(self) -> None:
        if self.phase not in ("t1", "t2"):
            raise ConfigError(f"mission.phase must be 't1' or 't2', got {self.phase!r}")
        if self.min_exit_width_m < 0.0:
            raise ConfigError("mission.min_exit_width_m must be >= 0")
        if not 0.0 <= self.t1_min_explored <= 1.0:
            raise ConfigError("mission.t1_min_explored must be between 0 and 1")


@dataclass
class SeekConfig:
    """The decision stage: what T2 looks for, and what it can pick up.

    Split from ``mission`` on purpose.  ``mission`` is what this run is about —
    the phase, the target, the exits — and changes per run; this is what the
    robot and its detector *are*, and changes when the hardware does.
    """

    # Run the decision stage at all.  Off leaves T2 a navigation phase with no
    # target search, which is what you want while the detector is being fitted.
    enabled: bool = True

    # -- the detector -------------------------------------------------------
    # "colour" needs no weights and matches a colour word in the target text: it
    #     exists so the geometry after the box can be exercised end to end
    #     without a checkpoint, in the same spirit as the "stats" processor.
    # "owl"    is OWLv2 through transformers: real open-vocabulary detection,
    #     the target text used as the query. This is the one that actually seeks.
    # "none"   finds nothing, for measuring T2 without the detector's cost.
    detector: str = "colour"
    # Passed verbatim to the detector's constructor.  For "owl": model, device,
    # score_min, max_detections.
    detector_options: Dict[str, Any] = field(default_factory=dict)
    # Run the detector on one frame in N.  The detector is the expensive half of
    # T2 and the target does not move at the frame rate; skipping in the stage is
    # better than letting the queue evict, for the same reason deep3r's every_n
    # is -- the frames that run are chosen rather than whichever ones happened to
    # arrive between forward passes.
    detect_every_n: int = 2

    # -- believing a detection ----------------------------------------------
    # Score below which a detection is not acted on.  The detector's own floor is
    # lower and deliberately so: it returns candidates, this decides.
    min_confidence: float = 0.25
    # Cloud points that must survive confidence and depth filtering inside the
    # box before a coordinate is believed.  Six points is a guess; the default is
    # the point at which a centroid means something.
    min_points: int = 8
    # Confidence gate on the points inside the box, on CUT3R's conf_self scale
    # (which starts at 1.0, not 0.0).  Looser than the cloud's own filter,
    # because a small object at range is exactly where the model is less certain
    # and dropping it entirely loses the target rather than a bit of a wall.
    min_point_conf: float = 1.5
    # Half-width of the depth band kept around the box's median range.  This is
    # what stops a bounding box's background dragging the coordinate to the wall
    # behind the object; 25 cm keeps a mug and drops the desk behind it.
    depth_band_m: float = 0.25
    # Fraction of each side of the box kept when sampling points.  The corners of
    # a bounding box are background and are also where a monocular
    # reconstruction is least reliable.
    box_keep_frac: float = 0.7

    # -- announcing ---------------------------------------------------------
    # Smallest gap between two "found" announcements, milliseconds.  Unlike the
    # map updates this is generous rather than tight: the robot acts on a found,
    # and a stream of them at the frame rate is a behaviour tree that never
    # finishes reacting to the first.
    min_found_interval_ms: float = 1000.0
    # Announce a "found: false" after this many consecutive detector runs in t2
    # with nothing.  It is what lets the robot stop waiting and start searching
    # rather than seeking until it times out.  0 disables.
    absent_after_runs: int = 30
    # Re-announce the same live sighting when it has moved by this much, even
    # inside the interval above.  A target that has actually moved is news.
    resend_move_m: float = 0.35

    # -- the gripper, in metres: tape-measure facts about this robot ---------
    # THE number to get right.  The Mecanumbot's gripper closes at floor level,
    # so an object higher than this is not collectable however close it drives.
    # Knowing that before driving is the whole value of having a reconstruction
    # rather than a scanner, which cannot tell the floor from the table at all.
    #
    # These two are SHARED WITH THE ROBOT and must match
    # `seek_grasp_height_min` / `_max` in mecanumbot_seek's constants file. The
    # shafts sit at z ~ 0.034 with a 0.116 m clear gap and there is no lift DOF,
    # which is where both numbers come from. They disagreed until 2026-09-08.
    grasp_z_max: float = 0.15
    # Bottom of the same gap: above the floor but under the shafts is a real
    # object in a recess, reported as `too_low`.
    grasp_z_min: float = 0.03
    # Below the map's floor plane is a reconstruction error, not an object.
    # Slightly negative because the floor plane is a plane and a real floor is
    # not. NOT the bottom of the grasp band -- these were one field, and
    # conflating "the model produced a point in the basement" with "the mug is
    # in a recess" loses the difference between a bug and something to tell a
    # person about.
    floor_tolerance_m: float = -0.05
    # How far from the base centre the gripper closes.  Used to ask whether any
    # standing position puts the object in range. Matches `seek_grasp_distance`
    # on the robot, which is where the tree actually attempts a grasp.
    reach_radius_m: float = 0.30
    # Footprint radius, for deciding whether a candidate standing cell is clear.
    # The Mecanumbot is 28 cm across the wheels; this is that plus a margin.
    robot_radius_m: float = 0.22

    # -- remembering T1 -----------------------------------------------------
    # Keep keyframes during t1 so that a target named at t2 launch can be looked
    # for in what t1 already saw.  This is what produces a goal coordinate before
    # t2 has taken a single frame, and it is off-by-default nowhere: the mission
    # shape in the README depends on it.
    keyframes: bool = True
    # Bound on the memory.  240 frames at the spacing below is a large room.
    keyframe_max: int = 240
    # Minimum movement between kept keyframes.  Thinned by distance rather than
    # time because a parked robot produces sixty identical frames and none of
    # them is new evidence.
    keyframe_spacing_m: float = 0.25
    keyframe_spacing_rad: float = 0.35
    # Keyframes searched per incoming frame when t2 starts.  The retro-search
    # runs against the whole store and must not stall the stream, so it is spread
    # over the frames that arrive while it runs.
    recall_budget_per_frame: int = 3
    # How long a remembered coordinate stays worth driving to.
    memory_ttl_s: float = 900.0

    def __post_init__(self) -> None:
        if self.detect_every_n < 1:
            raise ConfigError("seek.detect_every_n must be >= 1")
        if self.grasp_z_max <= self.grasp_z_min:
            raise ConfigError(
                f"seek.grasp_z_max ({self.grasp_z_max}) must be above grasp_z_min "
                f"({self.grasp_z_min}); as written the gripper envelope is empty and "
                "nothing would ever be reachable"
            )
        if self.floor_tolerance_m > self.grasp_z_min:
            raise ConfigError(
                f"seek.floor_tolerance_m ({self.floor_tolerance_m}) must be at or "
                f"below grasp_z_min ({self.grasp_z_min}); above it, an object in a "
                "recess is reported as a reconstruction error instead of as "
                "something to tell a person about"
            )
        if self.min_points < 1:
            raise ConfigError("seek.min_points must be >= 1")
        if not 0.0 < self.box_keep_frac <= 1.0:
            raise ConfigError("seek.box_keep_frac must be in (0, 1]")


@dataclass
class SnapshotConfig:
    """Periodically write a decoded frame to disk.

    Cheap way to confirm from the far end of an sshfs mount that real images are
    arriving, and that they are the right way up and not colour-swapped.
    """

    enabled: bool = True
    dir: str = "snapshots"
    # Write one snapshot every N frames.  0 disables.
    every_n_frames: int = 150
    # Overwrite a single file instead of accumulating numbered ones.
    latest_only: bool = True
    jpeg_quality: int = 85
    # Draw the attached scan on the snapshot: a bird's-eye plot plus a depth
    # strip aligned to the image columns.  This is how you check the LiDAR
    # agrees with what the camera sees without writing any code.
    lidar_overlay: bool = True
    # Draw the attached IMU burst: an attitude disc in the opposite corner.  A
    # robot on a flat floor must show a level horizon, which is how you check
    # the board's mounting without writing any code either.
    imu_overlay: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    # Emit a throughput summary this often.  0 disables.
    stats_interval_s: float = 5.0


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    odom: OdomConfig = field(default_factory=OdomConfig)
    map: MapConfig = field(default_factory=MapConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    seek: SeekConfig = field(default_factory=SeekConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")

        kwargs: Dict[str, Any] = {}
        for name, factory in (
            ("server", ServerConfig),
            ("queue", QueueConfig),
            ("processor", ProcessorConfig),
            ("lidar", LidarConfig),
            ("imu", ImuConfig),
            ("odom", OdomConfig),
            ("map", MapConfig),
            ("compare", CompareConfig),
            ("mission", MissionConfig),
            ("seek", SeekConfig),
            ("snapshot", SnapshotConfig),
            ("logging", LoggingConfig),
        ):
            section = raw.get(name) or {}
            if not isinstance(section, dict):
                raise ConfigError(f"config section '{name}' must be a mapping")
            valid = {f.name for f in dataclasses.fields(factory)}
            bad = set(section) - valid
            if bad:
                raise ConfigError(
                    f"unknown key(s) in '{name}': {', '.join(sorted(bad))}. "
                    f"valid keys: {', '.join(sorted(valid))}"
                )
            kwargs[name] = factory(**section)
        return cls(**kwargs)
