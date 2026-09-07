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

    def __post_init__(self) -> None:
        if self.z_min >= self.z_max:
            raise ConfigError("compare.z_min must be below compare.z_max")
        if self.min_points < 1:
            raise ConfigError("compare.min_points must be >= 1")
        if self.search_cells < 0:
            raise ConfigError("compare.search_cells must be >= 0")


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

    def __post_init__(self) -> None:
        if self.phase not in ("t1", "t2"):
            raise ConfigError(f"mission.phase must be 't1' or 't2', got {self.phase!r}")
        if self.min_exit_width_m < 0.0:
            raise ConfigError("mission.min_exit_width_m must be >= 0")


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
