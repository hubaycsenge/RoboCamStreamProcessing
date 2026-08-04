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
