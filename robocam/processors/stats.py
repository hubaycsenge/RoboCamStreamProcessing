"""The default processor: confirm the frame arrived and describe it.

This is deliberately the whole job for now — it answers "is the image influx
actually happening, and what shape is what I am receiving?" without any model
weights involved.  It is also a useful permanent smoke test: point the robot at
this processor to check the link before switching to a heavy model.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .base import Frame, Processor


class StatsProcessor(Processor):
    """Reports geometry, a rate estimate and a cheap content summary.

    Options
    -------
    brightness:
        Include mean/std of the image.  Costs a full pass over the pixels
        (~1 ms for 720p), and is the quickest way to tell a live camera from a
        lens cap or a frozen buffer.
    checksum:
        Include a cheap hash of the pixel data.  Useful for spotting a stream
        that is re-sending one identical frame.
    lidar:
        Include the summary of the scan attached to the frame, when there is
        one.  Free — the server computed it on the IO thread — and it answers
        "is the LiDAR arriving too?" from the same result line that answers the
        question for the camera.  Use the ``fusion`` processor for anything that
        relates the ranges to the image.
    imu:
        Same for the inertial burst from the OpenCR: attitude, rates and whether
        the robot is moving, already reduced on the IO thread.
    """

    name = "stats"

    def __init__(self, brightness: bool = True, checksum: bool = False,
                 lidar: bool = True, imu: bool = True, **options: Any) -> None:
        super().__init__(brightness=brightness, checksum=checksum, lidar=lidar,
                         imu=imu, **options)
        self.brightness = bool(brightness)
        self.checksum = bool(checksum)
        self.lidar = bool(lidar)
        self.imu = bool(imu)
        self._count = 0
        self._scans = 0
        self._bursts = 0
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._last_interval_ms = 0.0

    def setup(self) -> None:
        self._count = 0
        self._scans = 0
        self._bursts = 0
        self._first_ts = None
        self._last_ts = None

    def process(self, frame: Frame) -> Dict[str, Any]:
        now = time.monotonic()
        if self._first_ts is None:
            self._first_ts = now
        if self._last_ts is not None:
            self._last_interval_ms = (now - self._last_ts) * 1000.0
        self._last_ts = now
        self._count += 1

        img = frame.image
        elapsed = now - self._first_ts

        data: Dict[str, Any] = {
            # The headline answer to "did a frame get through?".
            "received": True,
            "shape": list(img.shape),
            "width": frame.width,
            "height": frame.height,
            "channels": frame.channels,
            "dtype": str(img.dtype),
            "nbytes": int(img.nbytes),
            "payload_bytes": frame.payload_bytes,
            # Compression ratio achieved on the wire; a sanity check on codec settings.
            "compression_ratio": round(img.nbytes / frame.payload_bytes, 2) if frame.payload_bytes else 0.0,
            "frames_seen": self._count,
            "session_fps": round(self._count / elapsed, 2) if elapsed > 0.5 else 0.0,
            "since_prev_ms": round(self._last_interval_ms, 2),
        }

        if self.brightness:
            # Sample rather than reduce the whole array: a stride of 4 in each
            # axis is 16x less work and tells you the same thing.
            sample = img[::4, ::4]
            data["mean"] = round(float(np.mean(sample)), 2)
            data["std"] = round(float(np.std(sample)), 2)
            # A dead camera gives a near-zero std; so does a lens cap.
            data["looks_blank"] = bool(data["std"] < 1.0)

        if self.checksum:
            # Not cryptographic, just enough to notice a repeated frame.
            data["checksum"] = format(int(np.sum(img[::8, ::8].astype(np.int64))) & 0xFFFFFFFF, "08x")

        if self.lidar:
            if frame.scan is None:
                # An explicit None, not a missing key: the difference between
                # "no scanner on this robot" and "this processor forgot to look"
                # is exactly what you are debugging when you read this field.
                data["lidar"] = None
            else:
                self._scans += 1
                data["lidar"] = dict(frame.scan.summary)
                data["lidar"]["scan_seq"] = frame.scan.seq
                data["lidar"]["age_ms"] = round(frame.scan_age_ms, 1)
                # How often a frame arrives with ranges to go with it.  Well
                # below 1.0 is expected — the LiDAR runs at a sixth of the
                # camera's rate — but near 0.0 means the pairing window in
                # lidar.stale_after_ms is too tight for the scanner's speed.
                data["scan_fraction"] = round(self._scans / self._count, 3)

        if self.imu:
            if frame.imu is None:
                # Explicit None for the same reason as the scan: "this robot has
                # no IMU" and "the burst went stale" are both actionable, and a
                # missing key is neither.
                data["imu"] = None
            else:
                self._bursts += 1
                data["imu"] = dict(frame.imu.summary)
                data["imu"]["imu_seq"] = frame.imu.seq
                data["imu"]["age_ms"] = round(frame.imu_age_ms, 1)
                # Should sit near 1.0: the IMU runs several times faster than the
                # camera, so every frame ought to find a burst waiting.  Well
                # below 1.0 means bursts are not arriving, not that they are rare.
                data["imu_fraction"] = round(self._bursts / self._count, 3)

        return data
