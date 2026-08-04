"""Camera and LiDAR together: what the frame shows, and how far away it is.

This is the processor to run once the robot has both sensors, and the reference
for how a model should use the scan.  It answers the question the two sensors
can only answer jointly — "there is something at column 400 of the image; how
many metres away is it?" — without any model weights involved, so you can check
the pairing is sane before trusting it under a detector.

The seam for YOLO is deliberately explicit: :meth:`FusionProcessor.range_for_box`
takes a box in image coordinates and returns metres.  Once ``yolo.py`` exists,
the fusion of the two is one call per detection.

Nothing here is a calibration.  Both sensors are assumed to sit on the same
vertical axis looking the same way, up to ``mount_yaw_deg``.  On a TurtleBot-like
platform where the scanner sits some centimetres above and behind the camera,
that assumption is wrong by a parallax that matters at close range and vanishes
by a couple of metres — so the numbers are steering information, not metrology.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .. import lidar as lidar_mod
from .base import Frame, Processor


class FusionProcessor(Processor):
    """Image summary plus the scan projected into the camera's field of view.

    Options
    -------
    hfov_deg:
        Camera horizontal field of view.  Must match the lens, or every bearing
        maps to the wrong column — see the README for the one-minute check.
    mount_yaw_deg:
        Bearing, in the LiDAR's frame, that the camera looks along.
    bins:
        Horizontal slices of the image that get their own range.
    brightness:
        Keep the cheap image statistics from the ``stats`` processor, so this can
        replace it outright rather than being run alongside it.

    These default to None, meaning "take the server's ``lidar:`` config", which
    is where they belong — they describe the robot, not this processor.  Set them
    here only to override for one run.
    """

    name = "fusion"

    def __init__(
        self,
        hfov_deg: Optional[float] = None,
        mount_yaw_deg: Optional[float] = None,
        bins: Optional[int] = None,
        brightness: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(hfov_deg=hfov_deg, mount_yaw_deg=mount_yaw_deg,
                         bins=bins, brightness=brightness, **options)
        # Defaults matching LidarConfig; the server overwrites them at build
        # time via configure() when the operator has not set them explicitly.
        self.hfov_deg = 70.0 if hfov_deg is None else float(hfov_deg)
        self.mount_yaw_deg = 0.0 if mount_yaw_deg is None else float(mount_yaw_deg)
        self.bins = 32 if bins is None else int(bins)
        self.brightness = bool(brightness)
        self._explicit = {
            "hfov_deg": hfov_deg is not None,
            "mount_yaw_deg": mount_yaw_deg is not None,
            "bins": bins is not None,
        }
        self._frames = 0
        self._frames_with_scan = 0

    def configure(self, lidar_cfg: Any) -> None:
        """Adopt the server's LiDAR geometry for anything not set explicitly."""
        if not self._explicit["hfov_deg"]:
            self.hfov_deg = float(lidar_cfg.camera_hfov_deg)
        if not self._explicit["mount_yaw_deg"]:
            self.mount_yaw_deg = float(lidar_cfg.mount_yaw_deg)
        if not self._explicit["bins"]:
            self.bins = int(lidar_cfg.fov_bins)

    def process(self, frame: Frame) -> Dict[str, Any]:
        self._frames += 1
        img = frame.image

        data: Dict[str, Any] = {
            "received": True,
            "width": frame.width,
            "height": frame.height,
            "channels": frame.channels,
            "frames_seen": self._frames,
        }
        if self.brightness:
            sample = img[::4, ::4]
            data["mean"] = round(float(np.mean(sample)), 2)
            data["std"] = round(float(np.std(sample)), 2)
            data["looks_blank"] = bool(data["std"] < 1.0)

        scan = frame.scan
        if scan is None:
            # Say why there is no geometry rather than omitting the key: a robot
            # that silently gets no ranges will happily drive as if the world
            # were empty.  "no scan attached" is actionable; a missing field is
            # indistinguishable from a bug in this processor.
            data["lidar"] = None
            data["lidar_status"] = "no scan attached (no scanner, disabled, or stale)"
            data["scan_coverage"] = 0.0
            return data

        self._frames_with_scan += 1
        # summary is computed once on the IO thread; reuse rather than redo.
        data["lidar"] = dict(scan.summary)
        data["lidar_status"] = "ok"
        data["scan_seq"] = scan.seq
        data["scan_age_ms"] = round(frame.scan_age_ms, 1)
        data["scan_fraction"] = round(self._frames_with_scan / self._frames, 3)

        bins = lidar_mod.project_columns(
            scan, frame.width, hfov_deg=self.hfov_deg,
            bins=self.bins, mount_yaw_deg=self.mount_yaw_deg,
        )
        # None rather than NaN: NaN is not valid JSON and json.dumps emits a
        # bare NaN token that strict parsers on the robot will reject.
        data["fov_bins_m"] = [None if math.isnan(v) else round(float(v), 3) for v in bins]
        data["fov_bin_width_px"] = round(frame.width / self.bins, 1) if self.bins else 0.0
        data["hfov_deg"] = self.hfov_deg
        data["mount_yaw_deg"] = self.mount_yaw_deg

        in_view = bins[np.isfinite(bins)]
        if in_view.size:
            nearest_bin = int(np.nanargmin(bins))
            data["nearest_in_view_m"] = round(float(in_view.min()), 3)
            # Where to look in the image for the closest thing the LiDAR sees.
            data["nearest_in_view_px"] = int((nearest_bin + 0.5) * frame.width / self.bins)
            data["in_view_bins"] = int(in_view.size)
        else:
            data["nearest_in_view_m"] = None
            data["nearest_in_view_px"] = None
            data["in_view_bins"] = 0

        return data

    # -- the seam a detector plugs into -----------------------------------

    def range_for_box(self, frame: Frame, box: List[float], reducer: str = "median") -> Optional[float]:
        """Metres to whatever the scan sees across a detection box.

        ``box`` is ``[x0, y0, x1, y1]`` in image pixels — the vertical extent is
        ignored, because a planar scanner has no opinion about it.  Returns None
        when no beam fell inside the box.
        """
        if frame.scan is None:
            return None
        return lidar_mod.range_for_box(
            frame.scan, float(box[0]), float(box[2]), frame.width,
            hfov_deg=self.hfov_deg, mount_yaw_deg=self.mount_yaw_deg, reducer=reducer,
        )
