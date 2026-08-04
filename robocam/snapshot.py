"""Occasional frame dumps to disk.

The point is verification, not recording: when the server sits behind an sshfs
mount, being able to open ``snapshots/latest.jpg`` from your laptop is the
fastest way to confirm that real images — right way up, right colour order — are
arriving from the robot.

Writes happen on their own thread with a single-slot latest-wins mailbox, so a
slow filesystem (and sshfs is a slow filesystem) can never add jitter to the IO
loop.  The sensor overlays are drawn on that same thread for the same reason:
they are a few milliseconds of OpenCV per snapshot, which is nothing every 150
frames but is not something to put in the path of every frame.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .decode import encode_jpeg
from .imu import ImuBatch
from .lidar import Scan

log = logging.getLogger(__name__)


class SnapshotWriter:
    def __init__(
        self,
        directory: str | Path,
        every_n_frames: int = 150,
        latest_only: bool = True,
        jpeg_quality: int = 85,
        enabled: bool = True,
        lidar_overlay: bool = True,
        hfov_deg: float = 70.0,
        mount_yaw_deg: float = 0.0,
        fov_bins: int = 32,
        imu_overlay: bool = True,
    ) -> None:
        self.dir = Path(directory)
        self.every_n = int(every_n_frames)
        self.latest_only = bool(latest_only)
        self.quality = int(jpeg_quality)
        self.enabled = bool(enabled) and self.every_n > 0
        self.lidar_overlay = bool(lidar_overlay)
        self.hfov_deg = float(hfov_deg)
        self.mount_yaw_deg = float(mount_yaw_deg)
        self.fov_bins = int(fov_bins)
        self.imu_overlay = bool(imu_overlay)

        self._slot: Optional[Tuple[str, int, np.ndarray, Optional[Scan], Optional[ImuBatch]]] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.written = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="snapshot", daemon=True)
        self._thread.start()
        log.info("snapshots -> %s (every %d frames)", self.dir.resolve(), self.every_n)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def maybe_offer(self, session_id: str, seq: int, image: np.ndarray, frame_index: int,
                    scan: Optional[Scan] = None, imu: Optional[ImuBatch] = None) -> None:
        """Offer a frame; it is written only if it lands on the interval.

        Copies the array, because the caller's buffer may be recycled before the
        writer thread gets to it.  The scan and the burst are not copied — the
        server replaces the whole object each time and never mutates one in place.
        """
        if not self.enabled or frame_index % self.every_n != 0:
            return
        with self._lock:
            self._slot = (session_id, seq, image.copy(), scan, imu)
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                item = self._slot
                self._slot = None
            if item is None:
                continue
            session_id, seq, image, scan, imu = item
            try:
                if scan is not None and self.lidar_overlay:
                    try:
                        from .overlay import draw_scan

                        image = draw_scan(image, scan, hfov_deg=self.hfov_deg,
                                          mount_yaw_deg=self.mount_yaw_deg,
                                          bins=self.fov_bins)
                    except Exception:
                        # A snapshot without the overlay is still worth having;
                        # losing the frame dump because a drawing call objected
                        # to some geometry would be the wrong trade.
                        log.exception("lidar overlay failed for seq=%d", seq)
                if imu is not None and self.imu_overlay:
                    try:
                        from .overlay import draw_imu

                        image = draw_imu(image, imu)
                    except Exception:
                        log.exception("imu overlay failed for seq=%d", seq)
                blob = encode_jpeg(image, self.quality)
                if blob is None:
                    log.warning("snapshot encode failed for seq=%d", seq)
                    continue
                if self.latest_only:
                    target = self.dir / "latest.jpg"
                    # Write-then-rename so a reader never sees a half-written file.
                    tmp = self.dir / ".latest.jpg.tmp"
                    tmp.write_bytes(blob)
                    tmp.replace(target)
                else:
                    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:32]
                    (self.dir / f"{safe}_{seq:08d}.jpg").write_bytes(blob)
                self.written += 1
            except OSError:
                log.exception("snapshot write failed")
