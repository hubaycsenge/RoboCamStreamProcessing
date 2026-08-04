"""Drawing a scan onto a snapshot.

The README's third check is "open snapshots/latest.jpg and see whether real
images are arriving".  With a second sensor there is a second question — do the
ranges agree with the picture? — and it has the same cheap answer if the scan is
drawn on the frame.  Two things go on:

* a bird's-eye plot in the corner, robot at the centre looking up, so a corridor
  looks like a corridor and a scanner mounted backwards is obvious immediately;
* a strip along the bottom, one cell per field-of-view bin, aligned with the
  columns above it.  A person standing at the left of the image should colour
  the left of the strip.  If they colour the right, ``mount_yaw_deg`` is wrong
  by 180°, and that is a one-glance diagnosis rather than an afternoon.

All of this runs on the snapshot thread, well away from the IO loop.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .lidar import Scan, project_columns, to_xy

# Blue-green-red ramp for the depth strip: near is red, far is blue.  Chosen so
# that the dangerous end of the scale is the one that draws the eye.
_NEAR = (60, 60, 235)
_MID = (60, 200, 235)
_FAR = (200, 160, 60)


def draw_scan(
    img: np.ndarray,
    scan: Scan,
    hfov_deg: float = 70.0,
    mount_yaw_deg: float = 0.0,
    bins: int = 32,
    plot_range_m: float = 4.0,
) -> np.ndarray:
    """Return a copy of ``img`` with the scan drawn on it.

    ``plot_range_m`` is the radius of the bird's-eye plot.  4 m rather than the
    device's 12 m because indoor scans are mostly walls at 2–3 m, and a plot
    scaled to the maximum range puts everything interesting in the middle pixel.
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    height, width = out.shape[:2]

    # The bottom of the frame is laid out once, here, so the strip and the
    # caption cannot end up drawn on top of each other.
    caption_h = 22
    strip_h = max(10, height // 28)
    strip_top = max(0, height - caption_h - strip_h)

    _draw_depth_strip(out, scan, hfov_deg, mount_yaw_deg, bins, strip_top, strip_h)
    _draw_birdseye(out, scan, hfov_deg, mount_yaw_deg, plot_range_m)

    summary = scan.summary or {}
    nearest = summary.get("nearest_m")
    label = "lidar: no returns" if nearest is None else (
        "lidar: nearest %.2f m @%+.0f deg | cover %.0f%% | %.1f Hz"
        % (nearest, summary.get("nearest_deg", 0.0),
           100 * summary.get("coverage", 0.0), summary.get("hz", 0.0))
    )
    if summary.get("obstacle"):
        label += " | OBSTACLE"
    _caption(out, label, height - caption_h, caption_h)
    return out


def _draw_depth_strip(img: np.ndarray, scan: Scan, hfov_deg: float, mount_yaw_deg: float,
                      bins: int, top: int, strip_h: int) -> None:
    """One cell per FOV bin along the bottom, aligned with the columns above."""
    height, width = img.shape[:2]
    ranges = project_columns(scan, width, hfov_deg=hfov_deg, bins=bins,
                             mount_yaw_deg=mount_yaw_deg)
    if ranges.size == 0:
        return

    far = max(1.0, float(np.nanmax(ranges)) if np.isfinite(ranges).any() else 1.0)
    cell_w = width / bins
    # Only label the cells if the numbers will actually fit in them.
    label = cell_w >= 30 and strip_h >= 12
    for i, value in enumerate(ranges):
        x0 = int(i * cell_w)
        x1 = int((i + 1) * cell_w)
        if math.isnan(value):
            # No return in this direction: leave it dark rather than picking a
            # colour, so "unknown" never looks like a measurement.
            cv2.rectangle(img, (x0, top), (x1 - 1, top + strip_h), (40, 40, 40), -1)
            continue
        cv2.rectangle(img, (x0, top), (x1 - 1, top + strip_h), _ramp(value / far), -1)
        if label and value < 100.0:
            cv2.putText(img, "%.1f" % value, (x0 + 3, top + strip_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_birdseye(img: np.ndarray, scan: Scan, hfov_deg: float,
                   mount_yaw_deg: float, plot_range_m: float) -> None:
    """Top-down plot in the top-right corner, robot at centre looking up."""
    height, width = img.shape[:2]
    size = max(96, min(width, height) // 3)
    pad = 10
    x0, y0 = width - size - pad, pad

    panel = np.full((size, size, 3), 24, dtype=np.uint8)
    cx = cy = size // 2
    scale = (size / 2 - 6) / max(plot_range_m, 0.1)

    # Range rings every metre, so distances are readable without a legend.
    for metres in range(1, int(plot_range_m) + 1):
        cv2.circle(panel, (cx, cy), int(metres * scale), (60, 60, 60), 1, cv2.LINE_AA)

    # The camera's field of view, so it is obvious which returns the image can
    # possibly show and which are behind the robot.
    half = math.radians(hfov_deg) / 2.0
    for sign in (-1, 1):
        end = (int(cx - math.sin(sign * half) * size), int(cy - math.cos(sign * half) * size))
        cv2.line(panel, (cx, cy), end, (90, 90, 90), 1, cv2.LINE_AA)

    xs, ys = to_xy(scan, mount_yaw_deg=mount_yaw_deg)
    if xs.size:
        # Screen: up is forward (+x), left is +y.  Points beyond the plot radius
        # are dropped rather than clamped to the rim, where they would read as a
        # wall that is not there.
        px = (cx - ys * scale).astype(np.int32)
        py = (cy - xs * scale).astype(np.int32)
        keep = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        for x, y, r in zip(px[keep], py[keep], np.hypot(xs, ys)[keep]):
            cv2.circle(panel, (int(x), int(y)), 1, _ramp(r / max(plot_range_m, 0.1)), -1)

    # The robot, and its heading.
    cv2.circle(panel, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.line(panel, (cx, cy), (cx, cy - 12), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{plot_range_m:.0f}m", (4, size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

    roi = img[y0:y0 + size, x0:x0 + size]
    if roi.shape[:2] != (size, size):  # image smaller than the panel
        return
    cv2.addWeighted(panel, 0.75, roi, 0.25, 0, dst=roi)
    cv2.rectangle(img, (x0, y0), (x0 + size - 1, y0 + size - 1), (200, 200, 200), 1)


def _ramp(t: float):
    """Near (0) to far (1) colour, in BGR."""
    t = min(max(float(t), 0.0), 1.0)
    if t < 0.5:
        a, b, u = _NEAR, _MID, t * 2.0
    else:
        a, b, u = _MID, _FAR, (t - 0.5) * 2.0
    return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))


def _caption(img: np.ndarray, text: str, top: int, band_h: int) -> None:
    """White text on its own solid band across the bottom of the frame.

    A band rather than an outlined font: at this size a thick outline closes up
    the counters of the glyphs and the line becomes unreadable, which is a poor
    outcome for the one line that says whether the scanner is working.
    """
    height, width = img.shape[:2]
    cv2.rectangle(img, (0, top), (width, height), (20, 20, 20), -1)
    cv2.putText(img, text, (8, top + band_h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (245, 245, 245), 1, cv2.LINE_AA)


def maybe_draw(img: np.ndarray, scan: Optional[Scan], **kwargs) -> np.ndarray:
    """``draw_scan`` when there is a scan, the image untouched when there is not."""
    if scan is None:
        return img
    return draw_scan(img, scan, **kwargs)
