"""Drawing the other sensors onto a snapshot.

The README's third check is "open snapshots/latest.jpg and see whether real
images are arriving".  With more sensors there are more questions — do the ranges
agree with the picture?  does the board agree about which way is up? — and they
have the same cheap answer if the data is drawn on the frame.

From the scan:

* a bird's-eye plot in the corner, robot at the centre looking up, so a corridor
  looks like a corridor and a scanner mounted backwards is obvious immediately;
* a strip along the bottom, one cell per field-of-view bin, aligned with the
  columns above it.  A person standing at the left of the image should colour
  the left of the strip.  If they colour the right, ``mount_yaw_deg`` is wrong
  by 180°, and that is a one-glance diagnosis rather than an afternoon.

From the IMU:

* an attitude disc in the opposite corner.  A robot standing on a flat floor
  must show a level horizon; anything else means the OpenCR is not mounted the
  way the axis convention assumes, and every attitude number is rotated with it.
  That is the same class of mistake as ``mount_yaw_deg``, and it is caught the
  same way — by looking, rather than by reasoning about signs.

All of this runs on the snapshot thread, well away from the IO loop.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .imu import ImuBatch
from .lidar import Scan, project_columns, to_xy

# Blue-green-red ramp for the depth strip: near is red, far is blue.  Chosen so
# that the dangerous end of the scale is the one that draws the eye.
_NEAR = (60, 60, 235)
_MID = (60, 200, 235)
_FAR = (200, 160, 60)

# Attitude disc, BGR.  Ground warmer than sky so which half is which survives
# being looked at on a phone in a corridor.
_SKY = (150, 110, 60)
_GROUND = (55, 85, 120)


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


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------


def draw_imu(img: np.ndarray, batch: ImuBatch) -> np.ndarray:
    """Return a copy of ``img`` with the burst's attitude drawn on it.

    Top-left, opposite the scan's bird's-eye plot, with its own small caption
    rather than a full-width band — so the two overlays can be drawn on the same
    snapshot without either having to know about the other.
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    summary = batch.summary or {}
    roll, pitch = summary.get("roll_deg"), summary.get("pitch_deg")
    text_top = 10
    if roll is not None and pitch is not None:
        size = _draw_attitude(out, float(roll), float(pitch), summary)
        text_top = 10 + size + 4

    _label(out, _imu_label(batch, summary), 10, text_top)
    return out


def _imu_label(batch: ImuBatch, summary) -> str:
    if not summary:
        return "imu: no summary"
    if summary.get("attitude_from") is None:
        # Samples arriving with nothing to say about attitude: no quaternion and
        # a specific force that does not look like gravity.  Worth naming, since
        # it is what a wrong unit or a dead axis looks like from here.
        return "imu: %d samples, no attitude (accel does not look like gravity)" % summary.get("samples", 0)

    bits = ["imu: roll %+.0f pitch %+.0f" % (summary["roll_deg"], summary["pitch_deg"])]
    if summary.get("yaw_rate_dps") is not None:
        bits.append("yaw %+.0f deg/s" % summary["yaw_rate_dps"])
    bits.append("%.0f Hz" % summary.get("rate_hz", 0.0))
    # The state words, only when they are true, so the line stays short when
    # everything is ordinary.
    for flag, word in (("tilted", "TILTED"), ("shock", "SHOCK"), ("still", "still")):
        if summary.get(flag):
            bits.append(word)
    if not summary.get("gravity_ok", True):
        bits.append("|a|=%.1f NOT GRAVITY" % (summary.get("accel_mag_mean") or 0.0))
    return " | ".join(bits) + ("" if not batch.source else " | " + batch.source)


def _draw_attitude(img: np.ndarray, roll_deg: float, pitch_deg: float, summary) -> int:
    """Artificial horizon in the top-left corner.  Returns the panel size."""
    height, width = img.shape[:2]
    size = max(80, min(width, height) // 4)
    pad = 10
    if height < size + 2 * pad or width < size + 2 * pad:
        return 0

    panel = np.full((size, size, 3), 24, dtype=np.uint8)
    cx = cy = size // 2
    radius = size // 2 - 3
    # 45° from centre to rim: enough range that a robot on a ramp still shows a
    # horizon inside the disc rather than a solid block of ground.
    px_per_deg = radius / 45.0

    a = math.radians(roll_deg)
    along = (math.cos(a), math.sin(a))
    # The disc shows the world as the robot sees it, so the horizon rotates
    # against the roll and drops as the nose comes up.
    down = (-math.sin(a), math.cos(a))
    hx = cx + down[0] * pitch_deg * px_per_deg
    hy = cy + down[1] * pitch_deg * px_per_deg
    reach = size * 2

    sky = np.array([[
        (hx - along[0] * reach - down[0] * reach, hy - along[1] * reach - down[1] * reach),
        (hx + along[0] * reach - down[0] * reach, hy + along[1] * reach - down[1] * reach),
        (hx + along[0] * reach, hy + along[1] * reach),
        (hx - along[0] * reach, hy - along[1] * reach),
    ]], dtype=np.int32)
    ground = np.array([[
        (hx - along[0] * reach, hy - along[1] * reach),
        (hx + along[0] * reach, hy + along[1] * reach),
        (hx + along[0] * reach + down[0] * reach, hy + along[1] * reach + down[1] * reach),
        (hx - along[0] * reach + down[0] * reach, hy - along[1] * reach + down[1] * reach),
    ]], dtype=np.int32)
    cv2.fillPoly(panel, sky, _SKY)
    cv2.fillPoly(panel, ground, _GROUND)
    cv2.line(panel,
             (int(hx - along[0] * reach), int(hy - along[1] * reach)),
             (int(hx + along[0] * reach), int(hy + along[1] * reach)),
             (240, 240, 240), 1, cv2.LINE_AA)

    # The robot's own axis, fixed to the frame: the horizon moves against it, so
    # the gap between the two is the tilt without needing to read a number.
    cv2.line(panel, (cx - radius // 2, cy), (cx - 6, cy), (60, 240, 240), 2, cv2.LINE_AA)
    cv2.line(panel, (cx + 6, cy), (cx + radius // 2, cy), (60, 240, 240), 2, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), 2, (60, 240, 240), -1, cv2.LINE_AA)

    tilt = summary.get("tilt_deg")
    if tilt is not None:
        # Centred and well inside the rim: the circular mask below keeps only
        # what falls within the disc, so a corner caption would be clipped away
        # entirely rather than merely cropped.
        text = "%.0f deg" % tilt
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(panel, text, (cx - tw // 2, cy + int(radius * 0.62)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)

    # Circular mask, so the disc reads as an instrument rather than as a
    # rectangle of colour pasted over the picture.
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    roi = img[pad:pad + size, pad:pad + size]
    blended = cv2.addWeighted(panel, 0.8, roi, 0.2, 0)
    np.copyto(roi, blended, where=mask[:, :, None].astype(bool))
    cv2.circle(img, (pad + cx, pad + cy), radius, (200, 200, 200), 1, cv2.LINE_AA)
    return size


def _label(img: np.ndarray, text: str, x: int, y: int) -> None:
    """Small caption on its own backing box, sized to the text.

    Not the full-width band the scan uses: this one has to coexist with whatever
    else is on the frame, and a band would cover it.
    """
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    height, width = img.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - th - base - 1))
    cv2.rectangle(img, (x, y), (min(width - 1, x + tw + 8), y + th + base + 4), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 4, y + th + 3), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def maybe_draw(img: np.ndarray, scan: Optional[Scan] = None,
               imu: Optional[ImuBatch] = None, **kwargs) -> np.ndarray:
    """Draw whichever sensors are present, and return the image untouched if none are."""
    if scan is not None:
        img = draw_scan(img, scan, **kwargs)
    if imu is not None:
        img = draw_imu(img, imu)
    return img
