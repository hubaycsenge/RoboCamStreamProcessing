"""Streaming 3D reconstruction with CUT3R: webcam frames in, point cloud out.

This is the processor behind "Deep3R" in the system diagram.  The robot sends
frames; this returns, for every frame it actually runs, a metric-scale point
cloud in a world frame that is common to the whole session, plus the camera
pose that produced it.  Nav2 consumes the cloud; the pose is what lets the
robot relate that world frame to its own ``odom``.

Why CUT3R rather than MASt3R or VGGT
------------------------------------
CUT3R is *recurrent*: it carries a persistent state and updates it with each
new frame, so cost per frame is constant and the pointmaps of every frame
already live in one coordinate system.  The alternatives reconstruct a window
of frames from scratch on each call, which for a robot means paying for the
window every frame and getting a world frame that jumps whenever the window
slides.  CUT3R also predicts **metric** scale, so the cloud can go into a
costmap without an external scale estimate — the one property that decides
whether a monocular reconstruction is usable for navigation at all.

The cost of that choice is coupling.  The public entry points in
``src/dust3r/inference.py`` are all batch: ``inference_recurrent`` starts from
a fresh state every call, and ``inference_step`` is a *probe* — it queries the
state at a virtual view and discards the updated state, which is CUT3R's
"imagine an unobserved viewpoint" feature, not the streaming path.  The
streaming path is ``_encode_views`` + ``_forward_decoder_step``, mirroring the
loop in ``ARCroco3DStereo._forward_impl``.  Both are private, so pin
``cut3r_root`` to a known commit: an upstream refactor of those two names is
what will break this, and nothing else here depends on CUT3R internals.

State, resets and gaps
----------------------
The recurrent state is the map.  Three consequences the config has to face:

* **One worker only.**  Two workers stepping one state interleave frames into
  it and corrupt the reconstruction silently.  ``process`` refuses to run
  concurrently rather than let that happen.
* **Dropped frames are gaps, not corruption.**  The server's queue evicts
  under load, so the state sees a discontinuous trajectory.  CUT3R tolerates
  that — it accepts unordered collections — but a *large* jump means the next
  frame overlaps nothing in the state, so ``reset_on_gap`` starts a new map
  instead of fusing two unrelated scenes.
* **State drifts over a long run.**  ``reset_every`` bounds it.  The model has
  the mechanism built in: a view flagged ``reset`` restores the initial state,
  so a reset costs nothing beyond the map it discards.

Comparing against the robot's own map
-------------------------------------
The cloud is only half of what the system diagram asks for.  The other half is
the ``Compare`` box: the robot uploads the 2D occupancy grid its SLAM is
building, and this processor puts each cloud into that grid's frame and asks what
the two disagree about.  What comes back is a ``map_update`` — the cells the
reconstruction says are occupied and the LiDAR's single horizontal plane could
not see, which is to say the table tops — and, optionally, a ``pose_hint``.

It runs here, in the worker, rather than on the IO thread, for the reason
everything expensive does: it is numpy over a hundred thousand points and it
must not sit in front of the socket.  It runs *in this processor* rather than in
a separate stage because the cloud in its own world frame and the pose that
produced it exist together only here — after the wire encoding, the cloud has
been quantised and the camera pose is a list of lists.

Everything about *whether* to send what it finds is the server's: see
``StreamServer._send_announcements`` for the switches and the rate limits.  This
processor decides only whether there is anything to say.

Wire size
---------
A 512-mode pointmap is ~200k points; as JSON that is tens of megabytes per
frame, which the README rightly warns against.  The cloud is therefore
confidence-filtered, voxel-downsampled to ``max_points``, quantised to 16 bits
against a per-cloud origin and scale, and base64'd.  At the defaults a cloud
is ~30 kB, which fits the existing result envelope with no protocol change.
The quantisation is per-cloud rather than fixed so that it adapts to the extent
of what was seen, instead of clipping a long corridor.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .. import detect as detect_mod
from .. import seek as seek_mod
from .. import wire
from ..compare import CameraMount, compare
from ..occupancy import encode_cells
from .base import Frame, Processor

log = logging.getLogger(__name__)

# Millimetre-ish resolution is far finer than the model's own accuracy, so 16
# bits over the cloud's own extent loses nothing that matters: a 20 m span
# quantises to 0.3 mm steps.
_QUANT_MAX = 65535.0

# Keys CUT3R's own inference deliberately leaves on the host.  ``true_shape``
# in particular is consumed as a plain shape, and moving it to the GPU makes
# the model index a position table with device tensors -- which surfaces as an
# asynchronous device-side assert deep inside RoPE, nowhere near the cause.
# Mirrors ``ignore_keys`` in CUT3R's ``src/dust3r/inference.py``.
_HOST_KEYS = frozenset(
    {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
)


class Deep3RProcessor(Processor):
    """Run CUT3R over the frame stream and return a point cloud per frame.

    Options
    -------
    cut3r_root:
        Checkout of https://github.com/CUT3R/CUT3R.  Added to ``sys.path``
        along with its ``src`` and ``src/croco`` subdirectories, which is how
        CUT3R expects to be imported.
    weights:
        Checkpoint path.  ``cut3r_512_dpt_4_64.pth`` (with ``size: 512``) is
        the accurate one; ``cut3r_224_linear_4.pth`` (with ``size: 224``) is
        several times faster and the right choice for a first end-to-end test.
        ``size`` must match the checkpoint — 224 crops square, 512 does not.
    device:
        CUDA device.  The model holds the state, so this processor pins one GPU
        for the life of the session.
    every_n:
        Run the model on one frame in N.  The robot streams faster than any
        reconstruction can absorb, and skipping in the processor is better than
        letting the queue evict, because skipped frames are then *chosen*
        rather than whichever ones happened to arrive during a forward pass.
        A skipped frame still returns a result — the client counts replies.
    min_conf:
        Confidence below which a point is dropped, on CUT3R's ``conf_self``.
        The model is confident on surfaces and unconfident on sky, glass and
        the image border, which is exactly what must not reach a costmap.
        Note the scale: the checkpoint's ``conf_mode`` is ``('exp', 1, inf)``,
        so confidence *starts* at 1.0, not 0.0.  Measured on a real frame from
        this robot's camera the distribution is p5 1.06, p50 1.9, p95 2.4 —
        so 1.5 keeps about 82% of points, 2.0 keeps about 38%, and anything
        at 3.0 or above keeps none at all.  The default errs towards the
        confident half, because a phantom obstacle in a costmap costs more
        than a thin one.
    voxel_m:
        Voxel edge for downsampling, in metres.  This is the knob that decides
        cloud size; ``max_points`` is a backstop, not the primary control.
    max_points:
        Hard cap after voxelisation, to bound the wire payload whatever the
        scene.  Points over the cap are dropped by lowest confidence first.
    reset_every:
        Frames between forced state resets; 0 disables.  A reset restarts the
        world frame, so the robot must treat the ``map_id`` change as "this is
        a new map", not as a continuation.
    reset_on_gap:
        Sequence-number jump that forces a reset.  Defaults to something a few
        times ``every_n``: a gap that large means the queue evicted a long run
        of frames and the state no longer overlaps what the camera sees.
    colors:
        Include per-point RGB.  Doubles nothing — colour is one byte per
        channel against six for position — but costs a third of the payload.
    lidar_check:
        When a scan is attached, compare the cloud's forward depth with the
        LiDAR's forward range.  CUT3R claims metric scale; this is the cheapest
        possible independent check of that claim, and it is the number to look
        at first when a map comes out plausibly shaped but wrongly sized.
    compare_map:
        Run the comparison against the robot's occupancy grid when the frame
        carries both a grid and a pose.  The geometry it needs — the camera
        mount, the height slice — comes from the server config through
        ``configure``, not from these options, because it describes the robot
        rather than the model and two copies of it could disagree.
    """

    name = "deep3r"

    def __init__(
        self,
        cut3r_root: str = "~/CUT3R",
        weights: str = "~/CUT3R/checkpoints/cut3r_512_dpt_4_64.pth",
        device: str = "cuda:0",
        size: int = 512,
        every_n: int = 1,
        min_conf: float = 2.0,
        voxel_m: float = 0.05,
        max_points: int = 4000,
        reset_every: int = 0,
        reset_on_gap: int = 0,
        colors: bool = True,
        lidar_check: bool = True,
        compare_map: bool = True,
        seek: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(
            cut3r_root=cut3r_root, weights=weights, device=device, size=size,
            every_n=every_n, min_conf=min_conf, voxel_m=voxel_m,
            max_points=max_points, reset_every=reset_every,
            reset_on_gap=reset_on_gap, colors=colors, lidar_check=lidar_check,
            compare_map=compare_map, seek=seek, **options,
        )
        self.cut3r_root = os.path.expanduser(str(cut3r_root))
        self.weights = os.path.expanduser(str(weights))
        self.device_str = str(device)
        self.size = int(size)
        self.every_n = max(1, int(every_n))
        self.min_conf = float(min_conf)
        self.voxel_m = float(voxel_m)
        self.max_points = int(max_points)
        self.reset_every = int(reset_every)
        self.reset_on_gap = int(reset_on_gap) or (4 * self.every_n)
        self.colors = bool(colors)
        self.lidar_check = bool(lidar_check)
        self.compare_map = bool(compare_map)
        self.seek = bool(seek)
        # Filled in by configure(); the defaults describe a camera at 45 cm
        # looking level, which is this robot with its neck at rest.
        self._mount = CameraMount()
        self._compare_cfg = None
        self._map_cfg = None
        self._seek_cfg = None
        self._mission_cfg = None

        # -- the decision stage --------------------------------------------
        # Created in setup() for the same reason the model is: a detector that
        # loaded here would load on the server's IO thread.
        self._detector = None
        self._envelope = seek_mod.ReachEnvelope()
        self._keyframes: Optional[seek_mod.KeyframeStore] = None
        self._memory = seek_mod.TargetMemory()
        self._view_geometry = seek_mod.ViewGeometry()
        # What the retro-search over T1's keyframes has left to do.  Non-empty
        # only between entering t2 and either finding the target or running out
        # of stored frames; drained a few frames at a time so the search never
        # stalls the stream.
        self._recall_queue: list = []
        self._recall_target = ""
        self._recall_done = False
        self._seek_runs = 0
        self._absent_runs = 0
        self._last_found_ns = 0
        self._last_announced: Optional[seek_mod.Sighting] = None
        self._prev_phase = wire.PHASE_EXPLORE

        # Everything below is created in setup(), on the worker thread.
        self._torch = None
        self._model = None
        self._device = None
        self._img_norm = None
        self._resize_pil = None
        self._pose_encoding_to_camera = None
        self._geotrf = None
        self._pil = None

        # The recurrent state, and the bookkeeping that decides when to drop it.
        self._state: Optional[Tuple[Any, ...]] = None
        self._frames_in_state = 0
        self._last_seq: Optional[int] = None
        self._map_id = 0
        self._seen = 0
        # One worker only; this is how that is enforced rather than assumed.
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def configure(self, lidar_cfg: Any, imu_cfg: Any = None, config: Any = None) -> None:
        """Take the camera mount and the height slice from the server config.

        These are tape-measure facts about the robot, not tunings of the model,
        and the failure they cause is the quiet kind: a camera height that is
        10 cm wrong puts every reconstructed surface 10 cm off in z, which moves
        a table top out of the obstacle slice entirely and produces a comparison
        that runs perfectly and finds nothing.
        """
        if config is None:
            return
        self._compare_cfg = getattr(config, "compare", None)
        self._map_cfg = getattr(config, "map", None)
        if self._compare_cfg is not None:
            self._mount = CameraMount(
                x=self._compare_cfg.camera_x,
                y=self._compare_cfg.camera_y,
                z=self._compare_cfg.camera_z,
                roll=self._compare_cfg.camera_roll,
                pitch=self._compare_cfg.camera_pitch,
                yaw=self._compare_cfg.camera_yaw,
            )

        # The decision stage's geometry is the robot's, on the same terms: the
        # gripper envelope is a tape-measure fact and the standoff belongs to the
        # mission, so neither is a processor option.
        self._seek_cfg = getattr(config, "seek", None)
        self._mission_cfg = getattr(config, "mission", None)
        if self._seek_cfg is not None:
            self._envelope = seek_mod.ReachEnvelope(
                grasp_z_min=self._seek_cfg.grasp_z_min,
                grasp_z_max=self._seek_cfg.grasp_z_max,
                reach_radius_m=self._seek_cfg.reach_radius_m,
                robot_radius_m=self._seek_cfg.robot_radius_m,
                standoff_m=(self._mission_cfg.approach_standoff_m
                            if self._mission_cfg is not None else 0.8),
            )
            self._memory = seek_mod.TargetMemory(ttl_s=self._seek_cfg.memory_ttl_s)
            if self._seek_cfg.keyframes:
                self._keyframes = seek_mod.KeyframeStore(
                    max_frames=self._seek_cfg.keyframe_max,
                    min_spacing_m=self._seek_cfg.keyframe_spacing_m,
                    min_spacing_rad=self._seek_cfg.keyframe_spacing_rad,
                )

    def setup(self) -> None:
        """Import CUT3R, load the checkpoint and warm the kernels up.

        Every heavy import lives here rather than at module scope on purpose:
        ``robocam.processors`` imports every registered processor at startup,
        so a module-level ``import torch`` would stop a server that only wanted
        ``stats`` from starting at all on a machine without CUDA.
        """
        for path in (self.cut3r_root,
                     os.path.join(self.cut3r_root, "src"),
                     os.path.join(self.cut3r_root, "src", "croco")):
            if path not in sys.path:
                sys.path.insert(0, path)

        import torch  # noqa: PLC0415 - deliberately deferred, see docstring
        import PIL.Image  # noqa: PLC0415
        from src.dust3r.model import ARCroco3DStereo  # noqa: PLC0415
        from src.dust3r.utils.image import ImgNorm, _resize_pil_image  # noqa: PLC0415
        from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: PLC0415
        from src.dust3r.utils.geometry import geotrf  # noqa: PLC0415

        self._patch_rope_for_pose_token()

        self._torch = torch
        self._pil = PIL.Image
        self._img_norm = ImgNorm
        self._resize_pil = _resize_pil_image
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._geotrf = geotrf

        if not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"CUT3R checkpoint not found: {self.weights}. Download it from the "
                "links in the CUT3R README; 'size' must match the checkpoint "
                "(224 for the linear head, 512 for the DPT head)."
            )

        self._device = torch.device(self.device_str)
        model = ARCroco3DStereo.from_pretrained(self.weights)
        model = model.to(self._device).eval()
        self._model = model

        # One synthetic frame through the whole path: cuDNN autotuning and the
        # first CUDA allocation cost seconds, and paying that on the robot's
        # first real frame looks like a network fault rather than a warm-up.
        warm = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self._run_model(warm, reset=True)
        self._reset_state()
        # The warm-up is not a map; number the first real one 1.
        self._map_id = 0

        self._setup_detector()

    def _setup_detector(self) -> None:
        """Build the T2 detector, on this thread, and never fatally.

        A detector that fails to load must not take the reconstruction down with
        it.  T1 does not use the detector at all and T2 without it still
        navigates and still reports what T1 remembered, so the useful response to
        a missing checkpoint is a loud log line and a server that keeps running —
        not a Slurm job that exits twenty minutes into an experiment.
        """
        if not (self.seek and self._seek_cfg is not None and self._seek_cfg.enabled):
            log.info("decision stage disabled; t2 will navigate without a target search")
            return
        name = self._seek_cfg.detector
        try:
            self._detector = detect_mod.build(name, dict(self._seek_cfg.detector_options))
            self._detector.setup()
        except Exception:  # noqa: BLE001 - see docstring
            log.exception("detector %r failed to load; t2 will run without one", name)
            self._detector = None
            return
        log.info("decision stage ready: detector=%s open_vocabulary=%s",
                 self._detector.name, self._detector.open_vocabulary)

    @staticmethod
    def _patch_rope_for_pose_token() -> None:
        """Let the pure-PyTorch RoPE accept CUT3R's -1 pose-token position.

        CUT3R marks the pose token with position -1, a sentinel meaning "one
        step before the first patch".  ``curope``, the compiled kernel, takes
        that in its stride because it evaluates the sinusoid from the position
        *value*.  The pure-PyTorch fallback instead builds a table for
        ``0..positions.max()`` and looks positions up with
        ``F.embedding`` — so -1 indexes out of bounds and the model dies in an
        asynchronous device-side assert several layers away from the cause.
        The ImportError fallback in croco's ``pos_embed.py`` is therefore not
        equivalent to the kernel for this model, however much it looks like it.

        Shifting every position by a *global* constant is exact, not an
        approximation: RoPE reaches attention only through ``q · k``, which
        depends on the difference of two positions, and a constant added to
        both cancels.  The shift has to be global rather than per-call —
        offsetting by each call's own minimum would shift the decoder's
        queries and keys by different amounts in cross-attention, and that
        would not cancel.

        Compiling ``curope`` makes this unnecessary and is faster; it needs an
        nvcc matching the installed torch, which this cluster does not have.
        """
        from models import pos_embed  # noqa: PLC0415

        rope = pos_embed.RoPE2D
        if rope.__name__ == "cuRoPE2D" or getattr(rope, "_deep3r_patched", False):
            return
        original = rope.forward

        def forward(self, tokens, positions):
            return original(self, tokens, positions + 1)

        rope.forward = forward
        rope._deep3r_patched = True

    def close(self) -> None:
        self._state = None
        self._model = None
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:  # pragma: no cover - teardown best effort
                log.exception("detector close failed")
            self._detector = None
        if self._keyframes is not None:
            self._keyframes.clear()
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    # -- per frame -----------------------------------------------------------

    def process(self, frame: Frame) -> Dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            # Two workers on one recurrent state would interleave frames into
            # it and corrupt the map without any error surfacing.
            raise RuntimeError(
                "deep3r is stateful and must run with processor.workers = 1; "
                "a second worker tried to step the same CUT3R state."
            )
        try:
            return self._process_locked(frame)
        finally:
            self._lock.release()

    def _process_locked(self, frame: Frame) -> Dict[str, Any]:
        self._seen += 1
        gap = 0 if self._last_seq is None else int(frame.seq) - self._last_seq
        self._last_seq = int(frame.seq)

        # Count from the frame just taken, so the first frame of a session
        # runs rather than waiting out a whole skip cycle before the robot
        # gets its first cloud.
        if (self._seen - 1) % self.every_n:
            # Still a result: the client counts replies to decide when to send.
            return {"status": "skipped", "map_id": self._map_id,
                    "frames_in_state": self._frames_in_state}

        reset = False
        reason = ""
        if self._state is None:
            reset, reason = True, "first frame"
        elif gap > self.reset_on_gap:
            reset, reason = True, f"sequence gap of {gap} frames"
        elif self.reset_every and self._frames_in_state >= self.reset_every:
            reset, reason = True, f"reset_every {self.reset_every} reached"
        if reset:
            self._reset_state()

        t0 = time.monotonic()
        pts, rgb, conf, pose_c2w = self._run_model(frame.image, reset=reset)
        infer_ms = (time.monotonic() - t0) * 1000.0

        t1 = time.monotonic()
        kept_pts, kept_rgb, kept_conf, counts = self._reduce(pts, rgb, conf)
        cloud = self._pack_cloud(kept_pts, kept_rgb, kept_conf, counts)
        encode_ms = (time.monotonic() - t1) * 1000.0

        self._frames_in_state += 1
        data: Dict[str, Any] = {
            "map_id": self._map_id,
            "frames_in_state": self._frames_in_state,
            "reset": reset,
            "reset_reason": reason,
            "seq_gap": gap,
            "cloud": cloud,
            "pose_c2w": [[round(float(v), 5) for v in row] for row in pose_c2w],
            "infer_ms": round(infer_ms, 1),
            "encode_ms": round(encode_ms, 1),
        }
        if self.lidar_check:
            data["scale_check"] = self._scale_check(frame, pts, conf)
        if self.compare_map:
            data["compare"] = self._compare_with_map(frame, kept_pts, pose_c2w)
        if self.seek:
            # The raw pointmap, not the reduced cloud.  The decision stage looks
            # up the points behind a *pixel box*, which only the per-pixel map
            # can answer: `kept_pts` has been confidence-filtered, voxelised and
            # capped, and none of those survive the row-major indexing that ties
            # a point back to the pixel it came from.
            data["seek"] = self._decide(frame, pts, conf, pose_c2w)
        return data

    # -- the decision stage --------------------------------------------------

    def _placement(self, frame: Frame, pose_c2w) -> Optional["seek_mod.Placement"]:
        """Tie this frame's cloud to the robot's map frame, or admit we cannot.

        Everything the decision stage produces is a coordinate in the robot's
        map, so without a pose there is nothing to produce: the cloud sits in
        CUT3R's own world frame, whose origin nobody chose and which no
        behaviour tree can navigate in.
        """
        if frame.odom is None:
            return None
        return seek_mod.Placement(
            pose_c2w=pose_c2w,
            odom=frame.odom,
            mount=self._mount,
            map_id=(frame.grid.map_id if frame.grid is not None else ""),
            cloud_map_id=self._map_id,
            frame=(frame.grid.frame if frame.grid is not None else frame.odom.frame),
            t_ns=wire.monotonic_ns(),
        )

    def _store_keyframe(self, frame: Frame, points, conf, placement) -> bool:
        """Keep this frame if T1 has not already seen this pose.

        The stored copy is downscaled to the size the reconstruction ran at,
        which is also all the detector needs: it is looking for an object that
        fills a tenth of the frame, and a session of 720p frames is gigabytes.
        The *placement* is not approximated — it is what turns a hit in an old
        image into a coordinate in the current map.
        """
        if self._keyframes is None or not self._keyframes.should_keep(frame.odom):
            return False
        view = self._view_geometry
        try:
            small = cv2.resize(frame.image, (view.out_w, view.out_h),
                               interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001 - a keyframe is never worth a failed frame
            log.exception("keyframe resize failed at seq=%d", frame.seq)
            return False
        # Rewrite the geometry for the *stored* image, which is the crop itself:
        # a box found in it is already in pointmap coordinates.
        stored_view = seek_mod.ViewGeometry(
            scale_x=1.0, scale_y=1.0, crop_x0=0, crop_y0=0,
            out_w=view.out_w, out_h=view.out_h, src_w=view.out_w, src_h=view.out_h,
        )
        return self._keyframes.add(seek_mod.Keyframe(
            image=small,
            view=stored_view,
            placement=placement,
            pointmap=np.asarray(points, dtype=np.float32).copy(),
            conf=np.asarray(conf, dtype=np.float32).copy(),
            seq=int(frame.seq),
            t_ns=placement.t_ns,
        ))

    def _arm_recall(self, target: str) -> None:
        """Queue T1's keyframes to be searched for a target named just now.

        This is the half of the mission that cannot be done live: T1 explored
        without knowing what it would be asked for, so the only place the answer
        can already exist is in what it saw.  The queue is drained a few frames
        at a time by :meth:`_run_recall` rather than all at once, because the
        detector over two hundred keyframes is tens of seconds and the robot is
        still streaming.
        """
        self._recall_target = target
        self._recall_done = False
        self._recall_queue = (self._keyframes.newest_first()
                              if self._keyframes is not None else [])
        if self._recall_queue:
            log.info("t2 target %r: searching %d keyframes kept during t1",
                     target, len(self._recall_queue))
        else:
            self._recall_done = True

    def _run_recall(self, target: str, budget: int) -> Optional["seek_mod.Sighting"]:
        """Search a few stored keyframes.  Returns the first sighting, or None.

        Newest first, and it stops at the first hit rather than ranking every
        keyframe: a target seen several times during exploration is best
        described by the most recent sighting, and the robot's own live branch
        supersedes any of them the moment it sees the thing itself.
        """
        cfg = self._seek_cfg
        queries = detect_mod.queries_from_target(target)
        for _ in range(max(1, budget)):
            if not self._recall_queue:
                self._recall_done = True
                return None
            kf = self._recall_queue.pop(0)
            try:
                detections = self._detector.detect(kf.image, queries)
            except Exception:  # noqa: BLE001 - one bad keyframe is not fatal
                log.exception("detector raised on keyframe seq=%d", kf.seq)
                continue
            for det in detections:
                if det.score < cfg.min_confidence:
                    continue
                sighting, _why = seek_mod.locate(
                    det, kf.pointmap, kf.conf, kf.view, kf.placement, target,
                    min_conf=cfg.min_point_conf, min_points=cfg.min_points,
                    depth_band_m=cfg.depth_band_m, keep_frac=cfg.box_keep_frac,
                    detector=self._detector.name,
                )
                if sighting is not None:
                    sighting.basis = seek_mod.BASIS_MEMORY
                    log.info("t2 target %r found in a t1 keyframe (seq=%d, %.0fs old) "
                             "at (%.2f, %.2f, %.2f)",
                             target, kf.seq, sighting.age_s(), sighting.x,
                             sighting.y, sighting.z)
                    self._recall_queue.clear()
                    self._recall_done = True
                    return sighting
        return None

    def _decide(self, frame: Frame, points, conf, pose_c2w) -> Dict[str, Any]:
        """Watch this frame for the mission target and announce what is known.

        Returns statistics either way, including when it did not run and why —
        the same argument :meth:`_compare_with_map` makes.  "No detector", "no
        target named", "t1, so only remembering" and "looked and saw nothing"
        are four different situations that all produce no ``found``, and three of
        them are fixable in a minute by whoever is told.
        """
        cfg = self._seek_cfg
        if cfg is None or not cfg.enabled:
            return {"ran": False, "why": "the decision stage is disabled on the server"}
        if self._detector is None:
            return {"ran": False, "why": "no detector loaded; see the startup log"}

        placement = self._placement(frame, pose_c2w)
        if placement is None:
            return {"ran": False,
                    "why": "no pose fresh enough to put a sighting in the map frame"}

        if frame.phase != wire.PHASE_SEEK:
            # T1: remember, do not decide.  The target is not named yet, and
            # announcing a found in t1 would reach a robot that is not listening.
            stored = self._store_keyframe(frame, points, conf, placement)
            self._prev_phase = frame.phase
            out: Dict[str, Any] = {
                "ran": False,
                "why": "phase is t1: keeping keyframes so a target named at t2 can be "
                       "looked for in what t1 already saw",
                "kept_keyframe": stored,
            }
            if self._keyframes is not None:
                out["keyframes"] = self._keyframes.stats()
            return out

        target = str((frame.mission or {}).get("target", "") or "")
        if not target:
            return {"ran": False,
                    "why": "phase is t2 but no target is named; pass --target on the "
                           "server or send one in the robot's hello"}

        if self._prev_phase != wire.PHASE_SEEK or self._recall_target != target:
            self._arm_recall(target)
        self._prev_phase = wire.PHASE_SEEK

        stats: Dict[str, Any] = {
            "ran": True,
            "target": target,
            "detector": self._detector.name,
            "open_vocabulary": self._detector.open_vocabulary,
        }

        # -- the live branch -------------------------------------------------
        self._seek_runs += 1
        live: Optional[seek_mod.Sighting] = None
        # Count from the frame just taken, as deep3r's own every_n does, so the
        # first frame of t2 runs the detector rather than waiting out a whole
        # skip cycle before the robot's monitor branch gets anything.
        if (self._seek_runs - 1) % max(1, cfg.detect_every_n) == 0:
            live, detect_stats = self._detect_live(frame, points, conf, placement, target)
            stats.update(detect_stats)
        else:
            stats["detected"] = "skipped by seek.detect_every_n"

        # -- the memory branch -----------------------------------------------
        # Only while the live branch has nothing.  A remembered coordinate is a
        # claim about the past and the present beats it whenever there is one.
        recalled: Optional[seek_mod.Sighting] = None
        if live is None and not self._recall_done:
            recalled = self._run_recall(target, cfg.recall_budget_per_frame)
            if recalled is not None:
                self._memory.remember(recalled)
        stats["recall_remaining"] = len(self._recall_queue)

        sighting = live or recalled or self._memory.recall(
            target, map_id=(frame.grid.map_id if frame.grid is not None else ""))
        if sighting is not None:
            stats["sighting"] = sighting.as_dict()

        self._maybe_announce(frame, sighting, live is not None, target, stats)
        if self._keyframes is not None:
            stats["keyframes"] = self._keyframes.stats()
        return stats

    def _detect_live(self, frame: Frame, points, conf, placement, target):
        """Run the detector on this frame and place the best hit in the map."""
        cfg = self._seek_cfg
        out: Dict[str, Any] = {}
        queries = detect_mod.queries_from_target(target)
        t0 = time.monotonic()
        try:
            detections = self._detector.detect(frame.image, queries)
        except Exception:  # noqa: BLE001 - a detector fault is not a dead session
            log.exception("detector raised on seq=%d", frame.seq)
            out["detect_error"] = "the detector raised; see the server log"
            return None, out
        out["detect_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
        out["detections"] = [d.as_dict() for d in detections[:3]]

        for det in detections:
            if det.score < cfg.min_confidence:
                out["rejected"] = (f"best score {det.score:.2f} is under "
                                   f"seek.min_confidence {cfg.min_confidence:.2f}")
                break
            sighting, why = seek_mod.locate(
                det, points, conf, self._view_geometry, placement, target,
                min_conf=cfg.min_point_conf, min_points=cfg.min_points,
                depth_band_m=cfg.depth_band_m, keep_frac=cfg.box_keep_frac,
                detector=self._detector.name,
            )
            out["locate"] = why
            if sighting is not None:
                self._memory.remember(sighting)
                self._absent_runs = 0
                return sighting, out
        else:
            if not detections:
                out["rejected"] = "the detector found nothing in this frame"

        self._absent_runs += 1
        out["absent_runs"] = self._absent_runs
        return None, out

    def _maybe_announce(self, frame: Frame, sighting, is_live: bool,
                        target: str, stats: Dict[str, Any]) -> None:
        """Decide whether this frame's knowledge is worth a message.

        The robot *acts* on a found — it stops what it is doing and drives — so
        the rate limit here is generous where the map updates' is tight.  Three
        things override it, and each is genuinely news: the first announcement of
        a mission, a target that has moved further than the robot's own position
        error, and a coordinate whose basis has changed from remembered to seen.
        """
        cfg = self._seek_cfg
        now = wire.monotonic_ns()

        if sighting is None:
            # "I have looked and it is not here" is a message, and it is what
            # lets the robot stop waiting on the monitor branch and start
            # searching.  Sent once per absence rather than repeatedly.
            if cfg.absent_after_runs and self._absent_runs >= cfg.absent_after_runs:
                self._absent_runs = 0
                self._last_announced = None
                frame.announce(wire.found(
                    seq=0, target=target, found=False, confidence=0.0,
                    frame=(frame.grid.frame if frame.grid is not None else "map"),
                    map_id=(frame.grid.map_id if frame.grid is not None else ""),
                    basis="absent",
                    decider=self._detector.name,
                    rationale=(f"{cfg.absent_after_runs} consecutive detector runs with "
                               "no sighting, and nothing remembered from t1"),
                    reachable=None,
                    evidence={"absent_runs": cfg.absent_after_runs,
                              "recall_exhausted": self._recall_done},
                ))
                stats["announced"] = "absent"
            return

        previous = self._last_announced
        elapsed_ms = (now - self._last_found_ns) / 1e6 if self._last_found_ns else 1e9
        moved = (math.hypot(sighting.x - previous.x, sighting.y - previous.y)
                 if previous is not None else float("inf"))
        basis_changed = previous is not None and previous.basis != sighting.basis
        if not (elapsed_ms >= cfg.min_found_interval_ms
                or moved >= cfg.resend_move_m
                or basis_changed):
            stats["announced"] = "suppressed by seek.min_found_interval_ms"
            return

        reach = seek_mod.judge_reach(
            sighting, self._envelope, frame.grid,
            from_xy=(frame.odom.x, frame.odom.y) if frame.odom is not None else None,
        )
        stats["reach"] = reach.as_dict()

        approach = reach.approach
        yaw = approach[2] if approach is not None else sighting.bearing_rad
        frame.announce(wire.found(
            seq=0, target=target, found=True,
            confidence=sighting.confidence,
            x=sighting.x, y=sighting.y, z=sighting.z, yaw=yaw,
            approach=approach,
            frame=sighting.frame, map_id=sighting.map_id,
            basis=sighting.basis, age_s=sighting.age_s(now),
            reachable=reach.reachable, reach=reach.as_dict(),
            decider=self._detector.name,
            rationale=(f"{'seen now' if is_live else 'seen during t1'}; "
                       f"{sighting.points} cloud points at {sighting.range_m:.2f} m; "
                       f"{reach.reason}"),
            evidence={"box": list(sighting.box), "points": sighting.points,
                      "range_m": round(sighting.range_m, 3),
                      "spread_m": round(sighting.spread_m, 3),
                      "query": sighting.query,
                      "cloud_map_id": self._map_id,
                      "from_frame_seq": int(frame.seq)},
        ))
        self._last_found_ns = now
        self._last_announced = sighting
        stats["announced"] = sighting.basis

    # -- the Compare box -----------------------------------------------------

    def _compare_with_map(self, frame: Frame, points, pose_c2w) -> Dict[str, Any]:
        """Diff this cloud against the robot's occupancy grid and announce the news.

        Returns the statistics either way — including when it could not run, and
        why.  That is the point of returning a dict rather than None: "no map on
        the server", "no pose to place the cloud with" and "compared, found
        nothing" are three different situations that produce the same silence,
        and the first two are configuration mistakes the robot's operator can fix
        in a minute if told.
        """
        if self._compare_cfg is None or not self._compare_cfg.enabled:
            return {"ran": False, "why": "compare is disabled on the server"}
        if frame.grid is None:
            return {"ran": False, "why": "the robot has not uploaded an occupancy grid"}
        if frame.odom is None:
            return {"ran": False,
                    "why": "no pose fresh enough to place the cloud in the map frame"}
        if points.shape[0] == 0:
            return {"ran": False, "why": "no points survived the confidence filter"}

        cfg = self._compare_cfg
        result = compare(
            points, pose_c2w, frame.odom, frame.grid, self._mount,
            z_min=cfg.z_min, z_max=cfg.z_max, min_points=cfg.min_points,
            occupied_value=(self._map_cfg.occupied_value if self._map_cfg else 100),
            min_new_cells=(self._map_cfg.min_update_cells if self._map_cfg else 4),
            hint=cfg.send_hints, search_cells=cfg.search_cells,
        )
        stats = dict(result.stats)
        stats["ran"] = bool(result.stats.get("placed"))
        stats["odom_age_ms"] = round(frame.odom_age_ms, 1)
        stats["grid_age_ms"] = round(frame.grid_age_ms, 1)

        patch = result.patch
        if patch is not None:
            payload = encode_cells(patch.cells, wire.MAP_ENC_I8_ZLIB)
            frame.announce(
                wire.map_update(
                    # seq is stamped by the server, which owns the counter.
                    seq=0,
                    width=patch.width, height=patch.height,
                    resolution=patch.resolution,
                    encoding=wire.MAP_ENC_I8_ZLIB,
                    x0=patch.x0, y0=patch.y0,
                    origin=patch.origin, frame=patch.frame,
                    map_id=frame.grid.map_id,
                    merge=(self._map_cfg.merge if self._map_cfg else wire.MAP_MERGE_MAX),
                    cells_changed=int(stats.get("new_cells", 0)),
                    from_frame_seq=frame.seq,
                    cloud_map_id=self._map_id,
                    data={"agreement": stats.get("agreement"),
                          "over_free": stats.get("over_free"),
                          "z_slice": [cfg.z_min, cfg.z_max]},
                ),
                payload,
            )
            stats["announced_patch_bytes"] = len(payload)

        shift = stats.get("shift")
        if shift:
            frame.announce(wire.pose_hint(
                seq=0,
                dx=shift["dx"], dy=shift["dy"], dyaw=shift["dyaw"],
                confidence=shift["confidence"], inliers=shift["inliers"],
                method=shift["method"], frame=frame.grid.frame,
                map_id=frame.grid.map_id, from_frame_seq=frame.seq,
                data={"baseline_inliers": shift.get("baseline_inliers"),
                      "search_cells": shift.get("search_cells")},
            ))
        return stats

    def _reset_state(self) -> None:
        self._state = None
        self._frames_in_state = 0
        self._map_id += 1

    # -- model ---------------------------------------------------------------

    def _to_view(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Turn a decoded BGR frame into the view dict CUT3R expects.

        Preprocessing is CUT3R's own ``_resize_pil_image`` and ``ImgNorm``
        rather than a reimplementation: the crop-to-multiples-of-16 rule and
        the normalisation constants are part of the trained model, and a local
        copy of them is a silent accuracy bug waiting for an upstream change.
        ``load_images`` itself is unusable here because it only reads files.
        """
        torch = self._torch
        rgb = image_bgr[:, :, ::-1] if image_bgr.ndim == 3 else np.stack([image_bgr] * 3, -1)
        img = self._pil.fromarray(np.ascontiguousarray(rgb)).convert("RGB")

        W1, H1 = img.size
        if self.size == 224:
            img = self._resize_pil(img, round(self.size * max(W1 / H1, H1 / W1)))
            W, H = img.size
            cx, cy = W // 2, H // 2
            half = min(cx, cy)
            crop_x0, crop_y0 = cx - half, cy - half
            img = img.crop((crop_x0, crop_y0, cx + half, cy + half))
        else:
            img = self._resize_pil(img, self.size)
            W, H = img.size
            cx, cy = W // 2, H // 2
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if W == H:
                halfh = int(3 * halfw / 4)
            crop_x0, crop_y0 = cx - halfw, cy - halfh
            img = img.crop((crop_x0, crop_y0, cx + halfw, cy + halfh))

        # Record what was actually done to the frame.  A detector box is in the
        # source frame's pixels and the points behind it are indexed in the
        # pointmap's, and getting that mapping wrong does not raise -- it reads
        # the depth of whatever is a few centimetres to the side, which is a
        # coordinate that looks entirely plausible and is not the object.
        W2, H2 = img.size
        self._view_geometry = seek_mod.ViewGeometry(
            scale_x=(W / W1) if W1 else 1.0,
            scale_y=(H / H1) if H1 else 1.0,
            crop_x0=crop_x0, crop_y0=crop_y0,
            out_w=W2, out_h=H2, src_w=W1, src_h=H1,
        )

        tensor = self._img_norm(img)[None]
        return {
            "img": tensor,
            # NaN says "no ray map for this view"; CUT3R accepts an image-only
            # stream and infers the rays itself.
            "ray_map": torch.full((tensor.shape[0], 6, tensor.shape[-2], tensor.shape[-1]),
                                  torch.nan),
            "true_shape": torch.from_numpy(np.int32([img.size[::-1]])),
            "idx": 0,
            "instance": "0",
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }

    def _run_model(self, image_bgr: np.ndarray, reset: bool):
        """Step the recurrent state by one frame.

        Mirrors ``ARCroco3DStereo._forward_impl``'s loop body for a single
        view, carrying ``(state_feat, state_pos, init_state_feat, mem,
        init_mem)`` across calls instead of rebuilding it per batch.  That is
        the whole point of the model and the whole reason for touching private
        methods; see the module docstring.
        """
        torch = self._torch
        model = self._model
        view = self._to_view(image_bgr)
        for key, value in view.items():
            if key not in _HOST_KEYS and torch.is_tensor(value):
                view[key] = value.to(self._device, non_blocking=True)

        with torch.no_grad():
            shape, feat_ls, pos = model._encode_views([view])
            feat_i, pos_i, shape_i = feat_ls[-1][0], pos[0], shape[0]

            if self._state is None:
                state_feat, state_pos = model._init_state(feat_i, pos_i)
                mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                self._state = (state_feat, state_pos, state_feat.clone(),
                               mem, mem.clone())
            state_feat, state_pos, init_state_feat, mem, init_mem = self._state

            res, (state_feat, mem) = model._forward_decoder_step(
                [view], 0, feat_i, pos_i, shape_i,
                init_state_feat, init_mem, state_feat, state_pos, mem,
            )
            self._state = (state_feat, state_pos, init_state_feat, mem, init_mem)

            # Points come out in the camera's own frame; the pose puts every
            # frame's points into the one world frame that makes this a map
            # rather than a sequence of disconnected depth images.
            pts_self = res["pts3d_in_self_view"]
            conf = res["conf_self"]
            pose = self._pose_encoding_to_camera(res["camera_pose"].clone())
            pts_world = self._geotrf(pose, pts_self)

            rgb = None
            if self.colors:
                # ImgNorm maps to [-1, 1]; this is its exact inverse.
                rgb = (0.5 * (view["img"].permute(0, 2, 3, 1) + 1.0)).clamp(0, 1)
                rgb = (rgb * 255).to(torch.uint8).cpu().numpy().reshape(-1, 3)

        return (pts_world.float().cpu().numpy().reshape(-1, 3),
                rgb,
                conf.float().cpu().numpy().reshape(-1),
                pose.float().cpu().numpy().reshape(4, 4))

    # -- cloud reduction -----------------------------------------------------

    def _reduce(self, pts, rgb, conf):
        """Confidence-filter, voxelise and cap.  Returns (pts, rgb, conf, counts).

        Split out from the packing because the *reduced* points are what the
        comparison against the robot's map wants: already gated to the surfaces
        the model was sure about, already thinned to one point per voxel, and
        still in metres rather than quantised.  Comparing against the packed
        cloud instead would mean decoding what was just encoded, and comparing
        against the raw pointmap would mean carrying the model's low-confidence
        sky into the robot's costmap.
        """
        total = int(pts.shape[0])
        keep = np.isfinite(pts).all(axis=1) & (conf >= self.min_conf)
        pts, conf = pts[keep], conf[keep]
        if rgb is not None:
            rgb = rgb[keep]
        after_conf = int(pts.shape[0])

        if after_conf:
            pts, rgb, conf = self._voxel_downsample(pts, rgb, conf)
        if pts.shape[0] > self.max_points:
            # Drop the least confident first: a cap that dropped points at
            # random would thin the surfaces Nav2 needs along with the noise.
            order = np.argsort(conf)[::-1][: self.max_points]
            pts, conf = pts[order], conf[order]
            if rgb is not None:
                rgb = rgb[order]
        return pts, rgb, conf, (total, after_conf)

    def _encode_cloud(self, pts, rgb, conf) -> Dict[str, Any]:
        """Filter, voxelise, quantise and base64 one frame's points."""
        pts, rgb, conf, counts = self._reduce(pts, rgb, conf)
        return self._pack_cloud(pts, rgb, conf, counts)

    def _pack_cloud(self, pts, rgb, conf, counts) -> Dict[str, Any]:
        """Quantise and base64 an already-reduced cloud."""
        total, after_conf = counts

        out: Dict[str, Any] = {
            "n_points": int(pts.shape[0]),
            "n_raw": total,
            "n_after_conf": after_conf,
            "voxel_m": self.voxel_m,
            "min_conf": self.min_conf,
            # CUT3R's world frame: metric, right-handed, anchored on the first
            # frame of this map_id. Relating it to odom is the robot's job.
            "frame": "cut3r_world",
            # Everything a `sensor_msgs/PointCloud2` needs that is not in the
            # bytes themselves.  The robot republishes this cloud on a ROS topic
            # -- the "Pointcloud2" box in the system diagram -- and without a
            # declared layout it would have to hardcode one here, which is
            # exactly the kind of duplicated constant that survives a change at
            # this end and produces a cloud read as garbage at the other.
            #
            # Note this describes the cloud AFTER the robot has dequantised it
            # to float32 metres, not the uint16 on the wire: PointCloud2 has no
            # 16-bit-with-an-origin encoding, so the conversion happens on the
            # robot and this is what it converts to.
            "pc2": {
                "height": 1,             # unordered: the voxel grid destroyed the
                                         # image raster, so there are no rows
                "width": int(pts.shape[0]),
                "is_bigendian": False,
                "is_dense": True,        # every point is finite; NaNs were filtered
                "point_step": 16 if rgb is not None else 12,
                "fields": (
                    [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                     {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                     {"name": "z", "offset": 8, "datatype": 7, "count": 1}]
                    # datatype 7 is FLOAT32, 6 is UINT32.  rgb packed into one
                    # uint32 is what RViz and pcl expect; a float32 "rgb" field
                    # is the older convention and is a trap in Python.
                    + ([{"name": "rgb", "offset": 12, "datatype": 6, "count": 1}]
                       if rgb is not None else [])
                ),
            },
        }
        if pts.shape[0] == 0:
            out.update({"origin": [0.0, 0.0, 0.0], "scale": 0.0, "xyz_u16": "", "rgb_u8": ""})
            return out

        origin = pts.min(axis=0)
        extent = float(np.max(pts.max(axis=0) - origin))
        scale = extent / _QUANT_MAX if extent > 0 else 1.0
        quantised = np.clip((pts - origin) / scale, 0, _QUANT_MAX).astype("<u2")

        out.update({
            "origin": [round(float(v), 4) for v in origin],
            "scale": float(scale),
            # p = origin + scale * u, u little-endian uint16, xyz interleaved.
            "encoding": "u16le-xyz",
            "xyz_u16": base64.b64encode(quantised.tobytes()).decode("ascii"),
            "rgb_u8": base64.b64encode(rgb.tobytes()).decode("ascii") if rgb is not None else "",
        })
        return out

    def _voxel_downsample(self, pts, rgb, conf):
        """Keep the most confident point in each voxel.

        Averaging within a voxel would smear a surface across whatever noise
        shares the cell; taking the best-scoring point keeps the geometry the
        model was actually sure about.
        """
        keys = np.floor(pts / self.voxel_m).astype(np.int64)
        # Order by voxel, then by descending confidence, and take the first of
        # each run: one pass, no dictionary of a hundred thousand keys.
        flat = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
        ).ravel()
        order = np.lexsort((-conf, flat))
        flat_sorted = flat[order]
        first = np.ones(flat_sorted.shape[0], dtype=bool)
        first[1:] = flat_sorted[1:] != flat_sorted[:-1]
        pick = order[first]
        return pts[pick], (rgb[pick] if rgb is not None else None), conf[pick]

    # -- sanity --------------------------------------------------------------

    def _scale_check(self, frame: Frame, pts, conf) -> Optional[Dict[str, Any]]:
        """Compare the cloud's forward depth with the LiDAR's forward range.

        CUT3R predicts metric scale, which is the property this whole approach
        rests on, and nothing else in the pipeline would notice if it were
        wrong by a factor of two.  The scan is already attached to the frame
        and already reduced, so this costs one median.
        """
        if frame.scan is None:
            return None
        summary = dict(frame.scan.summary)
        # front_min_m is None whenever nothing returned inside the front arc,
        # which is a normal reading in open space rather than a fault; fall
        # back to the nearest return in any direction before giving up.
        front = summary.get("front_min_m") or summary.get("nearest_m")
        if not front or not np.isfinite(float(front)):
            return None
        front = float(front)
        good = conf >= self.min_conf
        if not good.any():
            return None
        # Camera looks down +z in CUT3R's camera frame; after the pose the
        # points are in world, so use distance from the camera centre instead
        # of a single axis, which is orientation-independent.
        depth = np.linalg.norm(pts[good], axis=1)
        cam_min = float(np.percentile(depth, 5))
        return {
            "lidar_front_m": round(float(front), 3),
            "cloud_near_m": round(cam_min, 3),
            "ratio": round(cam_min / front, 3) if front > 0 else None,
            "note": "ratio near 1.0 means CUT3R's metric scale agrees with the LiDAR",
        }
