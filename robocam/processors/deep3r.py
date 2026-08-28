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
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Frame, Processor

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
        **options: Any,
    ) -> None:
        super().__init__(
            cut3r_root=cut3r_root, weights=weights, device=device, size=size,
            every_n=every_n, min_conf=min_conf, voxel_m=voxel_m,
            max_points=max_points, reset_every=reset_every,
            reset_on_gap=reset_on_gap, colors=colors, lidar_check=lidar_check,
            **options,
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
        cloud = self._encode_cloud(pts, rgb, conf)
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
        return data

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
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            img = self._resize_pil(img, self.size)
            W, H = img.size
            cx, cy = W // 2, H // 2
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if W == H:
                halfh = int(3 * halfw / 4)
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

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

    def _encode_cloud(self, pts, rgb, conf) -> Dict[str, Any]:
        """Filter, voxelise, quantise and base64 one frame's points."""
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

        out: Dict[str, Any] = {
            "n_points": int(pts.shape[0]),
            "n_raw": total,
            "n_after_conf": after_conf,
            "voxel_m": self.voxel_m,
            "min_conf": self.min_conf,
            # CUT3R's world frame: metric, right-handed, anchored on the first
            # frame of this map_id. Relating it to odom is the robot's job.
            "frame": "cut3r_world",
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
