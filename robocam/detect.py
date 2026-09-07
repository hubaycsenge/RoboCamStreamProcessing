"""Detectors: turning the mission's words into a box in the image.

The first half of the ``LLM/decision making`` box in the system diagram.  It has
exactly one job — given a frame and a free-text target, say where in the image
the target is, or say it is not there — and it is deliberately the only part of
the decision stage that needs weights.  Everything downstream of a box (putting
it in the cloud, transforming to the map frame, judging whether the gripper can
reach it) is geometry, lives in :mod:`robocam.seek`, and is testable without a
GPU.

Why a seam rather than one detector
-----------------------------------
The mission target is free text on purpose ("the red mug on the desk"), which
rules out a fixed class list and points at an open-vocabulary model.  But the
same argument the README makes for keeping ``stats`` as the default processor
applies here: a stage whose only implementation needs a download, a GPU and a
network is a stage that cannot be checked when any of the three is missing, and
on this cluster all three go missing regularly.  So there are two backends and
the seam between them is three methods wide:

``colour``  weightless.  Picks the colour word out of the target text and finds
            the largest blob of it.  It is not a detector in any serious sense —
            it cannot tell a red mug from a red jumper — and it is the right
            thing to run when the question is "does the geometry downstream of
            the box work end to end", because it answers in a millisecond on a
            CPU and never fails to load.
``owl``     OWLv2 through transformers: genuine open-vocabulary detection, the
            target text used verbatim as the query.  This is the one that
            actually seeks.

``none``    answers "not here" to everything.  Useful for measuring what the
            rest of T2 costs without the detector in the budget.

What a detection is, and what it is not
---------------------------------------
A :class:`Detection` is a box in the **original frame's** pixel coordinates,
with a score.  It is not a decision.  Whether a score of 0.31 means the mug is
there is a policy question that depends on the mission, and it is settled in
:mod:`robocam.seek` against a configured threshold, not here — a detector that
silently applied its own would make the threshold two numbers that can disagree.
"""

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """One candidate, in the coordinates of the frame that was passed in.

    ``box`` is ``(x0, y0, x1, y1)`` in pixels, left-top to right-bottom, and
    clipped to the image — a box that runs off the edge would index the pointmap
    out of bounds, and a detector at the frame border is exactly where that
    happens.
    """

    box: Tuple[int, int, int, int]
    score: float
    label: str = ""
    query: str = ""
    #: Whatever the backend wants to keep for the log.  Never parsed downstream.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def centre(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "box": [int(v) for v in self.box],
            "score": round(float(self.score), 4),
            "label": self.label,
            "query": self.query,
            **({"extra": self.extra} if self.extra else {}),
        }


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clip a box to the image and put its corners in the right order.

    Both halves matter.  A model that emits normalised coordinates slightly
    outside ``[0, 1]`` is normal; a box whose ``x1`` is left of its ``x0``
    happens whenever a conversion gets a sign wrong, and an unclipped one of
    either kind indexes the pointmap out of bounds a step later, where the
    traceback says nothing about detection.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (
        int(max(0, min(width - 1, round(x0)))),
        int(max(0, min(height - 1, round(y0)))),
        int(max(0, min(width, round(x1)))),
        int(max(0, min(height, round(y1)))),
    )


class Detector(abc.ABC):
    """Base class.  ``setup`` loads weights on the worker thread, never in ``__init__``.

    The contract mirrors :class:`robocam.processors.Processor` for the same
    reason: a model that loaded in the constructor would load on the server's IO
    thread, on a CUDA context owned by nobody.
    """

    name: str = "detector"
    #: True when this backend can act on arbitrary text.  The decision stage
    #: reports it, because "the detector never found your mug" means something
    #: different when the detector could only ever have looked for a colour.
    open_vocabulary: bool = False

    def __init__(self, **options: Any) -> None:
        self.options = options

    def setup(self) -> None:
        """Load weights.  Called once, on the worker thread, before any frame."""

    @abc.abstractmethod
    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        """Find ``queries`` in ``image_bgr``.  Highest score first, may be empty."""

    def close(self) -> None:
        """Release whatever ``setup`` acquired."""

    def describe(self) -> Dict[str, Any]:
        return {"detector": self.name, "open_vocabulary": self.open_vocabulary}


# -- queries -----------------------------------------------------------------

#: Words that carry no information for a detector and hurt an open-vocabulary
#: one, which matches on the whole phrase: "the red mug on the desk" scores
#: worse against a mug than "red mug" does, because half the phrase describes
#: furniture the box is not meant to contain.
_STOP_PHRASES = re.compile(
    r"\b(on|in|under|near|next to|beside|behind|by|at|the|a|an)\b\s*", re.I
)


def queries_from_target(target: str) -> List[str]:
    """Turn the mission's free text into the phrases to search for.

    Two queries, not one, and the order is the point.  The full text goes first
    because it is what the operator meant and an open-vocabulary model does use
    the modifiers.  The stripped noun phrase goes second because prepositional
    context ("on the desk") describes where to look rather than what to find,
    and a model matching the whole phrase scores the mug lower for it.

    Returns an empty list for an empty target, which is the signal the decision
    stage uses to stay quiet rather than to search for "".
    """
    text = " ".join(str(target or "").split())
    if not text:
        return []
    queries = [text]
    stripped = " ".join(_STOP_PHRASES.sub("", text).split())
    # Only the head noun phrase: everything after the first preposition was
    # location, and it is already gone from `stripped` in the common case.
    if stripped and stripped.lower() != text.lower():
        queries.append(stripped)
    return queries


# -- the weightless one ------------------------------------------------------

#: Hue ranges in OpenCV's 0..179 scale.  Red wraps, hence two ranges; the
#: achromatic three are matched on saturation and value instead, which is why
#: they carry a None hue.
_COLOURS: Dict[str, Optional[Tuple[Tuple[int, int], ...]]] = {
    "red": ((0, 8), (172, 179)),
    "orange": ((9, 20),),
    "yellow": ((21, 34),),
    "green": ((35, 85),),
    "cyan": ((86, 95),),
    "blue": ((96, 130),),
    "purple": ((131, 155),),
    "magenta": ((156, 171),),
    "pink": ((156, 171),),
    "white": None,
    "black": None,
    "grey": None,
    "gray": None,
}


class ColourDetector(Detector):
    """Largest blob of the colour named in the target.  No weights, no network.

    This exists so that the whole of T2 — the box, the points behind it, the
    map-frame coordinate, the reachability verdict, the ``found`` on the wire and
    the robot driving to it — can be exercised on a laptop with a coloured sheet
    of paper, before anyone waits on a checkpoint.  Its limits are not
    incidental, they are the reason a real backend exists: it matches a colour,
    so it will happily report a red jumper as the red mug, and it reports nothing
    at all for "my keys".

    Options
    -------
    min_area_frac:  smallest blob, as a fraction of the frame.  Below this a
                    colour match is a reflection or a JPEG artefact.
    max_area_frac:  largest.  A blob covering half the frame is the wall, the
                    floor or a white balance failure, not the object.
    min_saturation: how vivid a pixel must be to count as its hue.  Low
                    saturation is where every hue is noise.
    """

    name = "colour"
    open_vocabulary = False

    def __init__(self, min_area_frac: float = 0.0008, max_area_frac: float = 0.35,
                 min_saturation: int = 90, min_value: int = 60, **options: Any) -> None:
        super().__init__(min_area_frac=min_area_frac, max_area_frac=max_area_frac,
                         min_saturation=min_saturation, min_value=min_value, **options)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.min_saturation = int(min_saturation)
        self.min_value = int(min_value)

    @staticmethod
    def colour_in(text: str) -> str:
        """The first colour word in the text, or "" if there is none."""
        for word in re.findall(r"[a-z]+", str(text or "").lower()):
            if word in _COLOURS:
                return word
        return ""

    def _mask(self, hsv: np.ndarray, colour: str) -> np.ndarray:
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        ranges = _COLOURS.get(colour)
        if ranges is None:
            # Achromatic: hue is meaningless, so these are separated by value
            # and by *low* saturation, which is what "not a colour" means.
            pale = sat < 60
            if colour == "white":
                return pale & (val >= 200)
            if colour == "black":
                return val <= 55
            return pale & (val >= 70) & (val < 200)
        chromatic = (sat >= self.min_saturation) & (val >= self.min_value)
        hits = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            hits |= (hue >= lo) & (hue <= hi)
        return hits & chromatic

    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        if image_bgr is None or image_bgr.ndim != 3:
            return []
        height, width = image_bgr.shape[:2]
        colour = ""
        query = ""
        for q in queries:
            colour = self.colour_in(q)
            if colour:
                query = q
                break
        if not colour:
            # Not a failure to find the object: a failure to have been asked a
            # question this backend can answer.  Said out loud once per call
            # site rather than returning an empty list that looks like absence.
            log.debug("colour detector: no colour word in %r, nothing to look for", queries)
            return []

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv, colour).astype(np.uint8)
        # Open then close: the first removes speckle that would otherwise become
        # a hundred one-pixel components, the second joins a highlight-split
        # object back into one blob.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_area = float(width * height)
        out: List[Detection] = []
        for i in range(1, n):  # 0 is the background
            x, y, w, h, area = (int(v) for v in stats[i])
            frac = area / frame_area
            if frac < self.min_area_frac or frac > self.max_area_frac:
                continue
            # Fill ratio as the score: a blob that fills its own bounding box is
            # a compact object, while one that fills a tenth of it is a scatter
            # of unrelated pixels the closing happened to join.  It is a shape
            # statistic and not a probability, and the name `score` promises no
            # more than an ordering.
            fill = area / float(max(1, w * h))
            out.append(Detection(
                box=clip_box((x, y, x + w, y + h), width, height),
                score=round(float(fill), 4),
                label=colour,
                query=query,
                extra={"area_px": area, "area_frac": round(frac, 5)},
            ))
        out.sort(key=lambda d: (d.score * d.area), reverse=True)
        return out[:5]


# -- the open-vocabulary one -------------------------------------------------

class OwlDetector(Detector):
    """OWLv2 through transformers: the target text used as the query verbatim.

    Zero-shot and text-conditioned, which is the property the mission needs — a
    target named at T2 launch cannot have been in a class list compiled when the
    server started.

    Options
    -------
    model:      HuggingFace id.  The ``base-patch16-ensemble`` default is the
                accuracy/latency compromise that fits alongside CUT3R on one
                card; ``owlv2-large-patch14-ensemble`` is better and roughly
                triples the per-frame cost, which at this robot's frame rate is
                paid on every frame the detector runs on.
    device:     ``cuda:0``, or ``cpu``.  Deliberately not defaulted to CUT3R's
                device by magic: the two share a card by default and that is a
                choice worth being able to undo in one line when the card is an
                11 GB 1080 Ti rather than a 24 GB 3090.
    score_min:  boxes below this are not returned at all.  A floor, not the
                mission's threshold — see the module docstring.
    """

    name = "owl"
    open_vocabulary = True

    def __init__(self, model: str = "google/owlv2-base-patch16-ensemble",
                 device: str = "cuda:0", score_min: float = 0.08,
                 max_detections: int = 5, **options: Any) -> None:
        super().__init__(model=model, device=device, score_min=score_min,
                         max_detections=max_detections, **options)
        self.model_id = str(model)
        self.device_str = str(device)
        self.score_min = float(score_min)
        self.max_detections = int(max_detections)
        self._torch = None
        self._model = None
        self._processor = None
        self._device = None

    def setup(self) -> None:
        import torch  # noqa: PLC0415 - deliberately not imported at module scope
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        self._torch = torch
        self._device = torch.device(self.device_str if torch.cuda.is_available()
                                    or not self.device_str.startswith("cuda")
                                    else "cpu")
        if str(self._device) != self.device_str:
            log.warning("detector %s: %s is unavailable, running on %s",
                        self.name, self.device_str, self._device)
        log.info("detector %s: loading %s on %s", self.name, self.model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(self._device)
        self._model.eval()
        log.info("detector %s: ready", self.name)

    def close(self) -> None:
        self._model = None
        self._processor = None

    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        if self._model is None or not queries:
            return []
        torch = self._torch
        height, width = image_bgr.shape[:2]
        # ascontiguousarray, not just the reversed view: `[:, :, ::-1]` has a
        # negative stride, and torch.from_numpy refuses those outright rather
        # than copying, so the reversal has to be made real here.
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])

        texts = [list(queries)]
        inputs = self._processor(text=texts, images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        # The square, not the frame.  OWLv2's image processor pads the image to
        # a square before resizing, so the boxes it returns are normalised over
        # that padded square rather than over the frame.  Passing the frame's own
        # (height, width) as the target size therefore stretches every box along
        # the shorter axis — on a 16:9 camera, by 44% in y, which puts a mug's
        # box over the desk below it and its depth on the desk.  Scaling by the
        # square's side and clipping to the frame is the correct undo, and it is
        # why `clip_box` is not merely defensive here.
        square = max(height, width)
        sizes = torch.tensor([[square, square]], device=self._device)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=sizes, threshold=self.score_min,
        )[0]

        out: List[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"],
                                     results["labels"]):
            idx = int(label)
            query = queries[idx] if 0 <= idx < len(queries) else ""
            out.append(Detection(
                box=clip_box([float(v) for v in box], width, height),
                score=float(score),
                label=query,
                query=query,
            ))
        out.sort(key=lambda d: d.score, reverse=True)
        return out[: self.max_detections]


class NullDetector(Detector):
    """Finds nothing, always.  For measuring T2 without the detector's cost."""

    name = "none"

    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        return []


# -- registry ----------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., Detector]] = {
    "colour": ColourDetector,
    "color": ColourDetector,   # the spelling half the world uses
    "owl": OwlDetector,
    "none": NullDetector,
}


def build(name: str, options: Optional[Dict[str, Any]] = None) -> Detector:
    if name not in REGISTRY:
        known = ", ".join(sorted(set(REGISTRY)))
        raise KeyError(f"unknown detector {name!r}; registered: {known}")
    return REGISTRY[name](**(options or {}))


def available() -> List[str]:
    return sorted(set(REGISTRY))
