"""Exits, the phase machine, and the shape of a decision.

The parts of the system diagram that are not sensor data: the ``Exit?`` box the
robot fills in, the ranking that comes back, the T1 → T2 transition, and the
``Found`` announcement that ends T2.

Why the exits come from the robot
---------------------------------
Frontier detection is a few lines over an occupancy grid and the server could do
it from the grid it already holds.  It does not, because the useful half of an
exit candidate is not where it is — it is what has already been tried.  The robot
knows which candidates it has driven to, which turned out to be a doorway into a
cupboard, and which it could not reach; the server would have to be told all of
that to rediscover the same list a step later.  So the robot proposes and
remembers, and the server ranks.  That division also means an exit survives a
server restart, which matters when the server is a Slurm job.

Ranking without the reconstruction is still ranking
---------------------------------------------------
:func:`rank_exits` is deliberately runnable with nothing but the candidates and
the robot's pose.  A server with no model loaded, or one whose reconstruction has
just reset, still answers — with the nearest unvisited exit, which is a poor
policy and an honest one.  ``method`` in the reply says which it was, so a robot
seeing ``nearest`` for a whole session knows the reconstruction never contributed
rather than concluding the ranking is simply bad.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import wire
from .odometry import Odom, wrap_angle

#: What an exit candidate may say about itself.  ``open`` is the only status the
#: ranking will recommend; the others exist so the robot can send its whole list
#: every time rather than maintaining a filtered copy, which is how a candidate
#: it has already rejected ends up being recommended again.
EXIT_OPEN = "open"
EXIT_VISITED = "visited"
EXIT_BLOCKED = "blocked"

EXIT_STATUSES = (EXIT_OPEN, EXIT_VISITED, EXIT_BLOCKED)


class MissionError(Exception):
    """Raised when an exits or phase message does not parse."""


def normalise_exit(raw: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Validate one exit candidate and fill in its defaults.

    Fields:

    ``id``       stable across messages; the robot's own name for this candidate.
                 Stable is what makes ``chosen`` meaningful in a reply, so one is
                 invented from the index only when the robot supplies none.
    ``x, y``     metres in the map frame: the point to drive *to*.
    ``yaw``      the heading to arrive on, radians.  A doorway approached
                 sideways is a doorway the robot cannot fit through.
    ``width_m``  how wide the gap is.  The robot has a width; a 40 cm frontier
                 is a piece of noise between two obstacle cells, not an exit.
    ``score``    the robot's own prior, if it has one.  Carried through and
                 blended rather than overridden — the robot may know the corridor
                 slopes, and the server does not.
    ``status``   see above.
    """
    if not isinstance(raw, dict):
        raise MissionError(f"exit candidate {index} is {type(raw).__name__}, not an object")

    try:
        x = float(raw.get("x", 0.0))
        y = float(raw.get("y", 0.0))
    except (TypeError, ValueError) as exc:
        raise MissionError(f"exit candidate {index} has non-numeric position: {exc}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise MissionError(f"exit candidate {index} is at a non-finite position")

    try:
        yaw = wrap_angle(float(raw.get("yaw", 0.0) or 0.0))
    except (TypeError, ValueError):
        yaw = 0.0

    status = str(raw.get("status", EXIT_OPEN) or EXIT_OPEN)
    if status not in EXIT_STATUSES:
        # Not fatal: an unknown status is treated as unusable rather than as
        # open, so a robot inventing a status word never gets a candidate
        # recommended by accident.
        status = EXIT_BLOCKED

    def _float(key: str, default: float) -> float:
        try:
            v = float(raw.get(key, default) or default)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    return {
        "id": str(raw.get("id") or f"e{index}"),
        "x": x,
        "y": y,
        "yaw": yaw,
        "width_m": max(0.0, _float("width_m", 0.0)),
        "score": _float("score", 0.0),
        "status": status,
        "source": str(raw.get("source", "")),
    }


def decode_exits(header: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate every candidate in one ``exits`` message."""
    raw = header.get("candidates")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MissionError(f"candidates is {type(raw).__name__}, not a list")
    return [normalise_exit(item, i) for i, item in enumerate(raw)]


def rank_exits(
    candidates: Iterable[Dict[str, Any]],
    pose: Optional[Odom] = None,
    *,
    min_width_m: float = 0.6,
    unseen: Optional[Dict[str, float]] = None,
    turn_cost_m_per_rad: float = 0.5,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Order the candidates.  Returns (ranked, chosen_id, method).

    The cost is distance plus a charge for turning, and the gain is whatever the
    server knows about unseen volume behind each exit (``unseen``, keyed by exit
    id, supplied by whatever has the reconstruction).  With no ``unseen`` the
    ranking degrades to nearest-first, and ``method`` says so.

    Turning is charged because a mecanum base can strafe but a doorway still has
    to be entered facing it, and an exit 2 m behind the robot is further away
    than an exit 3 m ahead.  Half a metre per radian is roughly the ratio at this
    robot's driving and turning speeds; it is a tie-breaker, not a model.

    Candidates that are too narrow, visited or blocked are kept in the output —
    with ``score: null`` and a reason — rather than dropped.  A robot that sent
    five exits and got three back has to work out which two were rejected and
    why; a robot that gets five back with reasons does not.
    """
    unseen = unseen or {}
    method = "unseen_volume" if unseen else "nearest"
    ranked: List[Dict[str, Any]] = []

    for exit_ in candidates:
        entry: Dict[str, Any] = {"id": exit_["id"], "x": exit_["x"], "y": exit_["y"]}

        if exit_["status"] != EXIT_OPEN:
            entry.update({"score": None, "reason": f"status is {exit_['status']}"})
            ranked.append(entry)
            continue
        if exit_["width_m"] and exit_["width_m"] < min_width_m:
            entry.update({
                "score": None,
                "reason": f"{exit_['width_m']:.2f} m gap is under the {min_width_m:.2f} m minimum",
            })
            ranked.append(entry)
            continue

        if pose is not None:
            dx, dy = exit_["x"] - pose.x, exit_["y"] - pose.y
            distance = math.hypot(dx, dy)
            turn = abs(wrap_angle(math.atan2(dy, dx) - pose.yaw))
            cost = distance + turn_cost_m_per_rad * turn
        else:
            distance, turn = float("nan"), float("nan")
            cost = 1.0

        gain = float(unseen.get(exit_["id"], 0.0)) + exit_["score"]
        # Cost in the denominator rather than subtracted: the two have different
        # units (metres against a volume-ish score) and there is no principled
        # rate to convert one into the other.  A ratio at least ranks "a lot
        # behind a far door" above "a little behind a near one" without
        # pretending to a conversion factor that was made up.
        score = (1.0 + gain) / max(cost, 0.25)

        entry.update({
            "score": round(score, 4),
            "distance_m": None if math.isnan(distance) else round(distance, 3),
            "turn_deg": None if math.isnan(turn) else round(math.degrees(turn), 1),
            "unseen": round(float(unseen.get(exit_["id"], 0.0)), 4),
            "reason": "open" if pose is not None else "open, no pose to measure from",
        })
        ranked.append(entry)

    scored = [e for e in ranked if e.get("score") is not None]
    scored.sort(key=lambda e: e["score"], reverse=True)
    rejected = [e for e in ranked if e.get("score") is None]
    chosen = scored[0]["id"] if scored else ""
    return scored + rejected, chosen, method


def check_phase(name: Any) -> str:
    """Validate a phase name, raising rather than defaulting.

    Defaulting an unrecognised phase to ``t1`` would be the wrong kind of
    forgiving: the two ends would then differ about which one is making
    decisions, and the symptom is a robot that explores forever while the server
    waits to be asked to seek.
    """
    if not isinstance(name, str) or name not in wire.SUPPORTED_PHASES:
        raise MissionError(
            f"unknown phase {name!r}; supported: {', '.join(wire.SUPPORTED_PHASES)}"
        )
    return name


def normalise_mission(raw: Any) -> Dict[str, Any]:
    """The mission, as the decision stage will read it.

    ``target`` is free text on purpose — it is a prompt for whatever runs in the
    ``LLM/decision making`` box, and constraining it to a class label here would
    throw away the half of the description that makes it findable ("the red mug
    *on the desk*").
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise MissionError(f"mission is {type(raw).__name__}, not an object")
    mission = {"target": str(raw.get("target", ""))}
    for key in ("description", "id", "deadline_s", "min_confidence"):
        if key in raw:
            mission[key] = raw[key]
    return mission


def approach_pose(x: float, y: float, from_x: float, from_y: float,
                  standoff_m: float = 0.8) -> Tuple[float, float, float]:
    """Where to stand to act on a target at ``(x, y)``, coming from ``(from_x, from_y)``.

    Back off along the line the robot is already approaching on and face the
    target.  This is the pose that goes in ``found.approach``: driving to the
    target's own coordinates means driving into it, and the standoff is the
    robot's reach plus its radius — a property of this robot, hence a parameter.

    Degenerate case: the robot is already on top of the target, so there is no
    direction to back off along.  Keeping its current heading is then the only
    answer that does not invent one, and the caller sees a zero-length approach
    vector rather than a random spin.
    """
    dx, dy = x - from_x, y - from_y
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx) if distance > 1e-6 else 0.0
    if distance <= standoff_m:
        return (from_x, from_y, heading)
    ux, uy = dx / distance, dy / distance
    return (x - ux * standoff_m, y - uy * standoff_m, heading)
