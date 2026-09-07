"""Exits, phases and the shape of a decision."""

from __future__ import annotations

import math

import pytest

from robocam import mission, wire
from robocam.mission import MissionError
from robocam.odometry import Odom


def candidate(**overrides):
    base = {"id": "e1", "x": 3.0, "y": 0.0, "yaw": 0.0, "width_m": 1.0,
            "status": "open"}
    base.update(overrides)
    return base


def test_a_candidate_without_an_id_gets_one_from_its_index():
    got = mission.normalise_exit({"x": 1.0, "y": 2.0}, index=4)
    assert got["id"] == "e4"


def test_an_unknown_status_is_treated_as_unusable_not_open():
    """A robot inventing a status word must never get that exit recommended."""
    got = mission.normalise_exit(candidate(status="probably-fine"))
    assert got["status"] == mission.EXIT_BLOCKED


def test_a_non_finite_position_is_refused():
    with pytest.raises(MissionError, match="non-finite"):
        mission.normalise_exit(candidate(x=float("nan")))


def test_decode_exits_rejects_a_non_list():
    with pytest.raises(MissionError, match="not a list"):
        mission.decode_exits({"candidates": {"id": "e1"}})


def test_decode_exits_of_a_message_with_none_is_empty():
    assert mission.decode_exits({"type": "exits"}) == []


def test_the_nearest_open_exit_wins_without_a_reconstruction():
    near = mission.normalise_exit(candidate(id="near", x=1.0))
    far = mission.normalise_exit(candidate(id="far", x=8.0))
    pose = Odom(x=0.0, y=0.0, yaw=0.0)
    ranked, chosen, method = mission.rank_exits([far, near], pose)
    assert chosen == "near"
    assert method == "nearest"


def test_unseen_volume_can_outrank_proximity():
    """What the reconstruction is for: a far door onto a lot beats a near cupboard."""
    near = mission.normalise_exit(candidate(id="near", x=1.0))
    far = mission.normalise_exit(candidate(id="far", x=6.0))
    pose = Odom(x=0.0, y=0.0, yaw=0.0)
    ranked, chosen, method = mission.rank_exits(
        [near, far], pose, unseen={"far": 30.0},
    )
    assert chosen == "far"
    assert method == "unseen_volume"


def test_turning_is_charged_for():
    """An exit behind the robot is further away than its distance suggests."""
    ahead = mission.normalise_exit(candidate(id="ahead", x=3.0, y=0.0))
    behind = mission.normalise_exit(candidate(id="behind", x=-2.9, y=0.0))
    pose = Odom(x=0.0, y=0.0, yaw=0.0)
    _, chosen, _ = mission.rank_exits([behind, ahead], pose)
    assert chosen == "ahead"


def test_rejected_candidates_come_back_with_a_reason_rather_than_vanishing():
    """A robot that sent five and got three back has to work out which two."""
    exits = [
        mission.normalise_exit(candidate(id="ok")),
        mission.normalise_exit(candidate(id="narrow", width_m=0.2)),
        mission.normalise_exit(candidate(id="done", status="visited")),
    ]
    ranked, chosen, _ = mission.rank_exits(exits, Odom(x=0.0, y=0.0, yaw=0.0))
    assert chosen == "ok"
    assert len(ranked) == 3
    by_id = {entry["id"]: entry for entry in ranked}
    assert by_id["narrow"]["score"] is None and "0.20 m gap" in by_id["narrow"]["reason"]
    assert by_id["done"]["score"] is None and "visited" in by_id["done"]["reason"]


def test_ranking_without_a_pose_still_answers():
    """A server with no pose ranks by the robot's own priors and says so."""
    exits = [mission.normalise_exit(candidate(id="a", score=0.1)),
             mission.normalise_exit(candidate(id="b", score=0.9))]
    ranked, chosen, _ = mission.rank_exits(exits, None)
    assert chosen == "b"
    assert ranked[0]["distance_m"] is None


def test_nothing_open_means_nothing_chosen():
    exits = [mission.normalise_exit(candidate(status="blocked"))]
    _, chosen, _ = mission.rank_exits(exits, Odom(x=0.0, y=0.0, yaw=0.0))
    assert chosen == ""


@pytest.mark.parametrize("name", ["t1", "t2"])
def test_the_supported_phases_pass(name):
    assert mission.check_phase(name) == name


@pytest.mark.parametrize("name", ["t3", "", None, 1, "explore"])
def test_an_unknown_phase_raises_rather_than_defaulting(name):
    """Defaulting would leave the two ends in different phases, silently."""
    with pytest.raises(MissionError):
        mission.check_phase(name)


def test_the_phase_constants_agree_with_the_wire():
    assert wire.PHASE_EXPLORE == "t1" and wire.PHASE_SEEK == "t2"


def test_a_mission_keeps_its_free_text_target():
    got = mission.normalise_mission({"target": "the red mug on the desk",
                                     "min_confidence": 0.7, "ignored": 1})
    assert got["target"] == "the red mug on the desk"
    assert got["min_confidence"] == 0.7
    assert "ignored" not in got


def test_approach_backs_off_along_the_line_of_travel():
    x, y, yaw = mission.approach_pose(3.0, 0.0, from_x=0.0, from_y=0.0, standoff_m=0.8)
    assert (x, y) == pytest.approx((2.2, 0.0))
    assert yaw == pytest.approx(0.0)


def test_approach_faces_the_target():
    _, _, yaw = mission.approach_pose(0.0, 3.0, from_x=0.0, from_y=0.0, standoff_m=1.0)
    assert yaw == pytest.approx(math.pi / 2)


def test_approach_from_on_top_of_the_target_does_not_invent_a_direction():
    """Standing there already, there is no line to back off along.

    Keeping the current position beats a random spin, and the caller sees a
    zero-length approach rather than a pose it cannot explain.
    """
    x, y, _ = mission.approach_pose(1.0, 1.0, from_x=1.0, from_y=1.0, standoff_m=0.8)
    assert (x, y) == pytest.approx((1.0, 1.0))
