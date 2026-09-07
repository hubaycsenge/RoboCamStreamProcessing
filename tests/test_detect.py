"""Detectors: the mission's words, and a box in the image.

Only the weightless backend is exercised here.  OWLv2 needs a download and a
card, and a test that skipped whenever either was missing would be a test that
never ran on this cluster; what is worth testing without one is the seam — that
a detection's box is in the frame's own coordinates, clipped to it, and that the
query splitting turns a mission's sentence into something a detector can match.
"""

from __future__ import annotations

import numpy as np
import pytest

from robocam.detect import (ColourDetector, Detection, NullDetector, available,
                            build, clip_box, queries_from_target)


def a_frame(width=200, height=150, colour=(30, 30, 30)):
    """A dark BGR frame with nothing in it."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = colour
    return img


def with_patch(img, box, colour):
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = colour
    return img


# -- queries -----------------------------------------------------------------

def test_the_whole_phrase_comes_first_and_the_noun_phrase_second():
    """Both, and in that order.

    The full text is what the operator meant and an open-vocabulary model does
    use the modifiers.  The stripped phrase is second because "on the desk"
    describes where to look rather than what to find, and a model matching the
    whole sentence scores the mug lower for the furniture in it.
    """
    assert queries_from_target("the red mug on the desk") == \
        ["the red mug on the desk", "red mug desk"]


def test_a_bare_noun_produces_one_query_not_two_identical_ones():
    assert queries_from_target("mug") == ["mug"]


def test_an_empty_target_produces_nothing_to_search_for():
    """The signal the decision stage uses to stay quiet rather than search for ""."""
    assert queries_from_target("") == []
    assert queries_from_target("   ") == []
    assert queries_from_target(None) == []


# -- boxes -------------------------------------------------------------------

def test_a_box_is_clipped_to_the_frame():
    """An unclipped box indexes the pointmap out of bounds a step later, where
    the traceback says nothing about detection."""
    assert clip_box((-10, -5, 300, 400), 200, 150) == (0, 0, 200, 150)


def test_reversed_corners_are_put_back_in_order():
    """Happens whenever a coordinate conversion gets a sign wrong."""
    assert clip_box((80, 90, 20, 30), 200, 150) == (20, 30, 80, 90)


# -- the colour backend ------------------------------------------------------

def test_it_finds_the_patch_of_the_colour_the_target_names():
    img = with_patch(a_frame(), (60, 40, 120, 100), (0, 0, 230))     # BGR red
    out = ColourDetector().detect(img, queries_from_target("the red mug on the desk"))
    assert out, "a large red rectangle should be found"
    x0, y0, x1, y1 = out[0].box
    assert 55 <= x0 <= 65 and 35 <= y0 <= 45
    assert 115 <= x1 <= 125 and 95 <= y1 <= 105
    assert out[0].label == "red"


def test_it_finds_nothing_when_the_colour_is_not_in_the_frame():
    img = with_patch(a_frame(), (60, 40, 120, 100), (0, 0, 230))
    assert ColourDetector().detect(img, queries_from_target("the green mug")) == []


def test_a_target_with_no_colour_word_is_not_answerable_by_this_backend():
    """Not a failure to find the object — a failure to have been asked a question
    this backend can answer.  It is why the real backend exists, and why the
    ``found`` message reports which detector produced it."""
    img = with_patch(a_frame(), (60, 40, 120, 100), (0, 0, 230))
    assert ColourDetector().detect(img, queries_from_target("my keys")) == []


def test_a_speck_is_not_an_object():
    """A few pixels of a colour is a reflection or a JPEG artefact."""
    img = with_patch(a_frame(), (10, 10, 13, 13), (0, 0, 230))
    assert ColourDetector().detect(img, ["red"]) == []


def test_a_colour_covering_most_of_the_frame_is_the_wall_not_the_object():
    img = a_frame(colour=(0, 0, 230))
    assert ColourDetector(max_area_frac=0.35).detect(img, ["red"]) == []


def test_two_patches_come_back_biggest_first():
    img = a_frame()
    with_patch(img, (10, 10, 40, 40), (0, 0, 230))       # small
    with_patch(img, (80, 60, 160, 130), (0, 0, 230))     # large
    out = ColourDetector().detect(img, ["red"])
    assert len(out) == 2
    assert out[0].area > out[1].area


def test_every_box_it_returns_is_inside_the_frame():
    """The property the geometry downstream depends on, checked against a patch
    that runs off the edge."""
    img = with_patch(a_frame(200, 150), (150, 100, 200, 150), (0, 0, 230))
    for det in ColourDetector().detect(img, ["red"]):
        x0, y0, x1, y1 = det.box
        assert 0 <= x0 < x1 <= 200
        assert 0 <= y0 < y1 <= 150


def test_a_detection_serialises_to_something_json_can_carry():
    det = Detection(box=(1, 2, 3, 4), score=0.5, label="red", query="red mug")
    payload = det.as_dict()
    assert payload["box"] == [1, 2, 3, 4]
    assert payload["score"] == 0.5


# -- the registry ------------------------------------------------------------

def test_both_spellings_of_colour_reach_the_same_backend():
    assert isinstance(build("color"), ColourDetector)
    assert isinstance(build("colour"), ColourDetector)


def test_the_null_backend_finds_nothing_by_construction():
    assert NullDetector().detect(a_frame(), ["red"]) == []


def test_an_unknown_detector_names_the_ones_that_exist():
    with pytest.raises(KeyError) as excinfo:
        build("yolo")
    assert "owl" in str(excinfo.value)


def test_the_registry_is_what_the_cli_offers():
    assert set(available()) == {"colour", "color", "owl", "none"}
