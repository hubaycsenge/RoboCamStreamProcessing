import pytest

from robocam.config import Config, ConfigError


def test_defaults():
    cfg = Config.load(None)
    assert cfg.server.bind.startswith("tcp://")
    assert cfg.queue.max_depth == 2
    assert cfg.processor.name == "stats"


def test_from_dict_overrides():
    cfg = Config.from_dict({
        "server": {"bind": "tcp://127.0.0.1:6000"},
        "processor": {"name": "noop", "workers": 3},
    })
    assert cfg.server.bind == "tcp://127.0.0.1:6000"
    assert cfg.processor.name == "noop"
    assert cfg.processor.workers == 3
    # Untouched sections keep their defaults.
    assert cfg.queue.max_depth == 2


def test_unknown_section_is_an_error():
    with pytest.raises(ConfigError, match="unknown config section"):
        Config.from_dict({"srever": {}})


def test_unknown_key_is_an_error():
    """A typo must fail at startup, not silently do nothing for hours."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"queue": {"max_dept": 4}})


def test_bad_drop_policy_rejected():
    with pytest.raises(ConfigError, match="drop_policy"):
        Config.from_dict({"queue": {"drop_policy": "whatever"}})


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        Config.load("/nonexistent/server.yaml")


def test_shipped_config_parses():
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "config" / "server.yaml"
    cfg = Config.load(path)
    assert cfg.processor.name in ("stats", "noop")


def test_the_seek_section_loads():
    cfg = Config.from_dict({"seek": {"detector": "owl", "grasp_z_max": 0.2}})
    assert cfg.seek.detector == "owl"
    assert cfg.seek.grasp_z_max == 0.2
    assert cfg.seek.enabled is True


def test_an_inverted_gripper_envelope_is_refused():
    """As written the envelope is empty and nothing would ever be reachable —
    which looks exactly like a detector that never finds anything."""
    with pytest.raises(ConfigError, match="grasp_z_max"):
        Config.from_dict({"seek": {"grasp_z_max": 0.0, "grasp_z_min": 0.5}})


def test_a_typo_in_the_seek_section_is_an_error_like_any_other():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"seek": {"grasp_z_maks": 0.2}})


def test_the_t1_thresholds_are_bounded():
    with pytest.raises(ConfigError, match="t1_min_explored"):
        Config.from_dict({"mission": {"t1_min_explored": 1.4}})
