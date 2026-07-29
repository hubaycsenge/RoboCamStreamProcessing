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
