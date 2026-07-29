import json

import pytest

from robocam import wire


def test_encode_decode_roundtrip():
    header = wire.frame(seq=7, codec=wire.CODEC_JPEG, width=640, height=480, t_capture_ns=123)
    head, body = wire.encode(header, b"\xff\xd8payload")
    got_header, got_payload = wire.decode([head, body])

    assert got_header["type"] == wire.MSG_FRAME
    assert got_header["seq"] == 7
    assert got_header["width"] == 640
    assert got_header["t_capture_ns"] == 123
    assert got_payload == b"\xff\xd8payload"


def test_decode_accepts_single_frame_messages():
    head, _ = wire.encode(wire.ping(1))
    header, payload = wire.decode([head])
    assert header["type"] == wire.MSG_PING
    assert payload == b""


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [b"a", b"b", b"c"],
    ],
)
def test_decode_rejects_wrong_frame_count(frames):
    with pytest.raises(wire.ProtocolError):
        wire.decode(frames)


def test_decode_rejects_non_json_header():
    with pytest.raises(wire.ProtocolError):
        wire.decode([b"\x00\x01not json", b""])


def test_decode_rejects_header_without_type():
    with pytest.raises(wire.ProtocolError):
        wire.decode([json.dumps({"seq": 1}).encode(), b""])


def test_decode_rejects_non_object_header():
    with pytest.raises(wire.ProtocolError):
        wire.decode([json.dumps([1, 2, 3]).encode(), b""])


def test_result_echoes_client_timestamps():
    """The client needs its own timestamps back to compute RTT on its own clock."""
    header = wire.result(seq=3, ok=True, t_capture_ns=111, t_send_ns=222)
    assert header["t_capture_ns"] == 111
    assert header["t_send_ns"] == 222


def test_result_is_json_serialisable():
    header = wire.result(seq=1, ok=True, width=64, height=48, data={"received": True})
    assert json.loads(json.dumps(header))["data"]["received"] is True
