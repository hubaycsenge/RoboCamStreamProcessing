"""RoboCam stream processing — server side.

Receives an encoded video stream from a robot (Jetson Orin Nano) over ZeroMQ,
decodes it, runs a pluggable processor on each frame and sends a result back
over the same connection.

Today the only processor reports image metadata.  The pipeline is built so that
YOLO / MASt3R / VGGT slot in as additional processors without the wire protocol
changing.
"""

__version__ = "0.1.0"

from . import wire  # noqa: F401

__all__ = ["wire", "__version__"]
