"""A processor that does nothing.

Use it to measure what the transport alone costs: run the client against
``processor.name: noop`` and whatever frame rate you get is the ceiling the
link can sustain before any model is involved.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Frame, Processor


class NoopProcessor(Processor):
    name = "noop"

    def process(self, frame: Frame) -> Dict[str, Any]:
        return {"received": True}
