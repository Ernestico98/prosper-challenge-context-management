#
# Making the context work visible.
#
# The whole of Phase 2 is invisible from the outside: a cheap, accurate agent
# and an expensive, confused one sound identical on the phone. This taps the
# pipeline for what it actually costs and streams it to the builder UI, so the
# numbers in solution.md can be watched happening rather than taken on trust.
#
# Everything here is measured, not estimated: prompt_tokens comes from the
# provider's own usage accounting, via Pipecat's MetricsFrame.
#

import asyncio
import json

from loguru import logger
from pipecat.frames.frames import Frame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MetricsBroadcaster:
    """Fans pipeline events out to whichever UI panels are watching.

    Deliberately lossy: a dropped frame costs a line in a debug panel, so a slow
    or dead socket must never apply backpressure to a live phone call.
    """

    def __init__(self):
        self._clients = set()

    async def connect(self, websocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        logger.debug(f"Metrics client connected ({len(self._clients)} watching)")

    def disconnect(self, websocket) -> None:
        self._clients.discard(websocket)

    def publish(self, event: dict) -> None:
        """Fire-and-forget from any pipeline context."""
        if not self._clients:
            return
        try:
            asyncio.get_running_loop().create_task(self._send(event))
        except RuntimeError:
            pass  # no loop running; nothing is watching anyway

    async def _send(self, event: dict) -> None:
        payload = json.dumps(event)
        for client in list(self._clients):
            try:
                await client.send_text(payload)
            except Exception:  # noqa: BLE001
                self.disconnect(client)


BROADCASTER = MetricsBroadcaster()


class MetricsTap(FrameProcessor):
    """Reads token usage off the wire and passes every frame straight through.

    Placed after the LLM in the pipeline. It only observes: a tap that could
    alter or delay frames would be measuring a system that no longer exists.
    """

    def __init__(self, broadcaster: MetricsBroadcaster = BROADCASTER, **kwargs):
        super().__init__(**kwargs)
        self._broadcaster = broadcaster

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, LLMUsageMetricsData):
                    usage = data.value
                    self._broadcaster.publish(
                        {
                            "type": "tokens",
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "cache_read_input_tokens": usage.cache_read_input_tokens,
                        }
                    )

        await self.push_frame(frame, direction)


def publish_tool_call(name: str, args: dict, result) -> None:
    """Called by the compiled tool handlers, so the trace shows what the model
    asked for and how big the answer was."""
    BROADCASTER.publish({"type": "tool", "name": name, "summary": _summarise(args, result)})


def publish_node(name: str, *, reset: bool) -> None:
    BROADCASTER.publish({"type": "node", "name": name, "reset": reset})


def publish_call_ended() -> None:
    """Say when a call is over, so the builder can hand the caller a fresh client.

    The prebuilt WebRTC client does not recover from a session the server ended:
    pressing Connect a second time posts /start, gets a new session, and then
    never sends an offer, so it hangs. The server is fine -- it issues sessions
    happily -- and the bug is in a dependency we ship rather than own, so the
    builder reloads the embedded client instead of leaving people to press F5.
    """
    BROADCASTER.publish({"type": "call_ended"})


def _summarise(args: dict, result) -> str:
    """One line: what was asked, and what it cost in characters of context."""
    asked = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
    if isinstance(result, dict) and result.get("error"):
        return f"({asked}) -> {result['error']}"
    return f"({asked}) -> {len(json.dumps(result, default=str))} chars"
