#
# Metrics broadcaster tests.
#
# The panel is a debug view, so the rule is that it must never get in the way of
# a live call: a slow or dead socket cannot be allowed to block the pipeline.
# The call_ended event matters for a second reason -- the builder uses it to
# hand the next call a fresh WebRTC client, because the prebuilt one does not
# recover from a session the server ended.
#

import asyncio
import json
import unittest

from observability import MetricsBroadcaster, publish_call_ended


class FakeSocket:
    def __init__(self, fails=False):
        self.sent = []
        self.accepted = False
        self._fails = fails

    async def accept(self):
        self.accepted = True

    async def send_text(self, payload):
        if self._fails:
            raise ConnectionError("client went away")
        self.sent.append(json.loads(payload))


async def drain():
    """Let the fire-and-forget send tasks run."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestBroadcaster(unittest.IsolatedAsyncioTestCase):
    async def test_connected_clients_receive_events(self):
        broadcaster = MetricsBroadcaster()
        socket = FakeSocket()
        await broadcaster.connect(socket)

        broadcaster.publish({"type": "tokens", "prompt_tokens": 42})
        await drain()

        self.assertTrue(socket.accepted)
        self.assertEqual([{"type": "tokens", "prompt_tokens": 42}], socket.sent)

    async def test_a_dead_socket_is_dropped_not_retried(self):
        """A debug panel that has gone away must not become the pipeline's
        problem."""
        broadcaster = MetricsBroadcaster()
        broken, healthy = FakeSocket(fails=True), FakeSocket()
        await broadcaster.connect(broken)
        await broadcaster.connect(healthy)

        broadcaster.publish({"type": "node", "name": "greeting", "reset": False})
        await drain()
        broadcaster.publish({"type": "node", "name": "intake", "reset": True})
        await drain()

        self.assertEqual(2, len(healthy.sent))
        self.assertEqual([], broken.sent)

    async def test_publishing_with_nobody_watching_is_harmless(self):
        MetricsBroadcaster().publish({"type": "tokens"})

    async def test_disconnect_stops_delivery(self):
        broadcaster = MetricsBroadcaster()
        socket = FakeSocket()
        await broadcaster.connect(socket)
        broadcaster.disconnect(socket)

        broadcaster.publish({"type": "tokens"})
        await drain()
        self.assertEqual([], socket.sent)


class TestCallEnded(unittest.IsolatedAsyncioTestCase):
    async def test_the_builder_is_told_when_a_call_finishes(self):
        """This is what triggers reloading the embedded client, so the next
        Connect works instead of hanging."""
        import observability

        broadcaster = MetricsBroadcaster()
        socket = FakeSocket()
        await broadcaster.connect(socket)

        original = observability.BROADCASTER
        observability.BROADCASTER = broadcaster
        try:
            publish_call_ended()
            await drain()
        finally:
            observability.BROADCASTER = original

        self.assertEqual([{"type": "call_ended"}], socket.sent)


if __name__ == "__main__":
    unittest.main()
