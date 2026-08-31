#
# Builder API tests.
#
# The UI can now create, wire and delete agents, so the API is the last thing
# standing between a half-edited graph and a live phone call. These cover the
# guard rails: what gets written, what gets refused, and what the test call runs.
#
# fastapi.testclient ships with the runner's own dependencies, so this needs no
# API keys and no network.
#

import json
import shutil
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from pipecat.runner.run import app


class ApiTestCase(unittest.TestCase):
    """Each test works on scratch agents and cleans up after itself."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._live_before = api._live_agent
        self._created = []

    def tearDown(self):
        api._live_agent = self._live_before
        for name in self._created:
            path = api.AGENTS_DIR / f"{name}.json"
            if path.exists():
                path.unlink()

    def template(self) -> dict:
        return self.client.get("/api/agents/template").json()

    def create(self, name: str, agent=None):
        self._created.append(name)
        return self.client.post(f"/api/agents/{name}", json=agent or self.template())


class TestTemplate(ApiTestCase):
    def test_the_template_is_a_runnable_agent(self):
        """A freshly created agent has to be callable straight away — a lone
        node would compile and then trap the caller."""
        template = self.template()
        self.assertTrue(template["nodes"])
        self.assertIn(
            template["initial_node"], [n["name"] for n in template["nodes"]]
        )
        self.assertTrue(any(n.get("end") for n in template["nodes"]))
        self.assertTrue(any(n.get("edges") for n in template["nodes"]))

    def test_the_template_compiles(self):
        response = self.create("tmp_template_check", self.template())
        self.assertEqual(200, response.status_code, response.text)


class TestCreate(ApiTestCase):
    def test_create_then_fetch(self):
        self.assertEqual(200, self.create("tmp_new").status_code)
        fetched = self.client.get("/api/agents/tmp_new").json()
        self.assertEqual(self.template()["initial_node"], fetched["initial_node"])

    def test_new_agent_appears_in_the_listing(self):
        self.create("tmp_listed")
        ids = [a["id"] for a in self.client.get("/api/agents").json()["agents"]]
        self.assertIn("tmp_listed", ids)

    def test_creating_over_an_existing_agent_is_refused(self):
        """POST is separate from PUT precisely so a mistyped name cannot
        silently overwrite an agent someone else built."""
        self.create("tmp_twice")
        self.assertEqual(409, self.create("tmp_twice").status_code)

    def test_traversal_attempts_never_reach_a_file(self):
        """Routing normalises these away before the handler sees them, so over
        HTTP they are 404s. The guard in `_agent_path` is defence in depth for
        any caller that is not the URL router, so it is asserted directly."""
        for url in ("/api/agents/..%2F..%2Fbot", "/api/agents/..", "/api/agents/.%2E"):
            self.assertNotEqual(200, self.client.get(url).status_code, url)

        for name in ("../bot", "/etc/passwd", "sub/dir", "..", "", ".hidden", "a b"):
            with self.assertRaises(HTTPException, msg=name):
                api._agent_path(name)


class TestValidationBeforeWriting(ApiTestCase):
    """Saving compiles first, so the UI cannot persist a graph that will not run.
    That is what makes it safe to let the editor touch this much."""

    def bad_agent(self, **overrides) -> dict:
        return {**self.template(), **overrides}

    def test_duplicate_node_names_are_refused(self):
        node = {"name": "greeting", "task_messages": [], "edges": []}
        response = self.create("tmp_dupe", self.bad_agent(nodes=[node, dict(node)]))
        self.assertEqual(422, response.status_code)
        self.assertIn("Duplicate", response.json()["error"])
        self.assertFalse((api.AGENTS_DIR / "tmp_dupe.json").exists())

    def test_edge_to_a_missing_node_is_refused(self):
        agent = self.template()
        agent["nodes"][0]["edges"][0]["target"] = "nowhere"
        self.assertEqual(422, self.create("tmp_dangling", agent).status_code)

    def test_unknown_tool_is_refused(self):
        agent = self.template()
        agent["nodes"][0]["tools"] = ["summon_a_doctor"]
        response = self.create("tmp_tool", agent)
        self.assertEqual(422, response.status_code)
        self.assertIn("summon_a_doctor", response.json()["error"])

    def test_missing_initial_node_is_refused(self):
        self.assertEqual(
            422, self.create("tmp_entry", self.bad_agent(initial_node="ghost")).status_code
        )


class TestLiveAgent(ApiTestCase):
    def test_the_default_agent_is_live(self):
        self.assertEqual(api.DEFAULT_AGENT, api.live_agent_path().stem)

    def test_activating_changes_what_a_call_would_run(self):
        self.create("tmp_live")
        self.assertEqual(200, self.client.post("/api/agents/tmp_live/activate").status_code)
        self.assertEqual("tmp_live", api.live_agent_path().stem)

    def test_the_listing_reports_which_is_live(self):
        self.create("tmp_flagged")
        self.client.post("/api/agents/tmp_flagged/activate")
        agents = self.client.get("/api/agents").json()["agents"]
        live = [a["id"] for a in agents if a["active"]]
        self.assertEqual(["tmp_flagged"], live)

    def test_activating_an_agent_that_does_not_exist(self):
        self.assertEqual(404, self.client.post("/api/agents/ghost/activate").status_code)

    def test_a_deleted_live_agent_falls_back_rather_than_crashing_a_call(self):
        api._live_agent = "was_deleted"
        self.assertEqual(api.DEFAULT_AGENT, api.live_agent_path().stem)


class TestDelete(ApiTestCase):
    def test_delete_removes_the_file(self):
        self.create("tmp_doomed")
        self.assertEqual(200, self.client.delete("/api/agents/tmp_doomed").status_code)
        self.assertFalse((api.AGENTS_DIR / "tmp_doomed.json").exists())

    def test_deleting_the_live_agent_is_refused(self):
        """Otherwise the next caller reaches whatever the fallback happens to be."""
        self.create("tmp_live_doomed")
        self.client.post("/api/agents/tmp_live_doomed/activate")
        response = self.client.delete("/api/agents/tmp_live_doomed")
        self.assertEqual(409, response.status_code)
        self.assertTrue((api.AGENTS_DIR / "tmp_live_doomed.json").exists())

    def test_deleting_a_missing_agent(self):
        self.assertEqual(404, self.client.delete("/api/agents/ghost").status_code)


class TestShippedAgentsAreIntact(ApiTestCase):
    def test_both_shipped_agents_still_load_from_their_new_home(self):
        for name in ("scheduler_flow", "example_flow"):
            response = self.client.get(f"/api/agents/{name}")
            self.assertEqual(200, response.status_code, name)
            self.assertTrue(response.json()["nodes"], name)


if __name__ == "__main__":
    unittest.main()
