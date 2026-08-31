#
# Static checks on the shipped scheduler agent.
#
# The agent is data, and data can be edited in the builder UI by someone who
# will not run a call afterwards. These are the mistakes worth catching without
# spending an LLM token: unreachable nodes, dead ends, prompts referring to
# slots nothing fills, and reset nodes that throw away facts they still need.
#

import json
import unittest
from pathlib import Path
from string import Formatter

from agent_builder import AgentBuilder
from tools import names as tool_names
from tools.context import SchedulingContext

FLOW_PATH = Path(__file__).parent.parent / "scheduler_flow.json"


def placeholders(text: str) -> set:
    return {
        field for _, field, _, _ in Formatter().parse(text or "") if field
    }


class SchedulerFlowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(FLOW_PATH.read_text())
        cls.ctx = SchedulingContext.default(with_vectors=False)
        cls.builder = AgentBuilder.from_json(FLOW_PATH, tool_context=cls.ctx)
        cls.nodes = {n.name: n for n in cls.builder.config.nodes}


class TestGraphShape(SchedulerFlowTestCase):
    def test_it_compiles(self):
        self.assertTrue(self.builder.build_initial_node()["functions"])

    def test_every_node_is_reachable_from_the_entry_point(self):
        reached = {self.builder.config.initial_node}
        frontier = list(reached)
        while frontier:
            node = self.nodes[frontier.pop()]
            for edge in node.edges:
                if edge.target not in reached:
                    reached.add(edge.target)
                    frontier.append(edge.target)
        self.assertEqual(set(self.nodes), reached, "unreachable nodes in the graph")

    def test_every_node_can_progress_or_ends_the_call(self):
        """A node with no edges and no `end` is a dead end: the caller is stuck
        talking to an agent that cannot move."""
        for name, node in self.nodes.items():
            self.assertTrue(node.edges or node.end, f"'{name}' is a dead end")

    def test_exactly_one_terminal_node(self):
        terminals = [name for name, node in self.nodes.items() if node.end]
        self.assertEqual(1, len(terminals), f"expected one terminal node, got {terminals}")

    def test_all_referenced_tools_exist(self):
        available = set(tool_names())
        for name, node in self.nodes.items():
            for tool in node.tools:
                self.assertIn(tool, available, f"'{name}' references missing tool")


class TestPrompts(SchedulerFlowTestCase):
    def fillable_keys(self) -> set:
        """Everything a prompt could legitimately interpolate: the static values
        bot.py supplies, plus every slot an edge writes into state."""
        keys = {"specialties", "clinic_name"}
        for node in self.nodes.values():
            for edge in node.edges:
                keys |= set(edge.properties)
        return keys

    def test_every_placeholder_is_filled_by_something(self):
        fillable = self.fillable_keys()
        for name, node in self.nodes.items():
            for message in node.task_messages:
                for field in placeholders(message.get("content", "")):
                    self.assertIn(
                        field, fillable, f"'{name}' interpolates unfillable '{field}'"
                    )

    def test_reset_nodes_restate_the_facts_they_need(self):
        """A context reset throws the conversation away, so anything the node
        still needs has to come back through templating."""
        for name, node in self.nodes.items():
            if node.context_strategy == "reset":
                interpolated = set()
                for message in node.task_messages:
                    interpolated |= placeholders(message.get("content", ""))
                self.assertTrue(
                    interpolated, f"reset node '{name}' restates nothing from state"
                )

    def test_persona_forbids_reading_ids_aloud(self):
        persona = self.builder.config.persona.lower()
        self.assertIn("id", persona)
        self.assertTrue(
            "spoken" in persona or "aloud" in persona,
            "the persona must say the output is spoken",
        )


class TestSchedulingPath(SchedulerFlowTestCase):
    def test_the_search_nodes_carry_data_tools(self):
        """The whole point of the schema change: these steps loop on tools
        instead of transitioning for every lookup."""
        for name in ("identify_need", "find_appointment"):
            self.assertTrue(self.nodes[name].tools, f"'{name}' has no data tools")

    def test_booking_happens_where_the_offers_are(self):
        """book_appointment consumes offers cached by find_appointments, so the
        two must live in the same node or the ids are already gone."""
        node = self.nodes["find_appointment"]
        self.assertIn("find_appointments", node.tools)
        self.assertIn("book_appointment", node.tools)

    def test_context_is_reset_once_the_type_is_settled(self):
        self.assertEqual("reset", self.nodes["find_appointment"].context_strategy)

    def test_patient_facts_are_collected_before_any_catalog_lookup(self):
        """is_new_patient prunes the catalog, so asking for it early is what
        keeps later searches from offering things they cannot have."""
        intake_slots = set()
        for edge in self.nodes["intake"].edges:
            intake_slots |= set(edge.properties)
        self.assertIn("is_new_patient", intake_slots)
        self.assertIn("full_name", intake_slots)
        self.assertEqual([], self.nodes["intake"].tools)

    def test_state_keys_match_what_the_tools_read(self):
        """The edges write these; tools/scheduling.py reads them to build the
        Patient. A rename on one side silently breaks policy filtering."""
        written = set()
        for node in self.nodes.values():
            for edge in node.edges:
                written |= set(edge.properties)
        self.assertIn("is_new_patient", written)
        self.assertIn("has_referral", written)


if __name__ == "__main__":
    unittest.main()
