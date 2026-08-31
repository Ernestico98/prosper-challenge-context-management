#
# AgentBuilder tests.
#
# The starter compiled every tool into a node transition. These cover the three
# additions that Phase 2 depends on:
#
#   - data tools that run and STAY in the node (Flows' non-edge functions),
#   - task_message templating from flow_manager.state, so a node gets its facts
#     back in a few tokens instead of the whole conversation,
#   - per-node context_strategy, which is what makes the context recyclable.
#

import asyncio
import unittest

from agent_builder import AgentBuilder
from agent_builder.builder import render
from pipecat_flows.types import ContextStrategy

MINIMAL = {
    "name": "Test Agent",
    "persona": "You are a test agent.",
    "initial_node": "start",
    "nodes": [
        {
            "name": "start",
            "task_messages": [{"role": "developer", "content": "Ask what they need."}],
            "tools": ["find_specialties"],
            "edges": [
                {
                    "function": "go_next",
                    "description": "Move on once the specialty is known.",
                    "target": "chosen",
                    "properties": {"specialty": {"type": "string"}},
                    "required": ["specialty"],
                }
            ],
        },
        {
            "name": "chosen",
            "context_strategy": "reset",
            "task_messages": [
                {"role": "developer", "content": "They want {specialty}. Confirm it."}
            ],
            "end": True,
        },
    ],
}


class FakeFlowManager:
    def __init__(self, **state):
        self.state = dict(state)


def build(overrides=None, **kwargs) -> AgentBuilder:
    config = {**MINIMAL, **(overrides or {})}
    return AgentBuilder.from_dict(config, **kwargs)


def functions_by_name(node_config) -> dict:
    return {f.name: f for f in node_config["functions"]}


class TestCompilation(unittest.TestCase):
    def test_edges_and_tools_both_become_callable_functions(self):
        node = build().build_initial_node()
        names = set(functions_by_name(node))
        self.assertEqual({"go_next", "find_specialties"}, names)

    def test_tool_schema_comes_from_the_registry(self):
        tool = functions_by_name(build().build_initial_node())["find_specialties"]
        self.assertIn("complaint", tool.properties)
        self.assertEqual(["complaint"], list(tool.required))

    def test_persona_is_applied_as_the_role_message(self):
        node = build().build_initial_node()
        self.assertEqual("You are a test agent.", node["role_message"])

    def test_terminal_node_ends_the_call(self):
        builder = build()
        node = builder._make_node(builder._nodes_by_name["chosen"])
        self.assertEqual([{"type": "end_conversation"}], node["post_actions"])


class TestDataToolsStayPut(unittest.TestCase):
    """The core of the Phase 2 change: refining a search must not move the graph."""

    def setUp(self):
        from tools.context import SchedulingContext

        self.ctx = SchedulingContext.default(with_vectors=False)
        self.builder = build(tool_context=self.ctx)

    def test_tool_handler_returns_no_next_node(self):
        tool = functions_by_name(self.builder.build_initial_node())["find_specialties"]
        result, next_node = asyncio.run(
            tool.handler({"complaint": "me duele la rodilla"}, FakeFlowManager())
        )
        self.assertIsNone(next_node, "a data tool must not transition")
        self.assertIn("matches", result)

    def test_edge_handler_returns_the_next_node(self):
        edge = functions_by_name(self.builder.build_initial_node())["go_next"]
        flow_manager = FakeFlowManager()
        result, next_node = asyncio.run(
            edge.handler({"specialty": "Orthopedics"}, flow_manager)
        )
        self.assertEqual("chosen", next_node["name"])
        self.assertEqual("success", result["status"])

    def test_edge_persists_its_arguments_into_state(self):
        edge = functions_by_name(self.builder.build_initial_node())["go_next"]
        flow_manager = FakeFlowManager()
        asyncio.run(edge.handler({"specialty": "Orthopedics"}, flow_manager))
        self.assertEqual("Orthopedics", flow_manager.state["specialty"])


class TestTemplating(unittest.TestCase):
    def test_next_node_prompt_is_filled_from_state(self):
        """How a node gets its facts back after a context reset: a few tokens
        of template instead of the whole conversation."""
        builder = build()
        edge = functions_by_name(builder.build_initial_node())["go_next"]
        flow_manager = FakeFlowManager()
        _, next_node = asyncio.run(edge.handler({"specialty": "Orthopedics"}, flow_manager))
        self.assertEqual(
            "They want Orthopedics. Confirm it.", next_node["task_messages"][0]["content"]
        )

    def test_static_values_reach_the_prompt(self):
        builder = build(
            {"nodes": [{**MINIMAL["nodes"][0],
                        "task_messages": [{"role": "developer", "content": "Options: {specialties}"}],
                        "tools": [], "edges": []}],
             "initial_node": "start"},
            template_values={"specialties": "Cardiology, Orthopedics"},
        )
        node = builder.build_initial_node()
        self.assertEqual("Options: Cardiology, Orthopedics", node["task_messages"][0]["content"])

    def test_unknown_placeholder_is_left_alone(self):
        """A typo in an authored prompt must not take down a live call."""
        self.assertEqual("hello {nope}", render("hello {nope}", {}))

    def test_unfilled_slot_is_not_an_error(self):
        builder = build()
        node = builder._make_node(builder._nodes_by_name["chosen"])
        self.assertIn("{specialty}", node["task_messages"][0]["content"])


class TestContextStrategy(unittest.TestCase):
    def test_reset_is_compiled_onto_the_node(self):
        builder = build()
        node = builder._make_node(builder._nodes_by_name["chosen"])
        self.assertEqual(ContextStrategy.RESET, node["context_strategy"].strategy)

    def test_default_leaves_the_strategy_unset(self):
        node = build().build_initial_node()
        self.assertNotIn("context_strategy", node)

    def test_unknown_strategy_is_rejected_at_build_time(self):
        broken = {**MINIMAL["nodes"][1], "context_strategy": "forget_everything"}
        with self.assertRaises(ValueError):
            build({"nodes": [MINIMAL["nodes"][0], broken]})


class TestValidation(unittest.TestCase):
    """Everything catchable must fail here, not mid-call in front of a caller."""

    def test_unknown_tool_name_is_rejected(self):
        broken = {**MINIMAL["nodes"][0], "tools": ["summon_a_doctor"]}
        with self.assertRaises(ValueError) as caught:
            build({"nodes": [broken, MINIMAL["nodes"][1]]})
        self.assertIn("summon_a_doctor", str(caught.exception))

    def test_unknown_edge_target_is_rejected(self):
        broken = {**MINIMAL["nodes"][0]}
        broken["edges"] = [{**broken["edges"][0], "target": "nowhere"}]
        with self.assertRaises(ValueError):
            build({"nodes": [broken, MINIMAL["nodes"][1]]})

    def test_unknown_initial_node_is_rejected(self):
        with self.assertRaises(ValueError):
            build({"initial_node": "missing"})

    def test_empty_agent_is_rejected(self):
        with self.assertRaises(ValueError):
            build({"nodes": [], "initial_node": "start"})


class TestExampleFlowStillCompiles(unittest.TestCase):
    def test_the_shipped_example_agent_is_unaffected(self):
        """The starter's format must keep working: tools and context_strategy
        are additive."""
        from pathlib import Path

        example = Path(__file__).parent.parent / "example_flow.json"
        builder = AgentBuilder.from_json(example)
        node = builder.build_initial_node()
        self.assertTrue(node["functions"])


if __name__ == "__main__":
    unittest.main()
