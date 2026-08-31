#
# AgentBuilder — loads a declarative agent (JSON / dict) and compiles its node
# graph into Pipecat Flows objects.
#
#   JSON  ->  AgentConfig (validated)  ->  Pipecat Flows NodeConfig graph
#
# This is the seam between "agent as data" (what the Phase 2 Copilot produces)
# and "agent as a running conversation" (what bot.py executes). Keeping the
# compile + validation here means bot.py never touches the graph internals.
#
# A node compiles to two kinds of callable:
#
#   edges  ->  handler returns (result, next_node)  ->  Flows transitions
#   tools  ->  handler returns (result, None)       ->  Flows stays put
#
# The second kind is what lets a node search the catalog as many times as the
# conversation needs. Only a closed decision moves the graph.
#

import json
from pathlib import Path
from string import Formatter
from typing import Optional, Union

from loguru import logger
from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat_flows.types import ContextStrategy, ContextStrategyConfig

from observability import publish_node, publish_tool_call
from tools import registry

from .schema import AgentConfig, Edge, Node


class _LenientValues(dict):
    """Leaves unknown placeholders untouched instead of raising.

    Prompts are authored data: a typo in the builder UI must not take down a
    live call, and a slot that hasn't been filled yet is normal, not an error.
    """

    def __missing__(self, key):
        return "{" + key + "}"


def render(template: str, values: dict) -> str:
    try:
        return Formatter().vformat(template, (), _LenientValues(values))
    except (IndexError, ValueError):
        return template


class AgentBuilder:
    """Builds a runnable Pipecat Flows graph from a declarative AgentConfig."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        tool_context=None,
        template_values: Optional[dict] = None,
    ):
        self.config = config
        # Passed to every tool handler; keeps the tool layer out of globals and
        # testable without a voice pipeline.
        self.tool_context = tool_context
        # Static values available to task_messages, merged with flow_manager.state
        # at node-build time. This is how the catalog gets into a prompt without
        # anyone hand-maintaining a copy of it.
        self.template_values = dict(template_values or {})
        self._nodes_by_name = {n.name: n for n in config.nodes}
        self._validate()

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "AgentBuilder":
        return cls(AgentConfig.from_dict(data), **kwargs)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs) -> "AgentBuilder":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data, **kwargs)

    # ---- validation --------------------------------------------------------
    def _validate(self) -> None:
        names = set(self._nodes_by_name)
        if not names:
            raise ValueError("Agent has no nodes.")
        # Names are identity here: edges target them and initial_node picks one.
        # A duplicate does not raise on its own -- the dict above quietly keeps
        # the last one and the other node becomes unreachable -- so it has to be
        # caught explicitly. Harmless while agents were hand-written; reachable
        # the moment the builder UI can add nodes.
        if len(names) != len(self.config.nodes):
            seen, duplicates = set(), set()
            for node in self.config.nodes:
                (duplicates if node.name in seen else seen).add(node.name)
            raise ValueError(
                f"Duplicate node name(s): {', '.join(sorted(duplicates))}. "
                "Node names must be unique."
            )
        if self.config.initial_node not in names:
            raise ValueError(
                f"initial_node '{self.config.initial_node}' is not a defined node."
            )
        for node in self.config.nodes:
            for edge in node.edges:
                if edge.target not in names:
                    raise ValueError(
                        f"Edge '{edge.function}' in node '{node.name}' targets "
                        f"unknown node '{edge.target}'."
                    )
            # Fail here, at build time, rather than mid-call when the model
            # tries to invoke a tool that was never registered.
            for tool_name in node.tools:
                if tool_name not in registry.REGISTRY:
                    raise ValueError(
                        f"Node '{node.name}' references unknown tool "
                        f"'{tool_name}'. Registered: {registry.names()}"
                    )
            if node.context_strategy and node.context_strategy not in _STRATEGIES:
                raise ValueError(
                    f"Node '{node.name}' has unknown context_strategy "
                    f"'{node.context_strategy}'. Use one of {sorted(_STRATEGIES)}."
                )

    # ---- compilation -------------------------------------------------------
    def build_initial_node(self) -> NodeConfig:
        """Return the entry NodeConfig; downstream nodes are built lazily on transition."""
        return self._make_node(self._nodes_by_name[self.config.initial_node])

    def _make_node(self, node: Node, state: Optional[dict] = None) -> NodeConfig:
        values = {**self.template_values, **(state or {})}
        node_config: NodeConfig = {
            "name": node.name,
            "role_message": node.role_message or self.config.persona,
            "task_messages": [
                {**message, "content": render(message.get("content", ""), values)}
                for message in node.task_messages
            ],
            "functions": [self._make_edge_function(edge) for edge in node.edges]
            + [self._make_tool_function(name) for name in node.tools],
        }
        if node.context_strategy:
            # RESET drops the conversation so far. Safe precisely at a closed
            # decision: the facts live on in flow_manager.state and return
            # through the next node's templated task_messages, so the context
            # is recycled per node instead of growing with the call.
            node_config["context_strategy"] = ContextStrategyConfig(
                strategy=_STRATEGIES[node.context_strategy]
            )
        if node.pre_actions:
            node_config["pre_actions"] = node.pre_actions
        # Explicit post_actions win; otherwise a terminal node ends the call.
        if node.post_actions:
            node_config["post_actions"] = node.post_actions
        elif node.end:
            node_config["post_actions"] = [{"type": "end_conversation"}]
        return node_config

    def _make_edge_function(self, edge: Edge) -> FlowsFunctionSchema:
        async def handler(args: dict, flow_manager: FlowManager):
            # Persist what the caller gave us so later nodes can use it.
            flow_manager.state.update(args)
            logger.info(f"[{edge.function}] -> {edge.target} | collected: {args}")
            target = self._nodes_by_name[edge.target]
            next_node = self._make_node(target, flow_manager.state)
            publish_node(target.name, reset=target.context_strategy == "reset")
            return {"status": "success", **args}, next_node

        return FlowsFunctionSchema(
            name=edge.function,
            description=edge.description,
            properties=edge.properties,
            required=edge.required,
            handler=handler,
        )

    def _make_tool_function(self, tool_name: str) -> FlowsFunctionSchema:
        spec = registry.get(tool_name)

        async def handler(args: dict, flow_manager: FlowManager):
            logger.info(f"[{spec.name}] {args}")
            result = spec.handler(args, self.tool_context, flow_manager)
            publish_tool_call(spec.name, args, result)
            # No next node: Flows treats this as a non-edge function and the
            # conversation stays in the current node, with the result appended
            # to the context.
            return result, None

        return FlowsFunctionSchema(
            name=spec.name,
            description=spec.description,
            properties=spec.properties,
            required=spec.required,
            handler=handler,
        )


_STRATEGIES = {
    "append": ContextStrategy.APPEND,
    "reset": ContextStrategy.RESET,
}
