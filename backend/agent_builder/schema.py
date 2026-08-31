#
# Agent schema — the declarative contract the Phase 2 Copilot reads and writes.
#
# Design rule: stay as close to Pipecat Flows' own vocabulary as possible. A node
# carries Pipecat's native fields (`role_message`, `task_messages`, `pre/post_actions`,
# `context_strategy`) verbatim. What we add is expressed as DATA rather than as Python
# closures — because a Copilot can emit a string, not a callable:
#
#   edges  transitions, as a string `target`
#   tools  data lookups, as registry names — they run and STAY in the node
#
# The edges/tools split is the whole point. Pipecat Flows already distinguishes them
# (a handler returning a next node is an "edge function"; one returning None is not),
# so a node can search the catalog repeatedly without leaving the step it is on.
# Refining a search is a tool call; only a closed decision is a transition.
#
# `AgentBuilder` turns both back into the callables Pipecat wants.
#

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs "Rachel"
DEFAULT_MODEL = "gpt-4o"


@dataclass
class Edge:
    """A transition out of a node, exposed to the LLM as a callable tool."""

    function: str            # tool name the LLM calls to take this edge
    description: str         # when the model should call it
    target: str              # node to transition to (by name)
    # Fields to collect on this edge, as JSON-schema properties.
    properties: dict = field(default_factory=dict)
    required: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            function=d["function"],
            description=d["description"],
            target=d["target"],
            properties=d.get("properties", {}),
            required=d.get("required", []),
        )


@dataclass
class Node:
    """A single conversational state. Fields mirror Pipecat Flows' NodeConfig."""

    name: str
    task_messages: list = field(default_factory=list)   # this node's objectives
    role_message: Optional[str] = None                  # overrides the global persona
    edges: list = field(default_factory=list)           # list[Edge]; transitions out
    tools: list = field(default_factory=list)           # list[str]; registry tool names
    pre_actions: list = field(default_factory=list)
    post_actions: list = field(default_factory=list)
    end: bool = False                                   # terminal -> ends the call
    # "append" (default) keeps the conversation so far; "reset" drops it and starts
    # this node from its own messages. Reset is how the search noise a node produced
    # is thrown away once its decision is closed — the collected facts survive in
    # flow_manager.state and come back through task_message templating.
    context_strategy: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            name=d["name"],
            task_messages=d.get("task_messages", []),
            role_message=d.get("role_message"),
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
            tools=list(d.get("tools", [])),
            pre_actions=d.get("pre_actions", []),
            post_actions=d.get("post_actions", []),
            end=d.get("end", False),
            context_strategy=d.get("context_strategy"),
        )


@dataclass
class AgentConfig:
    """A complete agent: identity + the conversation graph."""

    name: str
    initial_node: str
    nodes: list                          # list[Node]
    persona: str = ""                    # global role_message, applied to every node
    voice_id: str = DEFAULT_VOICE_ID
    model: str = DEFAULT_MODEL

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        return cls(
            name=d["name"],
            initial_node=d["initial_node"],
            nodes=[Node.from_dict(n) for n in d["nodes"]],
            persona=d.get("persona", ""),
            voice_id=d.get("voice_id", DEFAULT_VOICE_ID),
            model=d.get("model", DEFAULT_MODEL),
        )
