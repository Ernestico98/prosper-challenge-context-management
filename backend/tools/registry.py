#
# Tool registry — the seam between "agent as data" and "tool bodies as code".
#
# An agent is JSON, so it can only reference a tool by NAME. This registry maps
# those names to real handlers and to the JSON schema the LLM is shown. It also
# backs the builder UI, which needs to offer the tools that actually exist
# rather than letting someone type a name that will fail at call time.
#
# Handlers receive a context object rather than reaching for globals, so the
# whole tool layer is testable without a running voice pipeline.
#

from dataclasses import dataclass, field
from typing import Callable, Optional

REGISTRY = {}


@dataclass(frozen=True)
class ToolSpec:
    """One callable tool, as the LLM sees it and as we run it."""

    name: str
    description: str
    handler: Callable  # (args: dict, ctx) -> JSON-serialisable result
    properties: dict = field(default_factory=dict)
    required: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """Shape the builder UI reads to populate its tool picker."""
        return {
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "required": list(self.required),
        }


def register(
    name: str,
    description: str,
    *,
    properties: Optional[dict] = None,
    required: Optional[list] = None,
) -> Callable:
    """Decorator registering a handler as a named, LLM-callable tool."""

    def decorator(handler: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"Tool {name!r} is already registered.")
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            properties=properties or {},
            required=required or [],
        )
        return handler

    return decorator


def get(name: str) -> ToolSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown tool {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list:
    return sorted(REGISTRY)


def describe_all() -> list:
    return [REGISTRY[name].as_dict() for name in names()]
