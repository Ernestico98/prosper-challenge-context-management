"""Tool layer: thin adapters between the LLM and the catalog domain.

Importing this package registers every scheduling tool, so `registry.names()`
is populated by the time an agent is built.
"""

from . import scheduling  # noqa: F401  (import for its registration side effects)
from .context import SchedulingContext
from .registry import REGISTRY, ToolSpec, describe_all, get, names, register

__all__ = [
    "REGISTRY",
    "SchedulingContext",
    "ToolSpec",
    "describe_all",
    "get",
    "names",
    "register",
]
