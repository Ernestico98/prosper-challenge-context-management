#
# What the tool handlers are allowed to touch.
#
# Handlers receive this instead of reaching for module globals, so the tool
# layer can be exercised in tests and evals with no voice pipeline, and so a
# different catalog (or a database-backed index) can be swapped in per call.
#

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from catalog import Catalog, build_index


@dataclass
class SchedulingContext:
    catalog: Catalog
    index: object                      # a catalog.SpecialtyIndex
    bookings: list = field(default_factory=list)
    today: Optional[date] = None       # pinned in tests and evals; None means "now"

    # How many options a tool result may carry. This is a TOOL-layer concern:
    # the domain always returns everything legal, and this is where we decide
    # what is worth spending context tokens on and what a person can hold in
    # their head when it is read aloud to them.
    max_options: int = 3
    # How many bookable combinations to check availability for before ranking.
    # Bounds the work when a request matches dozens of provider/site pairs.
    max_combos_probed: int = 25

    @classmethod
    def default(cls, *, with_vectors: bool = True, **kwargs) -> "SchedulingContext":
        catalog = Catalog.load()
        return cls(catalog=catalog, index=build_index(catalog, with_vectors=with_vectors), **kwargs)
