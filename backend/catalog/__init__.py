"""The catalog domain: records and indices, the booking policies, and the one
constraint solver every scheduling tool projects from.

No LLM anywhere in this package — it is the deterministic, testable half of the
system, and the Phase 1 resource UI reads from it too.
"""

from .policies import Patient, Rule, Violation, is_bookable, needs_location_disambiguation, violations
from .query import (
    BookableCombo,
    bookable_specialties,
    distinct_locations,
    distinct_providers,
    distinct_types,
    find_bookable,
    unstaffed_appointment_types,
    why_not,
)
from .index import (
    InMemorySpecialtyIndex,
    ScoredSpecialty,
    SpecialtyIndex,
    build_index,
    load_aliases,
)
from .store import AppointmentType, Catalog, Location, Provider

__all__ = [
    "AppointmentType",
    "BookableCombo",
    "Catalog",
    "InMemorySpecialtyIndex",
    "Location",
    "Patient",
    "Provider",
    "Rule",
    "ScoredSpecialty",
    "SpecialtyIndex",
    "Violation",
    "bookable_specialties",
    "build_index",
    "distinct_locations",
    "distinct_providers",
    "distinct_types",
    "find_bookable",
    "is_bookable",
    "load_aliases",
    "needs_location_disambiguation",
    "unstaffed_appointment_types",
    "violations",
    "why_not",
]
