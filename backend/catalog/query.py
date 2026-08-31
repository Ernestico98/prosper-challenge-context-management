#
# The constraint solver — one query path for the whole catalog.
#
# Every scheduling tool is a projection of `find_bookable`. That is deliberate:
# a conversation accumulates constraints incrementally and out of order ("with
# Dr. Garcia" ... "near Mission" ... "oh, and I need an MRI"), so the primitive
# takes any subset of them and answers what is still viable. Because there is
# exactly one query path, the booking policies are applied in exactly one place
# and cannot be bypassed — including by `book_appointment`, which re-runs the
# solver to validate what the LLM chose.
#

from dataclasses import dataclass
from typing import Optional

from .policies import Patient, Violation, is_bookable, violations
from .store import AppointmentType, Catalog, Location, Provider


@dataclass(frozen=True)
class BookableCombo:
    """A legal (appointment type, provider, location) triple."""

    appointment_type: AppointmentType
    provider: Provider
    location: Location

    @property
    def key(self) -> tuple:
        return (self.appointment_type.id, self.provider.id, self.location.id)


def find_bookable(
    catalog: Catalog,
    *,
    specialty: Optional[str] = None,
    appointment_type_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    location_id: Optional[str] = None,
    patient: Patient = Patient(),
) -> list:
    """Combinations satisfying every policy, given the constraints known so far.

    Any argument left as ``None`` is simply unconstrained. Iteration walks from
    the most constrained axis outwards using the store's inverted indices, so
    the common case (a known appointment type) touches a handful of records
    rather than the 82x50x8 product.
    """
    type_ids = _candidate_type_ids(catalog, specialty, appointment_type_id)
    combos = []

    for type_id in type_ids:
        appt = catalog.appointment_types.get(type_id)
        if appt is None:
            continue

        # Capability gating prunes locations before we ever look at providers:
        # an MRI is bookable at 3 of 8 sites, so this is the cheapest cut.
        allowed_locations = _allowed_location_ids(catalog, appt, location_id)
        if not allowed_locations:
            continue

        for prov_id in catalog.providers_by_type.get(type_id, []):
            if provider_id and prov_id != provider_id:
                continue
            provider = catalog.providers[prov_id]

            for loc_id in sorted(provider.location_ids & allowed_locations):
                location = catalog.locations[loc_id]
                if is_bookable(appt, provider, location, patient):
                    combos.append(BookableCombo(appt, provider, location))

    combos.sort(key=lambda c: c.key)
    return combos


def _candidate_type_ids(
    catalog: Catalog, specialty: Optional[str], appointment_type_id: Optional[str]
) -> list:
    if appointment_type_id:
        return [appointment_type_id]
    if specialty:
        # Filter on the APPOINTMENT TYPE's specialty, never the provider's:
        # internists offer General and Family Medicine types, so filtering by
        # provider specialty loses most of primary care.
        return list(catalog.types_by_specialty.get(specialty, []))
    return list(catalog.appointment_types)


def _allowed_location_ids(
    catalog: Catalog, appt: AppointmentType, location_id: Optional[str]
) -> frozenset:
    if appt.required_capability:
        allowed = frozenset(catalog.locations_by_capability.get(appt.required_capability, []))
    else:
        allowed = frozenset(catalog.locations)
    if location_id:
        allowed &= {location_id}
    return allowed


# ---- projections -----------------------------------------------------------
# Views over a combo list. The tool layer decides how many of these to show the
# model; the domain always returns everything that is legal.


def distinct_types(combos: list) -> list:
    return _distinct(combos, lambda c: c.appointment_type, lambda a: a.name)


def distinct_providers(combos: list) -> list:
    return _distinct(combos, lambda c: c.provider, lambda p: p.name)


def distinct_locations(combos: list) -> list:
    return _distinct(combos, lambda c: c.location, lambda l: l.name)


def _distinct(combos: list, pick, sort_key) -> list:
    seen = {}
    for combo in combos:
        item = pick(combo)
        seen.setdefault(item.id, item)
    return sorted(seen.values(), key=sort_key)


def bookable_specialties(catalog: Catalog) -> list:
    """Specialties that actually have at least one legal combination.

    Distinct from ``Catalog.specialties``, which is every specialty appearing on
    an appointment type. The catalog advertises services nobody is currently
    staffed for (Ophthalmology, Physical Therapy and Urology have types but no
    provider offers them) — realistic, and a trap: routing a caller to a
    specialty with no bookable combination is a dead end. Note that the RETRIEVAL
    index still contains them (see catalog/index.py): hiding them there made the
    agent match the next-closest specialty and book the wrong thing. This list is
    for the places that need to enumerate what can actually be booked.
    """
    return [s for s in catalog.specialties if find_bookable(catalog, specialty=s)]


def unstaffed_appointment_types(catalog: Catalog) -> list:
    """Types no provider offers — answerable, but never bookable."""
    return sorted(
        (appt for appt in catalog.appointment_types.values()
         if not catalog.providers_by_type.get(appt.id)),
        key=lambda a: a.name,
    )


# ---- explainability --------------------------------------------------------


def why_not(
    catalog: Catalog,
    appointment_type_id: str,
    provider_id: str,
    location_id: str,
    patient: Patient = Patient(),
) -> list:
    """Why a specific combination is not bookable. Empty list means it is.

    Lets the agent say "Dr. Chen doesn't practise there" instead of going quiet
    when a caller asks for something impossible.
    """
    appt = catalog.appointment_types.get(appointment_type_id)
    provider = catalog.providers.get(provider_id)
    location = catalog.locations.get(location_id)

    missing = [
        (name, value)
        for name, value in (
            ("appointment type", appointment_type_id),
            ("provider", provider_id),
            ("location", location_id),
        )
        if not _resolved(catalog, name, value)
    ]
    if missing:
        return [
            Violation("unknown_entity", f"No such {name}: {value!r}.")
            for name, value in missing
        ]

    return violations(appt, provider, location, patient)


def _resolved(catalog: Catalog, kind: str, value: str) -> bool:
    lookup = {
        "appointment type": catalog.appointment_types,
        "provider": catalog.providers,
        "location": catalog.locations,
    }[kind]
    return value in lookup
