#
# Catalog store — the catalog as data, plus the indices that make it queryable.
#
# This layer knows NOTHING about booking policies, tools or the LLM. It loads
# catalog.json into typed records and builds the inverted indices the query
# layer needs, so no caller ever has to scan the raw lists.
#
#   catalog.json  ->  Catalog (records + indices)  ->  policies.py  ->  query.py
#

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Union

DEFAULT_CATALOG = Path(__file__).parent.parent / "data" / "catalog.json"


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    address: str
    city: str
    phone: str
    hours: str
    capabilities: frozenset  # frozenset[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Location":
        return cls(
            id=d["id"],
            name=d["name"],
            address=d["address"],
            city=d["city"],
            phone=d["phone"],
            hours=d["hours"],
            capabilities=frozenset(d.get("capabilities", [])),
        )


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    title: str
    specialty: str
    location_ids: frozenset
    accepting_new_patients: bool
    languages: frozenset
    appointment_type_ids: frozenset

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.title}"

    @classmethod
    def from_dict(cls, d: dict) -> "Provider":
        return cls(
            id=d["id"],
            name=d["name"],
            title=d["title"],
            specialty=d["specialty"],
            location_ids=frozenset(d["location_ids"]),
            accepting_new_patients=d["accepting_new_patients"],
            languages=frozenset(d.get("languages", [])),
            appointment_type_ids=frozenset(d["appointment_type_ids"]),
        )


@dataclass(frozen=True)
class AppointmentType:
    id: str
    name: str
    specialty: str
    duration_min: int
    requires_referral: bool
    new_patients_allowed: bool
    required_capability: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "AppointmentType":
        return cls(
            id=d["id"],
            name=d["name"],
            specialty=d["specialty"],
            duration_min=d["duration_min"],
            requires_referral=d["requires_referral"],
            new_patients_allowed=d["new_patients_allowed"],
            required_capability=d.get("required_capability"),
        )


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("/", " ").split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


@dataclass
class Catalog:
    """Typed catalog records plus the inverted indices used by the query layer."""

    locations: dict = field(default_factory=dict)           # id -> Location
    providers: dict = field(default_factory=dict)           # id -> Provider
    appointment_types: dict = field(default_factory=dict)   # id -> AppointmentType
    policies: list = field(default_factory=list)            # human-readable rule text

    # ---- indices (built in __post_init__) ---------------------------------
    types_by_specialty: dict = field(default_factory=dict, repr=False)
    providers_by_type: dict = field(default_factory=dict, repr=False)
    providers_by_location: dict = field(default_factory=dict, repr=False)
    locations_by_capability: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for appt in self.appointment_types.values():
            self.types_by_specialty.setdefault(appt.specialty, []).append(appt.id)
            self.providers_by_type.setdefault(appt.id, [])
        for prov in self.providers.values():
            for type_id in prov.appointment_type_ids:
                self.providers_by_type.setdefault(type_id, []).append(prov.id)
            for loc_id in prov.location_ids:
                self.providers_by_location.setdefault(loc_id, []).append(prov.id)
        for loc in self.locations.values():
            for cap in loc.capabilities:
                self.locations_by_capability.setdefault(cap, []).append(loc.id)

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Catalog":
        return cls(
            locations={x["id"]: Location.from_dict(x) for x in d["locations"]},
            providers={x["id"]: Provider.from_dict(x) for x in d["providers"]},
            appointment_types={
                x["id"]: AppointmentType.from_dict(x) for x in d["appointment_types"]
            },
            policies=list(d.get("policies", [])),
        )

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_CATALOG) -> "Catalog":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- lookups -----------------------------------------------------------
    @property
    def specialties(self) -> list:
        """Specialties that actually have bookable appointment types.

        This is the ~65-token list the agent carries in its prompt: the top of
        the hierarchy the LLM narrows from before touching the catalog at all.
        """
        return sorted(self.types_by_specialty)

    def capabilities(self) -> list:
        return sorted(self.locations_by_capability)

    # ---- fuzzy resolution --------------------------------------------------
    # The LLM will say "Traumatology" or "the doctor Chen". Resolving that to
    # real ids is this layer's job; hallucinated input must degrade into
    # suggestions, never into an empty result the agent can't recover from.

    def resolve_specialty(self, text: str, limit: int = 3) -> tuple:
        """Return ``(exact_match_or_None, [close_alternatives])``."""
        if not text:
            return None, []
        target = _norm(text)
        for specialty in self.specialties:
            if _norm(specialty) == target:
                return specialty, []
        scored = sorted(
            ((_similarity(text, s), s) for s in self.specialties), reverse=True
        )
        return None, [s for score, s in scored[:limit] if score > 0.4]

    def resolve_provider_name(self, text: str, limit: int = 5) -> list:
        """Providers whose name matches ``text``; ambiguous names return many.

        Duplicate names ('Dr. Maria Garcia') are the point: this returns every
        candidate and lets the conversation disambiguate.
        """
        if not text:
            return []
        target = _norm(text)
        # Match the display name too: the tools show candidates as "Dr. Maria
        # Garcia, MD", so that string has to be usable as an answer. Advertising
        # a name the resolver then rejects leaves the model unable to
        # disambiguate with the very words it was given.
        exact = [
            p for p in self.providers.values()
            if target in (_norm(p.name), _norm(p.display_name))
        ]
        if exact:
            return exact
        substring = [
            p
            for p in self.providers.values()
            if target in _norm(p.name) or _norm(p.name).endswith(f" {target}")
        ]
        if substring:
            return substring[:limit]
        scored = sorted(
            ((_similarity(text, p.name), p.id) for p in self.providers.values()),
            reverse=True,
        )
        return [self.providers[pid] for score, pid in scored[:limit] if score > 0.6]

    def resolve_location_name(self, text: str, limit: int = 5) -> list:
        if not text:
            return []
        target = _norm(text)
        exact = [l for l in self.locations.values() if _norm(l.name) == target]
        if exact:
            return exact
        substring = [
            l
            for l in self.locations.values()
            if target in _norm(l.name) or target in _norm(l.address)
        ]
        if substring:
            return substring[:limit]
        scored = sorted(
            ((_similarity(text, l.name), l.id) for l in self.locations.values()),
            reverse=True,
        )
        return [self.locations[lid] for score, lid in scored[:limit] if score > 0.6]
