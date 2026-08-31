#
# Booking policies — the clinic's rules, as code.
#
# These are the rules stated in catalog.json's `policies` array -- exactly
# those, and nothing invented. They live here, in one place, because they are
# cross-cutting: the same rule constrains searching for options, offering slots
# and confirming a booking. If each tool re-implemented them they would drift,
# and the drift would surface as an appointment the clinic cannot honour.
#
# A policy decides what is LEGAL and is never relaxed. A caller's preference
# (this doctor, that site, the soonest slot) decides what is DESIRABLE and is
# always relaxable. Keeping the two apart is what makes it safe to widen a
# search when nothing matches -- and a preference that sneaks into the policy
# list stops being a preference: it silently removes options the clinic would
# happily book.
#
# Every rule is an explainable predicate: on rejection it can say WHY, which is
# what lets the agent respond with "Dr. Chen doesn't practise at that location"
# instead of an unhelpful "no results".
#

from dataclasses import dataclass
from typing import Callable, Optional

from .store import AppointmentType, Location, Provider


@dataclass(frozen=True)
class Patient:
    """What we know about the caller so far.

    ``None`` means "not asked yet" and is deliberately distinct from ``False``:
    an unknown fact must not silently filter options away. Rules that need a
    fact they don't have abstain rather than reject.
    """

    is_new: Optional[bool] = None
    has_referral: Optional[bool] = None


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


@dataclass(frozen=True)
class Rule:
    """One booking policy.

    ``check`` returns None when the combination is acceptable (or when the rule
    cannot be evaluated yet), or a Violation explaining the rejection.
    """

    code: str
    description: str
    check: Callable  # (AppointmentType, Provider, Location, Patient) -> Violation | None


# ---- the rules that filter combinations ------------------------------------


def _provider_at_location(appt, provider, location, patient):
    if location.id not in provider.location_ids:
        return Violation(
            "provider_not_at_location",
            f"{provider.name} does not practise at {location.name}.",
        )
    return None


def _provider_offers_type(appt, provider, location, patient):
    if appt.id not in provider.appointment_type_ids:
        return Violation(
            "provider_does_not_offer_type",
            f"{provider.name} is not booked for {appt.name}.",
        )
    return None


def _location_has_capability(appt, provider, location, patient):
    required = appt.required_capability
    if required and required not in location.capabilities:
        return Violation(
            "location_lacks_capability",
            f"{location.name} does not have {required}, which {appt.name} requires.",
        )
    return None


def _referral_on_file(appt, provider, location, patient):
    # Unknown referral status is not a rejection: the agent asks only once the
    # appointment type is known to need one.
    if appt.requires_referral and patient.has_referral is False:
        return Violation(
            "referral_required",
            f"{appt.name} requires a referral on file.",
        )
    return None


def _new_patient_allowed(appt, provider, location, patient):
    if patient.is_new is not True:
        return None
    if not appt.new_patients_allowed:
        return Violation(
            "type_closed_to_new_patients",
            f"{appt.name} is not available to new patients.",
        )
    if not provider.accepting_new_patients:
        return Violation(
            "provider_closed_to_new_patients",
            f"{provider.name} is not accepting new patients.",
        )
    return None


POLICIES = [
    Rule(
        "provider_not_at_location",
        "A provider can only be booked at a location listed in their location_ids.",
        _provider_at_location,
    ),
    Rule(
        "provider_does_not_offer_type",
        "A provider can only be booked for an appointment type listed in their "
        "appointment_type_ids.",
        _provider_offers_type,
    ),
    Rule(
        "location_lacks_capability",
        "An appointment type with a required_capability can only be booked at a "
        "location whose capabilities include it.",
        _location_has_capability,
    ),
    Rule(
        "referral_required",
        "Appointment types with requires_referral need a referral on file.",
        _referral_on_file,
    ),
    Rule(
        "new_patient_restrictions",
        "A new patient may only book types where new_patients_allowed, and only "
        "with providers who are accepting_new_patients.",
        _new_patient_allowed,
    ),
]

# The sixth catalog policy — "when the caller names a provider who practises at
# multiple locations, the location must be disambiguated" — is not a filter but
# a conversational obligation. It is enforced by returning fully-specified
# offers (each naming its location) rather than by rejecting combinations, so it
# has no Rule here. See `needs_location_disambiguation` below.


def violations(
    appt: AppointmentType,
    provider: Provider,
    location: Location,
    patient: Patient = Patient(),
) -> list:
    """Every rule this combination breaks. Empty list means bookable."""
    found = []
    for rule in POLICIES:
        violation = rule.check(appt, provider, location, patient)
        if violation:
            found.append(violation)
    return found


def is_bookable(
    appt: AppointmentType,
    provider: Provider,
    location: Location,
    patient: Patient = Patient(),
) -> bool:
    return not violations(appt, provider, location, patient)


def needs_location_disambiguation(provider: Provider) -> bool:
    """Policy 6: a named provider practising at several sites needs a location."""
    return len(provider.location_ids) > 1
