#
# Scheduling tools — thin adapters between the LLM and the catalog domain.
#
# Everything correctness-critical lives in catalog/. This layer only does the
# three things the domain must NOT do:
#
#   1. Budget context. The domain returns every legal option; here we decide
#      how many are worth spending tokens on and how many a person can hold in
#      their head when they hear them read out.
#   2. Speak. Names instead of ids, one line per option.
#   3. Recover. A tool must never answer with an empty result the agent cannot
#      act on: a bad specialty comes back as suggestions, an impossible request
#      comes back with the reason, so the model can self-correct mid-turn
#      without the caller noticing.
#

from datetime import date
from typing import Optional

from availability import get_slots
from catalog import (
    Patient,
    distinct_types,
    find_bookable,
    needs_location_disambiguation,
)

from .registry import register

# ---- shared helpers --------------------------------------------------------


def _state(flow_manager) -> dict:
    return getattr(flow_manager, "state", None) if flow_manager is not None else None


def _patient(flow_manager) -> Patient:
    """Patient facts collected so far by the graph's edges.

    Absent facts stay None, which the solver reads as "not asked yet" rather
    than as a rejection — an unanswered question must not silently prune the
    catalog.
    """
    state = _state(flow_manager) or {}
    return Patient(
        is_new=state.get("is_new_patient"),
        has_referral=state.get("has_referral"),
    )


def _appt_summary(appt) -> dict:
    summary = {
        "id": appt.id,
        "name": appt.name,
        "specialty": appt.specialty,
        "duration_min": appt.duration_min,
    }
    if appt.requires_referral:
        summary["requires_referral"] = True
    if not appt.new_patients_allowed:
        summary["new_patients_allowed"] = False
    if appt.required_capability:
        summary["only_at_sites_with"] = appt.required_capability
    return summary


def _suggest_specialties(ctx, requested: str, fuzzy: list) -> list:
    """Best-effort recovery for a specialty the catalog doesn't have.

    Character similarity alone is close to useless here — "Traumatology" is
    nearest to "Dermatology" by edit distance and to Orthopedics by meaning. The
    retrieval index already knows the lay vocabulary, so ask it first and keep
    the fuzzy match only as a fallback.
    """
    from_index = [s.name for s in ctx.index.search(requested, limit=3)]
    return from_index or fuzzy


def _resolve_type(ctx, value: str):
    """Accept an id or a name — the model produces both."""
    if not value:
        return None
    if value in ctx.catalog.appointment_types:
        return ctx.catalog.appointment_types[value]
    lowered = value.strip().lower()
    for appt in ctx.catalog.appointment_types.values():
        if appt.name.lower() == lowered:
            return appt
    return None


# ---- 1. entering the hierarchy ---------------------------------------------


@register(
    "find_specialties",
    "Find which clinical specialties match what the caller describes. Pass their own "
    "words about the problem or the service they want, not a specialty name. Call this "
    "first, before listing any appointment types.",
    properties={
        "complaint": {
            "type": "string",
            "description": "What the caller said about their symptom, condition or "
            "the service they are asking for, in their own words.",
        }
    },
    required=["complaint"],
)
def find_specialties(args: dict, ctx, flow_manager=None) -> dict:
    results = ctx.index.search(args.get("complaint", ""), limit=ctx.max_options + 2)
    if not results:
        # An honest miss the agent can act on beats a confident wrong answer.
        return {
            "matches": [],
            "hint": "No specialty matched. Ask the caller to describe the problem "
            "differently, or which kind of doctor or service they had in mind.",
        }

    matches = [r.as_dict() for r in results[: ctx.max_options]]
    if matches[0].get("not_currently_offered"):
        return {
            "matches": matches,
            "hint": f"The clinic does not currently offer {matches[0]['specialty']} — "
            "no provider is available for it. Tell the caller plainly, and do not "
            "book one of the other matches as a substitute.",
        }
    return {"matches": matches}


@register(
    "list_appointment_types",
    "List the bookable appointment types within one specialty. Use after "
    "find_specialties has identified the specialty.",
    properties={
        "specialty": {
            "type": "string",
            "description": "Specialty name exactly as returned by find_specialties.",
        }
    },
    required=["specialty"],
)
def list_appointment_types(args: dict, ctx, flow_manager=None) -> dict:
    requested = args.get("specialty", "")
    exact, suggestions = ctx.catalog.resolve_specialty(requested)
    if not exact:
        # The recovery path: never an empty list. The model retries in the same
        # turn and the caller hears nothing.
        return {
            "error": "unknown_specialty",
            "requested": requested,
            "did_you_mean": _suggest_specialties(ctx, requested, suggestions),
        }

    combos = find_bookable(ctx.catalog, specialty=exact, patient=_patient(flow_manager))
    types = distinct_types(combos)
    if not types:
        return {
            "error": "nothing_bookable",
            "specialty": exact,
            "reason": "No provider currently offers appointments in this specialty.",
        }
    return {
        "specialty": exact,
        "appointment_types": [_appt_summary(a) for a in types],
    }


@register(
    "describe_appointment_type",
    "Explain one appointment type: how long it takes, whether it needs a referral, "
    "whether new patients can book it, and where it can be done.",
    properties={
        "appointment_type": {
            "type": "string",
            "description": "The appointment type id or its exact name.",
        }
    },
    required=["appointment_type"],
)
def describe_appointment_type(args: dict, ctx, flow_manager=None) -> dict:
    appt = _resolve_type(ctx, args.get("appointment_type", ""))
    if appt is None:
        return {"error": "unknown_appointment_type", "requested": args.get("appointment_type")}

    combos = find_bookable(ctx.catalog, appointment_type_id=appt.id)
    sites = sorted({c.location.name for c in combos})
    return {
        **_appt_summary(appt),
        "bookable_at": sites[: ctx.max_options + 2],
        "site_count": len(sites),
        "provider_count": len({c.provider.id for c in combos}),
    }


# ---- 2. finding an actual appointment --------------------------------------


@register(
    "find_appointments",
    "Find bookable appointments for a chosen appointment type. All preferences are "
    "optional — pass only what the caller has actually expressed. Call it again with "
    "different preferences whenever they want something else; it does not end the step.",
    properties={
        "appointment_type": {
            "type": "string",
            "description": "The chosen appointment type id or exact name.",
        },
        "provider_name": {
            "type": "string",
            "description": "Doctor the caller asked for, as they said it.",
        },
        "provider_id": {
            "type": "string",
            "description": "Exact provider_id, from the candidates of an earlier "
            "ambiguous_provider result. Prefer this over provider_name once you have it.",
        },
        "location_name": {
            "type": "string",
            "description": "Site, neighbourhood or street the caller asked for.",
        },
        "weekdays": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Days that work for them, e.g. ['Tuesday', 'Thursday'].",
        },
        "time_of_day": {
            "type": "string",
            "enum": ["morning", "afternoon", "evening"],
            "description": "Part of the day they prefer.",
        },
        "language": {
            "type": "string",
            "description": "Language the caller would like the doctor to speak, if they "
            "said so. Used to order the options, never to rule any out.",
        },
        "rank_by": {
            "type": "string",
            "enum": ["soonest", "provider", "location"],
            "description": "What matters most to them. Defaults to soonest.",
        },
    },
    required=["appointment_type"],
)
def find_appointments(args: dict, ctx, flow_manager=None) -> dict:
    appt = _resolve_type(ctx, args.get("appointment_type", ""))
    if appt is None:
        return {"error": "unknown_appointment_type", "requested": args.get("appointment_type")}

    patient = _patient(flow_manager)
    dropped = []

    if args.get("provider_id") in ctx.catalog.providers:
        # Picked from a list we produced, so there is nothing left to resolve.
        provider_ids = {args["provider_id"]}
    else:
        provider_ids, provider_error, note = _match_providers(ctx, args.get("provider_name"))
        if provider_error:
            return provider_error
        if note:
            dropped.append(note)

    location_ids, note = _match_locations(ctx, args.get("location_name"))
    if note:
        dropped.append(note)

    combos = _combos_for(ctx, appt, provider_ids, location_ids, patient)
    if not combos:
        return _explain_no_combinations(ctx, appt, provider_ids, location_ids, patient, args)

    offers = _build_offers(ctx, combos, args)
    if not offers:
        return {
            "error": "no_availability",
            "appointment_type": appt.name,
            "reason": "Those combinations have nothing free in the next three weeks "
            "with the requested days or times.",
            "hint": "Ask whether other days, times, sites or providers would work.",
        }

    _remember_offers(flow_manager, offers)
    result = {
        "appointment_type": appt.name,
        "options": [_offer_summary(o) for o in offers],
        "hint": "Read these out and pass the chosen option_id to book_appointment. "
        "Never say the option_id aloud.",
    }
    if dropped:
        # Say it out loud rather than quietly ignoring what they asked for.
        result["ignored_preferences"] = dropped
        result["hint"] += " Tell the caller you could not match " + \
            " and ".join(dropped) + ", so these are the options without it."
    return result


def _match_providers(ctx, provider_name: Optional[str]):
    """Returns (provider_ids, error, dropped_note).

    Ambiguous is not the same as unknown. Two doctors sharing a name is a
    question only the caller can settle, so that stops and asks. A phrase that
    is not a name at all cannot be settled by asking, so it is dropped.
    """
    if not provider_name:
        return None, None, None
    matches = ctx.catalog.resolve_provider_name(provider_name)
    if not matches:
        # "my usual family doctor" is not a name, and no amount of asking will
        # turn it into one. A preference nobody can resolve is dropped, not
        # turned into a dead end the agent asks its way around in circles.
        return None, None, f"the doctor {provider_name!r}"
    if len(matches) > 1:
        # Duplicate names are a designed trap in this catalog: a name is not a
        # key, so hand back the candidates and let the conversation resolve it.
        return None, {
            "error": "ambiguous_provider",
            "requested": provider_name,
            "candidates": [
                {
                    "provider_id": p.id,
                    "name": p.display_name,
                    "specialty": p.specialty,
                    "sites": sorted(ctx.catalog.locations[l].name for l in p.location_ids)[:3],
                }
                for p in matches[: ctx.max_options + 2]
            ],
            "hint": "Ask which one they mean, by specialty or site, then call again "
            "passing their provider_id.",
        }, None
    return {matches[0].id}, None, None


def _match_locations(ctx, location_name: Optional[str]):
    if not location_name:
        return None, None
    matches = ctx.catalog.resolve_location_name(location_name)
    if not matches:
        return None, f"the site {location_name!r}"  # widen, do not block
    return {l.id for l in matches}, None


def _combos_for(ctx, appt, provider_ids, location_ids, patient) -> list:
    combos = find_bookable(ctx.catalog, appointment_type_id=appt.id, patient=patient)
    if provider_ids is not None:
        combos = [c for c in combos if c.provider.id in provider_ids]
    if location_ids is not None:
        combos = [c for c in combos if c.location.id in location_ids]
    return combos


def _explain_no_combinations(ctx, appt, provider_ids, location_ids, patient, args) -> dict:
    """Say WHY nothing matched, by relaxing one constraint at a time.

    "Dr. Chen doesn't do MRIs at that site" is something the agent can act on;
    an empty list is not.
    """
    reasons = []
    if provider_ids and _combos_for(ctx, appt, None, location_ids, patient):
        reasons.append(f"{args.get('provider_name')} cannot be booked for {appt.name} there.")
    if location_ids and _combos_for(ctx, appt, provider_ids, None, patient):
        detail = f"{appt.name} is not available at {args.get('location_name')}"
        if appt.required_capability:
            detail += f" — it needs a site with {appt.required_capability}"
        reasons.append(detail + ".")
    if not reasons and find_bookable(ctx.catalog, appointment_type_id=appt.id):
        if patient.is_new:
            reasons.append(f"{appt.name} is not open to new patients with these providers.")
        elif patient.has_referral is False and appt.requires_referral:
            reasons.append(f"{appt.name} needs a referral on file first.")

    alternatives = find_bookable(ctx.catalog, appointment_type_id=appt.id, patient=patient)
    return {
        "error": "no_bookable_combination",
        "appointment_type": appt.name,
        "reasons": reasons or ["That combination is not bookable."],
        "available_instead": {
            "providers": sorted({c.provider.display_name for c in alternatives})[: ctx.max_options],
            "sites": sorted({c.location.name for c in alternatives})[: ctx.max_options],
        },
    }


def _build_offers(ctx, combos, args) -> list:
    """Fully-specified (provider, site, time) offers, ranked and deduped.

    Slots are searched across ALL viable combinations rather than within one
    already-chosen provider: "whatever is soonest" is a constraint, not a step
    the caller skipped.
    """
    rank_by = args.get("rank_by") or "soonest"
    weekdays = args.get("weekdays")
    time_of_day = args.get("time_of_day")
    language = (args.get("language") or "").strip().title()
    today = ctx.today or date.today()

    candidates = []
    for combo in combos[: ctx.max_combos_probed]:
        slots = get_slots(
            combo,
            today=today,
            weekdays=weekdays,
            time_of_day=time_of_day,
            limit=2,
        )
        for slot in slots:
            candidates.append((combo, slot))

    # A wish for a doctor who speaks a given language is a PREFERENCE, not a
    # policy: nothing in the catalog says such a booking is illegal. So it
    # orders the options and never removes any. Enforcing it as a filter turned
    # a perfectly bookable echocardiogram into "no availability" simply because
    # the caller happened to be speaking Spanish.
    def speaks_first(pair):
        return 0 if language and language in pair[0].provider.languages else 1

    if rank_by == "provider":
        candidates.sort(key=lambda cs: (speaks_first(cs), cs[0].provider.name, cs[1].start))
    elif rank_by == "location":
        candidates.sort(key=lambda cs: (speaks_first(cs), cs[0].location.name, cs[1].start))
    else:
        candidates.sort(key=lambda cs: (speaks_first(cs), cs[1].start))

    # One option per provider, and no two options at the same moment: three
    # near-identical choices read aloud is worse than two distinct ones.
    offers, seen_providers, seen_times = [], set(), set()
    for combo, slot in candidates:
        if combo.provider.id in seen_providers or slot.iso() in seen_times:
            continue
        seen_providers.add(combo.provider.id)
        seen_times.add(slot.iso())
        offers.append((f"opt_{len(offers) + 1}", combo, slot))
        if len(offers) >= ctx.max_options:
            break
    return offers


def _offer_summary(offer) -> dict:
    option_id, combo, slot = offer
    return {
        "option_id": option_id,
        "provider": combo.provider.display_name,
        "specialty": combo.provider.specialty,
        "site": combo.location.name,
        "when": slot.spoken(),
        "starts_at": slot.iso(),
    }


def _remember_offers(flow_manager, offers) -> None:
    state = _state(flow_manager)
    if state is None:
        return
    state["offers"] = {
        option_id: {
            "appointment_type_id": combo.appointment_type.id,
            "provider_id": combo.provider.id,
            "location_id": combo.location.id,
            "starts_at": slot.iso(),
        }
        for option_id, combo, slot in offers
    }


# ---- 3. booking ------------------------------------------------------------


@register(
    "book_appointment",
    "Book one of the options returned by find_appointments. Pass the option_id of the "
    "one the caller chose.",
    properties={
        "option_id": {
            "type": "string",
            "description": "The option_id from the most recent find_appointments result.",
        },
        "patient_name": {"type": "string", "description": "Caller's full name."},
    },
    required=["option_id"],
)
def book_appointment(args: dict, ctx, flow_manager=None) -> dict:
    state = _state(flow_manager) or {}
    offers = state.get("offers") or {}
    option_id = args.get("option_id")
    chosen = offers.get(option_id)
    if not chosen:
        return {
            "error": "unknown_option",
            "requested": option_id,
            "hint": "Call find_appointments again and offer the caller fresh options.",
        }

    # Re-validate through the same solver that produced the offer. The model
    # picks an opaque id from a list we just generated, so it cannot compose an
    # illegal combination — this is belt and braces on top of that.
    patient = _patient(flow_manager)
    combos = find_bookable(
        ctx.catalog,
        appointment_type_id=chosen["appointment_type_id"],
        provider_id=chosen["provider_id"],
        location_id=chosen["location_id"],
        patient=patient,
    )
    if len(combos) != 1:
        return {
            "error": "no_longer_bookable",
            "hint": "Something changed. Call find_appointments again.",
        }

    combo = combos[0]
    booking = {
        "reference": f"APT-{1000 + len(ctx.bookings)}",
        "patient_name": args.get("patient_name") or state.get("full_name"),
        "appointment_type": combo.appointment_type.name,
        "provider": combo.provider.display_name,
        "site": combo.location.name,
        "address": combo.location.address,
        "starts_at": chosen["starts_at"],
        "duration_min": combo.appointment_type.duration_min,
    }
    ctx.bookings.append(booking)
    state["booking"] = booking
    return {"status": "booked", **booking}


# ---- 4. answering questions ------------------------------------------------


@register(
    "find_providers",
    "Look up doctors: by name, by specialty, by site, or by language spoken. Use it to "
    "answer questions about who works where, and to resolve a name the caller gives.",
    properties={
        "name": {"type": "string", "description": "Doctor's name, as the caller said it."},
        "specialty": {"type": "string", "description": "Specialty to filter by."},
        "location_name": {"type": "string", "description": "Site or neighbourhood."},
        "language": {"type": "string", "description": "Language the doctor should speak."},
    },
)
def find_providers(args: dict, ctx, flow_manager=None) -> dict:
    providers = list(ctx.catalog.providers.values())
    if args.get("name"):
        providers = ctx.catalog.resolve_provider_name(args["name"])
        if not providers:
            return {"error": "unknown_provider", "requested": args["name"]}
    if args.get("specialty"):
        exact, suggestions = ctx.catalog.resolve_specialty(args["specialty"])
        if not exact:
            return {"error": "unknown_specialty", "did_you_mean": suggestions}
        providers = [p for p in providers if p.specialty == exact]
    if args.get("location_name"):
        matches = ctx.catalog.resolve_location_name(args["location_name"])
        if not matches:
            return {"error": "unknown_location", "requested": args["location_name"]}
        wanted = {l.id for l in matches}
        providers = [p for p in providers if p.location_ids & wanted]
    if args.get("language"):
        language = args["language"].strip().title()
        providers = [p for p in providers if language in p.languages]

    if not providers:
        return {"providers": [], "hint": "Nobody matches. Offer to relax one condition."}

    return {
        "providers": [
            {
                "name": p.display_name,
                "specialty": p.specialty,
                "sites": sorted(ctx.catalog.locations[l].name for l in p.location_ids),
                "accepting_new_patients": p.accepting_new_patients,
                "languages": sorted(p.languages),
                "needs_site_disambiguation": needs_location_disambiguation(p),
            }
            for p in providers[: ctx.max_options + 2]
        ],
        "total_matching": len(providers),
    }


@register(
    "get_location_info",
    "Address, phone, opening hours and on-site services for the clinic's sites. Omit "
    "the argument to list them all.",
    properties={
        "location_name": {
            "type": "string",
            "description": "Site name, neighbourhood or street. Omit to list every site.",
        }
    },
)
def get_location_info(args: dict, ctx, flow_manager=None) -> dict:
    requested = args.get("location_name")
    if requested:
        matches = ctx.catalog.resolve_location_name(requested)
        if not matches:
            return {"error": "unknown_location", "requested": requested}
    else:
        matches = sorted(ctx.catalog.locations.values(), key=lambda l: l.name)

    return {
        "locations": [
            {
                "name": l.name,
                "address": f"{l.address}, {l.city}",
                "phone": l.phone,
                "hours": l.hours,
                "on_site_services": sorted(l.capabilities) or ["consultations only"],
            }
            for l in matches
        ]
    }
