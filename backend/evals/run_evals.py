#
# Accuracy evals.
#
#     python backend/evals/run_evals.py --dry-run     list the scenarios, no API calls
#     python backend/evals/run_evals.py               run them all
#     python backend/evals/run_evals.py --only knee_mri_with_referral -v
#
# This is the evidence behind the word "reliably". Every scenario drives the
# same compiled graph the voice pipeline runs, over text, and asserts on the
# OUTCOME -- what was booked and whether it was legal -- rather than on wording.
#
# The assertions are deliberately about legality and routing, not about exact
# ids where several answers are genuinely correct. An eval that fails when the
# agent picks a different-but-valid provider teaches you nothing.
#

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv                             # noqa: E402

# Same .env bot.py uses, so the scripts run from the repo root or from backend/.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from agent_builder import AgentBuilder                     # noqa: E402
from catalog import Catalog, bookable_specialties, build_index  # noqa: E402
from evals.harness import HeadlessFlow                     # noqa: E402
from tools.context import SchedulingContext                # noqa: E402

SCENARIOS = Path(__file__).parent / "scenarios.json"
FLOW = Path(__file__).parent.parent / "data" / "agents" / "scheduler_flow.json"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---- assertions ------------------------------------------------------------


def check(expect: dict, transcript, ctx) -> list:
    """Returns the list of failures; empty means the scenario passed."""
    booking = ctx.bookings[-1] if ctx.bookings else None
    failures = []

    if "booked" in expect:
        if expect["booked"] and not booking:
            failures.append("expected a booking, none was made")
        if not expect["booked"] and booking:
            failures.append(f"expected no booking, got {booking['appointment_type']}")

    if booking:
        appt, provider, location = _resolve(ctx, booking)

        if "appointment_type_in" in expect and appt.id not in expect["appointment_type_in"]:
            failures.append(f"booked {appt.name} ({appt.id}), expected one of "
                            f"{expect['appointment_type_in']}")
        if "appointment_type_not_in" in expect and appt.id in expect["appointment_type_not_in"]:
            failures.append(f"booked {appt.name}, which was excluded")
        if "specialty_in" in expect and appt.specialty not in expect["specialty_in"]:
            failures.append(f"booked a {appt.specialty} type, expected one of "
                            f"{expect['specialty_in']}")
        if "location_capability" in expect:
            needed = expect["location_capability"]
            if needed not in location.capabilities:
                failures.append(f"booked at {location.name}, which has no {needed}")
        if "location_name_not" in expect and expect["location_name_not"] in location.name:
            failures.append(f"booked at {location.name}, which was excluded")
        if "provider_name_contains" in expect:
            if expect["provider_name_contains"].lower() not in provider.name.lower():
                failures.append(f"booked with {provider.name}, expected "
                                f"{expect['provider_name_contains']}")
        if "booked_on_weekday" in expect:
            from datetime import datetime

            weekday = WEEKDAYS[datetime.fromisoformat(booking["starts_at"]).weekday()]
            if weekday != expect["booked_on_weekday"]:
                failures.append(f"booked on {weekday}, expected {expect['booked_on_weekday']}")

        # Policy legality is re-checked here rather than trusted: the whole
        # claim is that an illegal booking cannot happen.
        failures += _policy_failures(expect, ctx, appt, provider, location, transcript)

    used = transcript.tools_used()
    for tool in expect.get("tools_used", []):
        if tool not in used:
            failures.append(f"never called {tool}")
    for tool, minimum in (expect.get("min_tool_calls") or {}).items():
        actual = sum(1 for name, _, _ in transcript.tool_calls if name == tool)
        if actual < minimum:
            failures.append(f"called {tool} {actual}x, expected at least {minimum}")
    if "saw_tool_error" in expect:
        errors = {r.get("error") for _, _, r in transcript.tool_calls if isinstance(r, dict)}
        if expect["saw_tool_error"] not in errors:
            failures.append(f"expected the agent to hit '{expect['saw_tool_error']}'")

    return failures


def _resolve(ctx, booking):
    appt = next(a for a in ctx.catalog.appointment_types.values()
                if a.name == booking["appointment_type"])
    provider = next(p for p in ctx.catalog.providers.values()
                    if p.display_name == booking["provider"])
    location = next(l for l in ctx.catalog.locations.values() if l.name == booking["site"])
    return appt, provider, location


def _policy_failures(expect, ctx, appt, provider, location, transcript) -> list:
    failures = []
    if location.id not in provider.location_ids:
        failures.append(f"ILLEGAL: {provider.name} does not practise at {location.name}")
    if appt.id not in provider.appointment_type_ids:
        failures.append(f"ILLEGAL: {provider.name} does not offer {appt.name}")
    if appt.required_capability and appt.required_capability not in location.capabilities:
        failures.append(f"ILLEGAL: {location.name} lacks {appt.required_capability}")

    if expect.get("new_patient_legal"):
        if not appt.new_patients_allowed:
            failures.append(f"ILLEGAL: {appt.name} is closed to new patients")
        if not provider.accepting_new_patients:
            failures.append(f"ILLEGAL: {provider.name} is not accepting new patients")
    if expect.get("referral_legal") and appt.requires_referral:
        failures.append(f"ILLEGAL: booked {appt.name} without a referral")
    return failures


# ---- runner ----------------------------------------------------------------


async def _with_retry(fn, *fn_args, attempts: int = 3):
    """Retry rate limits, which are about the account and not the agent."""
    from openai import RateLimitError

    for attempt in range(attempts):
        try:
            return await fn(*fn_args)
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(20 * (attempt + 1))


async def run_scenario(scenario, catalog, index, args) -> tuple:
    from openai import OpenAI

    ctx = SchedulingContext(catalog=catalog, index=index)
    builder = AgentBuilder.from_json(
        FLOW,
        tool_context=ctx,
        template_values={
            "specialties": ", ".join(bookable_specialties(catalog)),
            "clinic_name": "Prosper Clinic",
            "provider_preference": "nobody in particular",
            "location_preference": "no site in particular",
        },
    )
    flow = HeadlessFlow(builder, ctx, OpenAI(), model=args.model, temperature=args.temperature)
    transcript = await flow.run(scenario["script"])
    return check(scenario["expect"], transcript, ctx), transcript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="scenario id (repeatable)")
    parser.add_argument("--limit", type=int, help="run at most N scenarios")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Pinned at 0 so results are reproducible.")
    parser.add_argument("--dry-run", action="store_true", help="list scenarios, call nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="print transcripts")
    args = parser.parse_args()

    scenarios = json.loads(SCENARIOS.read_text())["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in set(args.only)]
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.dry_run:
        print(f"{len(scenarios)} scenarios (no API calls made):\n")
        for scenario in scenarios:
            print(f"  {scenario['id']}\n      {scenario['why']}")
        print(f"\nRunning them would make roughly {len(scenarios) * 8} chat completions.")
        return 0

    catalog = Catalog.load()
    index = build_index(catalog)

    passed, failed, errored, tokens = 0, [], [], 0
    for scenario in scenarios:
        try:
            failures, transcript = asyncio.run(
                _with_retry(run_scenario, scenario, catalog, index, args)
            )
        except Exception as error:  # noqa: BLE001
            # A suite that aborts on the first exception tells you about one
            # scenario and nothing about the other fifteen. Infrastructure
            # errors are also kept out of the pass rate: a 429 says nothing
            # about whether the agent books the right appointment.
            print(f"[ERROR] {scenario['id']}: {type(error).__name__}: {error}", flush=True)
            errored.append(scenario["id"])
            continue

        tokens += transcript.prompt_tokens
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] {scenario['id']}  "
              f"({len(transcript.tool_calls)} tool calls, "
              f"{transcript.prompt_tokens:,} prompt tokens)", flush=True)
        for failure in failures:
            print(f"         - {failure}")
        if args.verbose or failures:
            print("\n" + transcript.render() + "\n")
        if failures:
            failed.append(scenario["id"])
        else:
            passed += 1

    total = passed + len(failed)
    print(f"\n{passed}/{total} passed  |  {tokens:,} prompt tokens across the suite "
          f"({tokens // max(total, 1):,} per call)")
    if failed:
        print("failed: " + ", ".join(failed))
    if errored:
        print(f"not run ({len(errored)}): " + ", ".join(errored))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
