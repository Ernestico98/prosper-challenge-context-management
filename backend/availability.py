#
# Mocked availability.
#
# The challenge says to mock scheduling data: the interesting problem is
# navigating the catalog, not integrating a calendar. So this module invents
# slots — but *deterministically*, seeded from (provider, location, date), so a
# demo replays identically and the evals can assert on real values.
#
# Two behaviours are modelled because they change the conversation:
#   - A provider who practises at several sites is only at ONE of them on any
#     given day. That is what makes "book with Dr. Chen" genuinely need a
#     location, rather than it being a formality.
#   - Most of the grid is already taken, so "the soonest appointment" is a real
#     search across combinations instead of always being tomorrow at 08:00.
#

import hashlib
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta
from random import Random
from typing import Optional

BUSY_RATIO = 0.7          # fraction of the grid already booked
GRID_MINUTES = 15         # appointments start on a quarter-hour
HORIZON_DAYS = 21         # how far ahead we invent availability

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_OF_DAY = {
    "morning": (time(0, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(23, 59)),
}


@dataclass(frozen=True)
class Slot:
    start: datetime
    duration_min: int
    provider_id: str
    location_id: str

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_min)

    def spoken(self) -> str:
        """Phrasing meant to be read aloud, not printed."""
        hour = self.start.strftime("%I:%M %p").lstrip("0")
        return f"{WEEKDAY_NAMES[self.start.weekday()]} the {self.start.day} at {hour}"

    def iso(self) -> str:
        return self.start.isoformat(timespec="minutes")


# ---- deterministic randomness ----------------------------------------------


def _seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def working_location_on(provider, day: Date) -> str:
    """Which of a provider's sites they are at on ``day``.

    Single-site providers are trivially there; multi-site providers rotate
    deterministically, which is what forces a real location conversation.
    """
    sites = sorted(provider.location_ids)
    if len(sites) == 1:
        return sites[0]
    return sites[_seed(provider.id, day.isoformat()) % len(sites)]


def _parse_hours(hours: str) -> tuple:
    """Parse 'Mon-Fri 8:00-17:00' into (open_weekdays, open_time, close_time).

    Falls back to weekdays 08:00-17:00 for anything it doesn't recognise —
    availability is mocked, so a parse failure must not take the demo down.
    """
    default = (frozenset(range(5)), time(8, 0), time(17, 0))
    try:
        days_part, times_part = hours.split()
        start_str, end_str = times_part.split("-")
        abbrev = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if "-" in days_part:
            first, last = days_part.split("-")
            days = frozenset(range(abbrev.index(first), abbrev.index(last) + 1))
        else:
            days = frozenset(abbrev.index(d) for d in days_part.split(","))

        def _t(value):
            hour, minute = value.split(":")
            return time(int(hour), int(minute))

        return days, _t(start_str), _t(end_str)
    except (ValueError, IndexError):
        return default


# ---- slot generation -------------------------------------------------------


def get_slots(
    combo,
    *,
    today: Optional[Date] = None,
    days: int = HORIZON_DAYS,
    earliest_date: Optional[Date] = None,
    weekdays: Optional[list] = None,
    time_of_day: Optional[str] = None,
    limit: int = 20,
) -> list:
    """Free slots for one bookable combination, soonest first.

    ``combo`` is a ``catalog.BookableCombo``: the triple has already been
    validated against every booking policy, so this function only ever answers
    "when", never "whether".
    """
    today = today or Date.today()
    provider, location = combo.provider, combo.location
    duration = combo.appointment_type.duration_min

    open_days, opens, closes = _parse_hours(location.hours)
    wanted_weekdays = _normalise_weekdays(weekdays)
    window = TIME_OF_DAY.get(time_of_day) if time_of_day else None

    found = []
    for offset in range(1, days + 1):      # never today: same-day is not offered
        day = today + timedelta(days=offset)
        if earliest_date and day < earliest_date:
            continue
        if day.weekday() not in open_days:
            continue
        if wanted_weekdays is not None and day.weekday() not in wanted_weekdays:
            continue
        if working_location_on(provider, day) != location.id:
            continue

        rng = Random(_seed(provider.id, location.id, day.isoformat()))
        cursor = datetime.combine(day, opens)
        closing = datetime.combine(day, closes)

        while cursor + timedelta(minutes=duration) <= closing:
            taken = rng.random() < BUSY_RATIO
            if not taken and _within(cursor, window):
                found.append(Slot(cursor, duration, provider.id, location.id))
                if len(found) >= limit:
                    return found
            cursor += timedelta(minutes=GRID_MINUTES)

    return found


def _within(moment: datetime, window) -> bool:
    if window is None:
        return True
    start, end = window
    return start <= moment.time() < end


def _normalise_weekdays(weekdays) -> Optional[frozenset]:
    """Accept 'Tuesday', 'tue', or 1 — the LLM will produce all three."""
    if not weekdays:
        return None
    abbrev = [name[:3].lower() for name in WEEKDAY_NAMES]
    resolved = set()
    for value in weekdays:
        if isinstance(value, int):
            resolved.add(value % 7)
            continue
        key = str(value).strip()[:3].lower()
        if key in abbrev:
            resolved.add(abbrev.index(key))
    return frozenset(resolved) or None


def first_slot(combo, **kwargs) -> Optional[Slot]:
    """The soonest slot for a combination, or None within the horizon."""
    slots = get_slots(combo, limit=1, **kwargs)
    return slots[0] if slots else None
