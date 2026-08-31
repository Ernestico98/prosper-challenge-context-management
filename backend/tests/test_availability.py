#
# Availability tests.
#
# Availability is mocked, but it is mocked *deterministically*, and the evals
# and the demo both depend on that. These tests pin the properties the rest of
# the system is allowed to rely on.
#

import unittest
from datetime import date, timedelta

from availability import get_slots, working_location_on
from catalog import Catalog, find_bookable

TODAY = date(2026, 3, 2)  # a Monday, fixed so the suite never depends on "now"


class AvailabilityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog.load()
        cls.combo = find_bookable(cls.catalog, specialty="Cardiology")[0]


class TestDeterminism(AvailabilityTestCase):
    def test_same_inputs_give_same_slots(self):
        first = get_slots(self.combo, today=TODAY)
        second = get_slots(self.combo, today=TODAY)
        self.assertEqual([s.iso() for s in first], [s.iso() for s in second])

    def test_different_combos_give_different_slots(self):
        combos = find_bookable(self.catalog, specialty="Cardiology")[:2]
        a = {s.iso() for s in get_slots(combos[0], today=TODAY)}
        b = {s.iso() for s in get_slots(combos[1], today=TODAY)}
        self.assertNotEqual(a, b)


class TestOpeningHours(AvailabilityTestCase):
    def test_slots_fall_within_opening_hours(self):
        for slot in get_slots(self.combo, today=TODAY, limit=50):
            self.assertGreaterEqual(slot.start.hour, 8)
            self.assertLessEqual(slot.end.hour, 17)
            self.assertLess(slot.start.weekday(), 5, "clinic is closed at weekends")

    def test_slot_length_matches_appointment_duration(self):
        for slot in get_slots(self.combo, today=TODAY, limit=10):
            self.assertEqual(self.combo.appointment_type.duration_min, slot.duration_min)

    def test_no_same_day_appointments(self):
        for slot in get_slots(self.combo, today=TODAY, limit=10):
            self.assertGreater(slot.start.date(), TODAY)


class TestPreferences(AvailabilityTestCase):
    def test_time_of_day_filter(self):
        for slot in get_slots(self.combo, today=TODAY, time_of_day="afternoon", limit=10):
            self.assertGreaterEqual(slot.start.hour, 12)

    def test_weekday_filter_accepts_the_shapes_an_llm_produces(self):
        for weekdays in (["Tuesday"], ["tue"], [1]):
            slots = get_slots(self.combo, today=TODAY, weekdays=weekdays, limit=5)
            self.assertTrue(slots, f"no slots for {weekdays!r}")
            for slot in slots:
                self.assertEqual(1, slot.start.weekday())

    def test_earliest_date_filter(self):
        cutoff = TODAY + timedelta(days=10)
        for slot in get_slots(self.combo, today=TODAY, earliest_date=cutoff, limit=10):
            self.assertGreaterEqual(slot.start.date(), cutoff)


class TestProviderRotation(AvailabilityTestCase):
    def test_multi_site_provider_is_at_one_site_per_day(self):
        """What makes 'book with Dr. Chen' genuinely need a location."""
        provider = self.catalog.providers["prov_000"]
        self.assertGreater(len(provider.location_ids), 1)

        visited = set()
        for offset in range(21):
            day = TODAY + timedelta(days=offset)
            site = working_location_on(provider, day)
            self.assertIn(site, provider.location_ids)
            visited.add(site)

        self.assertGreater(len(visited), 1, "expected rotation across sites")

    def test_slots_only_on_days_the_provider_is_at_that_site(self):
        for slot in get_slots(self.combo, today=TODAY, limit=20):
            self.assertEqual(
                self.combo.location.id,
                working_location_on(self.combo.provider, slot.start.date()),
            )

    def test_single_site_provider_is_always_there(self):
        provider = next(
            p for p in self.catalog.providers.values() if len(p.location_ids) == 1
        )
        only = next(iter(provider.location_ids))
        for offset in range(14):
            self.assertEqual(only, working_location_on(provider, TODAY + timedelta(days=offset)))


if __name__ == "__main__":
    unittest.main()
