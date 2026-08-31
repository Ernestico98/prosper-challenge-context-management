#
# Domain-layer tests — the correctness-critical half of the system.
#
# No LLM, no network, no API keys: the booking policies are pure functions over
# the catalog, so the rules a voice agent is hardest to test on are covered here
# deterministically. Each policy gets a test, plus the catalog's deliberate
# messiness (duplicate names, capability gating, near-duplicate types).
#
#   python -m unittest discover -s backend/tests -t backend      (or: make test)
#

import unittest

from catalog import (
    Catalog,
    Patient,
    bookable_specialties,
    distinct_locations,
    distinct_providers,
    find_bookable,
    is_bookable,
    needs_location_disambiguation,
    unstaffed_appointment_types,
    why_not,
)
from catalog.policies import POLICIES
from catalog.store import Catalog as StoreCatalog


class CatalogTestCase(unittest.TestCase):
    """Shared fixture: the real catalog, loaded once for the whole suite."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog.load()

    def appt_named(self, name):
        matches = [a for a in self.catalog.appointment_types.values() if a.name == name]
        self.assertTrue(matches, f"no appointment type named {name!r}")
        return matches[0]

    def codes(self, violations):
        return {v.code for v in violations}


class TestPolicies(CatalogTestCase):
    def test_provider_must_practise_at_location(self):
        provider = self.catalog.providers["prov_000"]
        elsewhere = next(
            loc for loc in self.catalog.locations.values()
            if loc.id not in provider.location_ids
        )
        appt = self.catalog.appointment_types[sorted(provider.appointment_type_ids)[0]]

        self.assertFalse(is_bookable(appt, provider, elsewhere))
        self.assertIn(
            "provider_not_at_location",
            self.codes(why_not(self.catalog, appt.id, provider.id, elsewhere.id)),
        )

    def test_provider_must_offer_the_appointment_type(self):
        provider = self.catalog.providers["prov_000"]
        location = self.catalog.locations[sorted(provider.location_ids)[0]]
        unoffered = next(
            a for a in self.catalog.appointment_types.values()
            if a.id not in provider.appointment_type_ids
        )

        self.assertIn(
            "provider_does_not_offer_type",
            self.codes(why_not(self.catalog, unoffered.id, provider.id, location.id)),
        )

    def test_capability_gated_type_only_at_capable_locations(self):
        mri = self.appt_named("MRI - Knee")
        self.assertEqual(mri.required_capability, "imaging")

        combos = find_bookable(self.catalog, appointment_type_id=mri.id)
        self.assertTrue(combos, "MRI - Knee should be bookable somewhere")
        for location in distinct_locations(combos):
            self.assertIn("imaging", location.capabilities)

    def test_multi_location_provider_gated_by_capability(self):
        """The catalog's headline trap: a provider at four sites, offering an
        imaging service at only the subset that has imaging."""
        provider = self.catalog.providers["prov_000"]
        echo = self.appt_named("Echocardiogram")
        self.assertGreater(len(provider.location_ids), 1)

        combos = find_bookable(
            self.catalog, appointment_type_id=echo.id, provider_id=provider.id
        )
        offered_at = {c.location.id for c in combos}

        self.assertTrue(offered_at)
        self.assertLess(len(offered_at), len(provider.location_ids))
        self.assertTrue(offered_at <= provider.location_ids)
        for loc_id in offered_at:
            self.assertIn("imaging", self.catalog.locations[loc_id].capabilities)

    def test_referral_required(self):
        appt = next(
            a for a in self.catalog.appointment_types.values() if a.requires_referral
        )
        self.assertTrue(find_bookable(self.catalog, appointment_type_id=appt.id))
        self.assertEqual(
            [],
            find_bookable(
                self.catalog,
                appointment_type_id=appt.id,
                patient=Patient(has_referral=False),
            ),
        )

    def test_unknown_patient_facts_do_not_filter(self):
        """None means 'not asked yet' and must not behave like False, or the
        agent would silently rule out options before it has questioned the
        caller."""
        appt = next(
            a for a in self.catalog.appointment_types.values() if a.requires_referral
        )
        unknown = find_bookable(self.catalog, appointment_type_id=appt.id, patient=Patient())
        stated = find_bookable(
            self.catalog, appointment_type_id=appt.id, patient=Patient(has_referral=True)
        )
        self.assertEqual([c.key for c in unknown], [c.key for c in stated])

    def test_new_patient_blocked_from_restricted_type(self):
        appt = next(
            a for a in self.catalog.appointment_types.values() if not a.new_patients_allowed
        )
        self.assertTrue(find_bookable(self.catalog, appointment_type_id=appt.id))
        self.assertEqual(
            [],
            find_bookable(
                self.catalog, appointment_type_id=appt.id, patient=Patient(is_new=True)
            ),
        )

    def test_new_patient_blocked_from_closed_provider(self):
        provider = next(
            p for p in self.catalog.providers.values() if not p.accepting_new_patients
        )
        appt = next(
            self.catalog.appointment_types[a]
            for a in sorted(provider.appointment_type_ids)
            if self.catalog.appointment_types[a].new_patients_allowed
        )
        location = self.catalog.locations[sorted(provider.location_ids)[0]]

        self.assertTrue(is_bookable(appt, provider, location))
        self.assertIn(
            "provider_closed_to_new_patients",
            self.codes(
                why_not(
                    self.catalog, appt.id, provider.id, location.id, Patient(is_new=True)
                )
            ),
        )

    def test_policies_are_exactly_the_catalogs_rules(self):
        """Five filtering rules plus one conversational obligation — the six the
        catalog states, and nothing anyone invented. A preference that drifts in
        here silently removes options the clinic would happily book."""
        self.assertEqual(6, len(self.catalog.policies))
        self.assertEqual(5, len(POLICIES))
        for policy in POLICIES:
            self.assertNotIn("language", policy.description.lower())

    def test_language_is_not_a_policy(self):
        """Nothing in the catalog makes it illegal to book a doctor who does not
        speak the caller's language. It is a wish, so it belongs to ranking in
        the tool layer, not to the layer that decides what exists."""
        self.assertNotIn("language", Patient.__dataclass_fields__)

        combos = find_bookable(self.catalog, specialty="General")
        languages = {lang for c in combos for lang in c.provider.languages}
        self.assertGreater(len(languages), 1, "expected providers of several languages")

    def test_multi_location_provider_needs_disambiguation(self):
        many = self.catalog.providers["prov_000"]
        one = next(
            p for p in self.catalog.providers.values() if len(p.location_ids) == 1
        )
        self.assertTrue(needs_location_disambiguation(many))
        self.assertFalse(needs_location_disambiguation(one))


class TestSolver(CatalogTestCase):
    def test_specialty_filters_on_appointment_type_not_provider(self):
        """Internists and paediatricians offer General types. Filtering on the
        provider's own specialty would lose most of primary care."""
        combos = find_bookable(self.catalog, specialty="General")
        self.assertTrue(combos)

        for combo in combos:
            self.assertEqual("General", combo.appointment_type.specialty)

        provider_specialties = {p.specialty for p in distinct_providers(combos)}
        self.assertGreater(
            len(provider_specialties), 1, "expected General types across several specialties"
        )

    def test_every_returned_combo_satisfies_every_policy(self):
        for combo in find_bookable(self.catalog, specialty="Radiology"):
            self.assertTrue(
                is_bookable(combo.appointment_type, combo.provider, combo.location)
            )

    def test_constraints_narrow_monotonically(self):
        wide = find_bookable(self.catalog, specialty="Dermatology")
        narrow = find_bookable(
            self.catalog,
            specialty="Dermatology",
            location_id=wide[0].location.id,
        )
        self.assertLessEqual(len(narrow), len(wide))
        self.assertTrue({c.key for c in narrow} <= {c.key for c in wide})

    def test_results_are_deterministic(self):
        first = find_bookable(self.catalog, specialty="Cardiology")
        second = find_bookable(self.catalog, specialty="Cardiology")
        self.assertEqual([c.key for c in first], [c.key for c in second])

    def test_why_not_reports_unknown_entities(self):
        self.assertIn(
            "unknown_entity",
            self.codes(why_not(self.catalog, "appt_nope", "prov_nope", "loc_nope")),
        )

    def test_unknown_specialty_yields_no_combos(self):
        self.assertEqual([], find_bookable(self.catalog, specialty="Traumatology"))

    def test_relaxing_a_search_never_produces_an_illegal_result(self):
        """The property that makes it safe to widen a search when nothing matches.

        Dropping a filter enlarges the candidate set; it does not switch any
        policy off, because every combination returned is checked regardless of
        which arguments were passed.
        """
        appt = self.appt_named("Echocardiogram")
        patient = Patient(is_new=False, has_referral=True)
        narrow = find_bookable(
            self.catalog,
            appointment_type_id=appt.id,
            provider_id="prov_000",
            location_id="loc_004",
            patient=patient,
        )
        widened = find_bookable(
            self.catalog, appointment_type_id=appt.id, patient=patient
        )

        self.assertTrue({c.key for c in narrow} <= {c.key for c in widened})
        for combo in widened:
            self.assertTrue(
                is_bookable(combo.appointment_type, combo.provider, combo.location, patient)
            )

    def test_patient_facts_are_not_relaxable_filters(self):
        """is_new and has_referral feed policies, so they can never be dropped to
        widen a search the way a provider or site preference can."""
        appt = next(
            a for a in self.catalog.appointment_types.values() if a.requires_referral
        )
        without = find_bookable(self.catalog, appointment_type_id=appt.id)
        refused = find_bookable(
            self.catalog, appointment_type_id=appt.id, patient=Patient(has_referral=False)
        )
        self.assertTrue(without)
        self.assertEqual([], refused)


class TestMessiness(CatalogTestCase):
    def test_duplicate_provider_names_all_returned(self):
        """A name is not a unique key: the agent must disambiguate, so the
        store hands back every candidate rather than guessing."""
        matches = self.catalog.resolve_provider_name("Dr. Maria Garcia")
        self.assertGreater(len(matches), 1)
        self.assertEqual({"Dr. Maria Garcia"}, {p.name for p in matches})
        self.assertGreater(len({p.specialty for p in matches}), 1)

    def test_partial_provider_name_returns_candidates(self):
        matches = self.catalog.resolve_provider_name("Chen")
        self.assertGreater(len(matches), 1)
        for provider in matches:
            self.assertIn("chen", provider.name.lower())

    def test_hallucinated_specialty_degrades_into_suggestions(self):
        """The recovery path: never an empty result the agent can't act on."""
        exact, suggestions = self.catalog.resolve_specialty("Traumatology")
        self.assertIsNone(exact)
        self.assertTrue(suggestions)

    def test_specialty_resolution_is_case_insensitive(self):
        exact, _ = self.catalog.resolve_specialty("orthopedics")
        self.assertEqual("Orthopedics", exact)

    def test_near_duplicate_types_both_survive_narrowing(self):
        """Retrieval narrows; it must not silently pick between two types that
        only a business rule separates. Both reach the conversation."""
        names = {a.name for a in self.catalog.appointment_types.values()}
        self.assertIn("Skin Cancer Screening", names)
        self.assertIn("Full Body Skin Exam", names)

        derm = find_bookable(self.catalog, specialty="Dermatology")
        offered = {a.name for a in (c.appointment_type for c in derm)}
        self.assertIn("Skin Cancer Screening", offered)
        self.assertIn("Full Body Skin Exam", offered)


class TestStoreIntegrity(CatalogTestCase):
    def test_no_dangling_references(self):
        for provider in self.catalog.providers.values():
            for type_id in provider.appointment_type_ids:
                self.assertIn(type_id, self.catalog.appointment_types)
            for loc_id in provider.location_ids:
                self.assertIn(loc_id, self.catalog.locations)

    def test_specialties_list_is_small_enough_to_inject(self):
        """The escape hatch documented in solution.md is only valid while this
        list stays cheap. If it stops being true, the retrieval path is the
        only correct option."""
        specialties = self.catalog.specialties
        self.assertTrue(specialties)
        self.assertLess(len(", ".join(specialties)) // 4, 150)

    def test_bookable_specialties_excludes_unstaffed_ones(self):
        """The catalog advertises services nobody is staffed for. Routing a
        caller there is a dead end they only discover at the end of the call,
        so the booking funnel is built from this narrower list."""
        bookable = bookable_specialties(self.catalog)
        self.assertTrue(bookable)
        self.assertLess(len(bookable), len(self.catalog.specialties))

        for specialty in bookable:
            self.assertTrue(
                find_bookable(self.catalog, specialty=specialty),
                f"{specialty} has no bookable combination at all",
            )

    def test_unstaffed_types_are_visible_but_not_bookable(self):
        unstaffed = unstaffed_appointment_types(self.catalog)
        self.assertTrue(unstaffed, "expected some advertised-but-unstaffed types")

        for appt in unstaffed:
            self.assertIn(appt.id, self.catalog.appointment_types)
            self.assertEqual([], find_bookable(self.catalog, appointment_type_id=appt.id))

    def test_load_is_equivalent_from_dict(self):
        self.assertIsInstance(self.catalog, StoreCatalog)


if __name__ == "__main__":
    unittest.main()
