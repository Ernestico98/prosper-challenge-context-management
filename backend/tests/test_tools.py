#
# Tool-layer tests.
#
# The tools are adapters, so what is tested here is adapter behaviour: context
# budgeting, speech-friendly shaping, and above all RECOVERY -- a tool must
# never hand the agent an empty result it cannot act on. No LLM is involved;
# a stub stands in for the flow manager's state.
#

import unittest
from datetime import date

from tools import get
from tools.context import SchedulingContext

TODAY = date(2026, 3, 2)


class FakeFlowManager:
    """Stands in for pipecat's FlowManager: the tools only touch `.state`."""

    def __init__(self, **state):
        self.state = dict(state)


class ToolTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # with_vectors=False keeps the whole suite offline and keyless.
        cls.ctx = SchedulingContext.default(with_vectors=False, today=TODAY)

    def setUp(self):
        self.ctx.bookings.clear()
        self.fm = FakeFlowManager(full_name="Ana Ruiz")

    def call(self, tool, **args):
        return get(tool).handler(args, self.ctx, self.fm)


class TestFindSpecialties(ToolTestCase):
    def test_complaint_in_the_callers_own_words(self):
        result = self.call("find_specialties", complaint="me duele la rodilla")
        names = [m["specialty"] for m in result["matches"]]
        self.assertIn("Orthopedics", names)

    def test_result_carries_counts_and_examples_for_reranking(self):
        result = self.call("find_specialties", complaint="skin rash")
        for match in result["matches"]:
            self.assertGreater(match["appointment_types_available"], 0)
            self.assertTrue(match["examples"])

    def test_result_is_capped_to_the_context_budget(self):
        result = self.call("find_specialties", complaint="consultation follow up visit")
        self.assertLessEqual(len(result["matches"]), self.ctx.max_options)

    def test_no_match_returns_a_hint_not_silence(self):
        result = self.call("find_specialties", complaint="zzzz qqqq")
        self.assertEqual([], result["matches"])
        self.assertIn("hint", result)


class TestListAppointmentTypes(ToolTestCase):
    def test_lists_types_for_a_real_specialty(self):
        result = self.call("list_appointment_types", specialty="Orthopedics")
        self.assertEqual("Orthopedics", result["specialty"])
        self.assertTrue(result["appointment_types"])

    def test_hallucinated_specialty_recovers_via_the_index(self):
        """Edit distance puts 'Traumatology' next to 'Dermatology'; the lay
        vocabulary puts it next to Orthopedics, which is the useful answer."""
        result = self.call("list_appointment_types", specialty="Traumatology")
        self.assertEqual("unknown_specialty", result["error"])
        self.assertIn("Orthopedics", result["did_you_mean"])

    def test_requirements_are_surfaced_on_each_type(self):
        result = self.call("list_appointment_types", specialty="Radiology")
        gated = [t for t in result["appointment_types"] if t.get("only_at_sites_with")]
        self.assertTrue(gated, "imaging types should declare their site requirement")

    def test_new_patient_state_narrows_the_list(self):
        everyone = self.call("list_appointment_types", specialty="General")
        self.fm.state["is_new_patient"] = True
        new_patient = self.call("list_appointment_types", specialty="General")
        self.assertLess(
            len(new_patient["appointment_types"]), len(everyone["appointment_types"])
        )


class TestFindAppointments(ToolTestCase):
    KNEE_MRI = "appt_065"

    def test_returns_fully_specified_offers(self):
        result = self.call("find_appointments", appointment_type=self.KNEE_MRI)
        self.assertTrue(result["options"])
        for option in result["options"]:
            self.assertTrue(option["option_id"])
            self.assertTrue(option["provider"])
            self.assertTrue(option["site"])
            self.assertTrue(option["when"])

    def test_accepts_a_name_as_well_as_an_id(self):
        by_name = self.call("find_appointments", appointment_type="MRI - Knee")
        self.assertIn("options", by_name)

    def test_capability_gating_holds_in_the_offers(self):
        result = self.call("find_appointments", appointment_type=self.KNEE_MRI)
        for option in result["options"]:
            location = next(
                l for l in self.ctx.catalog.locations.values() if l.name == option["site"]
            )
            self.assertIn("imaging", location.capabilities)

    def test_refining_is_a_second_call_not_a_transition(self):
        """The graph must not have to move when the caller says 'not those
        times' — that is what keeps a linear flow from breaking."""
        first = self.call("find_appointments", appointment_type=self.KNEE_MRI)
        refined = self.call(
            "find_appointments",
            appointment_type=self.KNEE_MRI,
            weekdays=["Tuesday"],
            time_of_day="afternoon",
        )
        self.assertTrue(refined["options"])
        for option in refined["options"]:
            self.assertIn("Tuesday", option["when"])
        self.assertNotEqual(
            [o["starts_at"] for o in first["options"]],
            [o["starts_at"] for o in refined["options"]],
        )

    def test_soonest_ranking_is_chronological(self):
        result = self.call(
            "find_appointments", appointment_type=self.KNEE_MRI, rank_by="soonest"
        )
        times = [o["starts_at"] for o in result["options"]]
        self.assertEqual(times, sorted(times))

    def test_options_are_capped_and_deduped(self):
        result = self.call("find_appointments", appointment_type="appt_000")
        options = result["options"]
        self.assertLessEqual(len(options), self.ctx.max_options)
        self.assertEqual(len(options), len({o["provider"] for o in options}))
        self.assertEqual(len(options), len({o["starts_at"] for o in options}))

    def test_ambiguous_provider_name_asks_instead_of_guessing(self):
        result = self.call(
            "find_appointments",
            appointment_type="appt_000",
            provider_name="Dr. Maria Garcia",
        )
        self.assertEqual("ambiguous_provider", result["error"])
        self.assertGreater(len(result["candidates"]), 1)
        self.assertGreater(len({c["specialty"] for c in result["candidates"]}), 1)

    def test_impossible_request_explains_itself(self):
        """'MRI at the Sunset clinic' must come back with the reason, not an
        empty list."""
        result = self.call(
            "find_appointments", appointment_type=self.KNEE_MRI, location_name="Sunset"
        )
        self.assertEqual("no_bookable_combination", result["error"])
        self.assertTrue(any("imaging" in r for r in result["reasons"]))
        self.assertTrue(result["available_instead"]["sites"])

    def test_new_patient_restriction_is_explained(self):
        self.fm.state["is_new_patient"] = True
        result = self.call("find_appointments", appointment_type=self.KNEE_MRI)
        self.assertEqual("no_bookable_combination", result["error"])
        self.assertTrue(any("new patients" in r for r in result["reasons"]))

    def test_unresolvable_preference_widens_instead_of_blocking(self):
        """'my usual family doctor' is not a name and never will be. Erroring
        left the agent asking for a name the caller could not give, three times
        in a row. A preference nobody can resolve is dropped and declared."""
        result = self.call(
            "find_appointments",
            appointment_type="appt_002",
            provider_name="mi medico de familia",
        )
        self.assertTrue(result["options"])
        self.assertTrue(result["ignored_preferences"])

    def test_ambiguity_still_stops_and_asks(self):
        """Unknown and ambiguous are different: two doctors sharing a name is a
        question only the caller can settle."""
        result = self.call(
            "find_appointments", appointment_type="appt_015", provider_name="Maria Garcia"
        )
        self.assertEqual("ambiguous_provider", result["error"])
        self.assertNotIn("options", result)

    def test_candidates_can_be_chosen_by_id(self):
        """Same trick as option_id: pick from a list we produced rather than
        re-typing a string that may not resolve."""
        ambiguous = self.call(
            "find_appointments", appointment_type="appt_015", provider_name="Maria Garcia"
        )
        chosen = ambiguous["candidates"][0]["provider_id"]
        result = self.call(
            "find_appointments", appointment_type="appt_015", provider_id=chosen
        )
        self.assertTrue(result["options"])
        expected = self.ctx.catalog.providers[chosen].display_name
        self.assertEqual({expected}, {o["provider"] for o in result["options"]})

    def test_the_displayed_name_is_a_usable_answer(self):
        """The tool advertises candidates as 'Dr. Maria Garcia, MD', so that
        exact string has to resolve — otherwise the model cannot disambiguate
        using the words it was just given."""
        result = self.call(
            "find_appointments",
            appointment_type="appt_015",
            provider_name="Dr. Maria Garcia, MD",
        )
        self.assertNotIn("error", result)
        self.assertTrue(result["options"])

    def test_unknown_type_is_reported(self):
        result = self.call("find_appointments", appointment_type="appt_does_not_exist")
        self.assertEqual("unknown_appointment_type", result["error"])


class TestBooking(ToolTestCase):
    def offer(self):
        return self.call("find_appointments", appointment_type="appt_000")

    def test_books_an_offered_option(self):
        self.offer()
        result = self.call("book_appointment", option_id="opt_1")
        self.assertEqual("booked", result["status"])
        self.assertTrue(result["reference"])
        self.assertEqual("Ana Ruiz", result["patient_name"])
        self.assertEqual(1, len(self.ctx.bookings))

    def test_booking_matches_the_offer_exactly(self):
        offered = self.offer()["options"][0]
        booked = self.call("book_appointment", option_id=offered["option_id"])
        self.assertEqual(offered["provider"], booked["provider"])
        self.assertEqual(offered["site"], booked["site"])
        self.assertEqual(offered["starts_at"], booked["starts_at"])

    def test_invented_option_id_cannot_book(self):
        """The model chooses from a list we just produced, so it cannot compose
        an illegal combination — and an id it invents is rejected outright."""
        self.offer()
        result = self.call("book_appointment", option_id="opt_999")
        self.assertEqual("unknown_option", result["error"])
        self.assertEqual([], self.ctx.bookings)

    def test_booking_without_any_offer_is_rejected(self):
        result = self.call("book_appointment", option_id="opt_1")
        self.assertEqual("unknown_option", result["error"])


class TestQuestionAnswering(ToolTestCase):
    def test_find_providers_by_name_returns_every_namesake(self):
        result = self.call("find_providers", name="Dr. Maria Garcia")
        self.assertGreater(len(result["providers"]), 1)

    def test_find_providers_by_language(self):
        result = self.call("find_providers", specialty="Cardiology", language="Spanish")
        for provider in result["providers"]:
            self.assertIn("Spanish", provider["languages"])

    def test_find_providers_flags_when_a_site_must_be_asked_for(self):
        result = self.call("find_providers", name="Dr. David Chen")
        self.assertTrue(result["providers"][0]["needs_site_disambiguation"])

    def test_unknown_specialty_recovers(self):
        result = self.call("find_providers", specialty="Traumatology")
        self.assertEqual("unknown_specialty", result["error"])
        self.assertTrue(result["did_you_mean"])

    def test_get_location_info_lists_all_sites(self):
        result = self.call("get_location_info")
        self.assertEqual(len(self.ctx.catalog.locations), len(result["locations"]))
        for location in result["locations"]:
            self.assertTrue(location["address"])
            self.assertTrue(location["hours"])
            self.assertTrue(location["on_site_services"])

    def test_get_location_info_by_partial_name(self):
        result = self.call("get_location_info", location_name="Mission Bay")
        self.assertEqual(1, len(result["locations"]))

    def test_describe_appointment_type_states_its_requirements(self):
        result = self.call("describe_appointment_type", appointment_type="MRI - Knee")
        self.assertEqual("imaging", result["only_at_sites_with"])
        self.assertTrue(result["requires_referral"])
        self.assertGreater(result["site_count"], 0)


class TestResultsAreSpeakable(ToolTestCase):
    def test_offers_never_put_raw_ids_in_spoken_fields(self):
        result = self.call("find_appointments", appointment_type="appt_000")
        for option in result["options"]:
            self.assertNotIn("prov_", option["provider"])
            self.assertNotIn("loc_", option["site"])
            self.assertNotIn("appt_", option["when"])


if __name__ == "__main__":
    unittest.main()
