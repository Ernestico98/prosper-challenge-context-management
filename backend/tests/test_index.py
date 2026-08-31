#
# Retrieval tests.
#
# Deliberately offline: the lexical channel needs no network, and the vector
# channel is exercised with a stub embedder. The suite therefore makes no API
# calls and asserts the property that matters -- that the index still works
# when embeddings are unavailable.
#

import unittest

from catalog import Catalog, bookable_specialties
from catalog.index import (
    InMemorySpecialtyIndex,
    ScoredSpecialty,
    load_aliases,
    normalise,
    tokenise,
)


class StubEmbedder:
    """A toy 'embedding': one dimension per keyword. Enough to prove fusion
    happens without touching the network."""

    KEYWORDS = ["knee", "heart", "skin", "child", "imaging", "mental"]

    def embed(self, texts):
        return [
            [1.0 if word in normalise(text) else 0.0 for word in self.KEYWORDS]
            for text in texts
        ]


class IndexTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog.load()
        cls.aliases = load_aliases()
        cls.index = InMemorySpecialtyIndex(cls.catalog, aliases=cls.aliases)

    def top(self, query, limit=3):
        return [s.name for s in self.index.search(query, limit=limit)]


class TestLexicalChannel(IndexTestCase):
    def test_symptom_in_spanish_finds_the_specialty(self):
        self.assertIn("Orthopedics", self.top("me duele la rodilla desde hace semanas"))

    def test_symptom_in_english_finds_the_specialty(self):
        self.assertIn("Orthopedics", self.top("my knee has been hurting for weeks"))

    def test_rare_exact_token_is_caught(self):
        """The case embeddings are worst at: acronyms and procedure names."""
        for query in ("I need an MRI", "necesito una mamografia", "a DEXA scan"):
            self.assertIn("Radiology", self.top(query), query)

    def test_multiword_alias_outranks_single_tokens(self):
        ranked = self.top("dolor de pecho al subir escaleras", limit=3)
        self.assertEqual("Cardiology", ranked[0])

    def test_ambiguous_complaint_returns_several_candidates(self):
        """Retrieval narrows; it does not decide. A knee MRI is legitimately
        both Radiology and Orthopedics, and the conversation resolves it."""
        results = self.top("I need an MRI of my knee", limit=5)
        self.assertIn("Radiology", results)
        self.assertIn("Orthopedics", results)

    def test_accents_do_not_matter(self):
        with_accent = self.top("necesito una resonancia magnética")
        without = self.top("necesito una resonancia magnetica")
        self.assertEqual(with_accent, without)
        self.assertIn("Radiology", with_accent)

    def test_unmatched_query_returns_empty_rather_than_noise(self):
        """An honest miss the agent can recover from beats a confident wrong
        answer it cannot."""
        self.assertEqual([], self.index.search("zzzz qqqq wwww", limit=5))


class TestResultShape(IndexTestCase):
    def test_results_are_scored_specialties_with_counts(self):
        results = self.index.search("skin rash", limit=3)
        self.assertTrue(results)
        for item in results:
            self.assertIsInstance(item, ScoredSpecialty)
            self.assertGreater(item.type_count, 0)
            self.assertTrue(item.sample_types)

    def test_limit_is_honoured(self):
        self.assertLessEqual(len(self.index.search("consultation", limit=2)), 2)

    def test_results_are_ordered_by_score(self):
        results = self.index.search("dolor de cabeza y mareo", limit=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ranking_is_deterministic(self):
        first = self.top("tengo ansiedad", limit=5)
        second = self.top("tengo ansiedad", limit=5)
        self.assertEqual(first, second)


class TestUnstaffedSpecialties(IndexTestCase):
    def test_unstaffed_specialties_are_found_not_hidden(self):
        """Hiding them looked safer and was worse: the query then matched the
        next-closest specialty, and the agent booked a general follow-up for a
        caller who asked for an eye exam."""
        self.assertIn("Ophthalmology", self.top("revision de la vista", limit=3))
        self.assertIn("Physical Therapy", self.top("necesito fisioterapia", limit=3))

    def test_unstaffed_specialties_report_nothing_bookable(self):
        """Found, but unmistakably unavailable — that is what lets the agent
        say so instead of substituting something else."""
        result = self.index.search("revision de la vista", limit=1)[0]
        self.assertEqual(0, result.type_count)
        self.assertTrue(result.as_dict()["not_currently_offered"])

    def test_staffed_specialties_are_not_flagged(self):
        result = self.index.search("chest pain", limit=1)[0]
        self.assertGreater(result.type_count, 0)
        self.assertNotIn("not_currently_offered", result.as_dict())

    def test_index_covers_every_specialty_in_the_catalog(self):
        self.assertEqual(set(self.catalog.specialties), set(self.index.documents))
        self.assertLess(
            len(bookable_specialties(self.catalog)), len(self.index.documents)
        )


class TestVectorChannel(IndexTestCase):
    def test_index_works_with_no_vectors_at_all(self):
        """The property that keeps the system runnable offline and keyless."""
        index = InMemorySpecialtyIndex(self.catalog, aliases=self.aliases, vectors={})
        self.assertIn("Cardiology", [s.name for s in index.search("chest pain")])

    def test_vector_channel_contributes_to_fusion(self):
        embedder = StubEmbedder()
        vectors = {
            name: embedder.embed([text])[0]
            for name, text in self.index.document_texts().items()
        }
        hybrid = InMemorySpecialtyIndex(
            self.catalog, aliases=self.aliases, vectors=vectors, embedder=embedder
        )
        results = hybrid.search("knee", limit=5)
        self.assertTrue(results)
        self.assertIn("Orthopedics", [r.name for r in results])

    def test_embedder_failure_degrades_to_lexical(self):
        class Broken:
            def embed(self, texts):
                raise RuntimeError("no network")

        # Full coverage, so this exercises the embedder failing at query time
        # rather than the stale-cache check disabling the channel first.
        vectors = {name: [1.0] for name in self.index.documents}
        index = InMemorySpecialtyIndex(
            self.catalog, aliases=self.aliases, vectors=vectors, embedder=Broken()
        )
        self.assertIn("Cardiology", [s.name for s in index.search("chest pain")])

    def test_partial_vector_coverage_disables_the_channel(self):
        """Rank fusion rewards appearing in both channels, so a specialty missing
        from a stale cache sinks below worse matches that happen to be in it.
        Half an index is worse than none."""
        index = InMemorySpecialtyIndex(
            self.catalog, aliases=self.aliases, vectors={"Cardiology": [1.0]}
        )
        self.assertEqual({}, index.vectors)


class TestNormalisation(unittest.TestCase):
    def test_accents_and_case_are_stripped(self):
        self.assertEqual(normalise("Resonancia Magnética"), "resonancia magnetica")

    def test_stopwords_are_dropped(self):
        self.assertEqual(tokenise("me duele la rodilla"), ["duele", "rodilla"])


if __name__ == "__main__":
    unittest.main()
