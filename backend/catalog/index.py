#
# Retrieval — entering the catalog hierarchy without paying for it.
#
# The top of the hierarchy is RETRIEVED, not injected. The tool takes the
# caller's own words ("me duele la rodilla desde hace semanas") rather than a
# specialty name, which costs zero prompt tokens and removes the hallucinated-
# specialty failure at this step: the model relays text instead of picking from
# a vocabulary it was never shown.
#
# Three deliberate choices, all defended in solution.md:
#
#   1. Enriched documents, not labels. Embedding "knee pain" against the bare
#      string "Orthopedics" is a weak match -- the link is clinical inference,
#      not similarity. Each specialty's document is its name + its appointment
#      type names + curated bilingual aliases, which gives the complaint
#      something real to match against.
#   2. Hybrid, not pure vector. Embeddings are poor at rare exact tokens (MRI,
#      DEXA, mamografia); the lexical channel nails them. Fused with reciprocal
#      rank fusion, which needs no score calibration between channels.
#   3. Retrieval narrows, it never decides. The shortlist goes back to the LLM,
#      which reranks with clinical knowledge it already has. No cross-encoder:
#      the model is in the loop and paid for either way.
#
# The lexical channel works with no network at all, so the index degrades to
# offline operation when embeddings are unavailable.
#

import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union

from loguru import logger

from .query import bookable_specialties, find_bookable
from .store import Catalog

ALIASES_PATH = Path(__file__).parent.parent / "data" / "specialty_aliases.json"
VECTORS_PATH = Path(__file__).parent.parent / "data" / "specialty_vectors.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 256
RRF_K = 60  # reciprocal-rank-fusion damping; 60 is the usual default

STOPWORDS = frozenset(
    """
    a al algo and any as at de del el en es esta este for from he her his i in is it la
    las lo los me mi my necesito need of on or para pero por que quiero se su tengo the
    that to un una uno with y yo
    """.split()
)


@dataclass(frozen=True)
class ScoredSpecialty:
    name: str
    score: float
    type_count: int
    sample_types: tuple = ()
    matched: tuple = ()  # which query terms hit, for debugging and eval output

    def as_dict(self) -> dict:
        summary = {
            "specialty": self.name,
            "appointment_types_available": self.type_count,
            "examples": list(self.sample_types),
        }
        if self.type_count == 0:
            # The clinic lists the service but has nobody to staff it. Saying so
            # is the point: hiding it makes the agent route the caller to
            # whatever matched next, which is how someone asking for an eye exam
            # ends up booked for a general follow-up.
            summary["not_currently_offered"] = True
        return summary


class SpecialtyIndex(Protocol):
    """The seam that keeps the tool handler independent of the search backend.

    Today: an in-memory index over a JSON catalog. In production: the same
    method backed by pgvector or Qdrant, with the top-N fetched from the
    database and reranked in code. The tool layer never changes.
    """

    def search(self, query: str, *, limit: int = 10) -> list: ...


class Embedder(Protocol):
    def embed(self, texts: list) -> list: ...


class OpenAIEmbedder:
    """Batched embeddings. Used by the build script, and once per search when a
    vector cache is present."""

    def __init__(self, model: str = EMBEDDING_MODEL, dimensions: int = EMBEDDING_DIMENSIONS):
        self.model = model
        self.dimensions = dimensions
        self._client = None

    def embed(self, texts: list) -> list:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        response = self._client.embeddings.create(
            model=self.model, dimensions=self.dimensions, input=texts
        )
        return [item.embedding for item in response.data]


# ---- text normalisation ----------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation.

    Accent stripping is not cosmetic: callers say "resonancia magnética" and the
    alias file says "resonancia magnetica".
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch if ch.isalnum() else " " for ch in stripped)


def stem(token: str) -> str:
    """Crude plural stripping, applied to both sides of the comparison.

    Callers say "huesos" while the alias file says "hueso". A real stemmer would
    need a per-language model for a gain this small; folding plurals is the 90%
    of it and stays predictable, which matters for a vocabulary a clinic edits
    by hand.
    """
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenise(text: str) -> list:
    return [
        stem(t) for t in normalise(text).split() if t and t not in STOPWORDS
    ]


# ---- the index -------------------------------------------------------------


@dataclass
class SpecialtyDocument:
    name: str
    type_count: int
    sample_types: tuple
    aliases: tuple
    text: str
    tokens: frozenset = field(default_factory=frozenset)


class InMemorySpecialtyIndex:
    """Hybrid lexical + vector search over enriched specialty documents."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        aliases: Optional[dict] = None,
        vectors: Optional[dict] = None,
        embedder: Optional[Embedder] = None,
        include_unstaffed: bool = True,
    ):
        self.catalog = catalog
        self.aliases = aliases if aliases is not None else load_aliases()
        self.vectors = vectors or {}
        self.embedder = embedder

        names = (
            catalog.specialties if include_unstaffed else bookable_specialties(catalog)
        )
        self.documents = {name: self._build_document(name) for name in names}
        self._idf = self._compute_idf()

        # Partial vector coverage is worse than none. Reciprocal rank fusion
        # rewards appearing in both channels, so a specialty missing from a
        # stale cache is pushed below worse matches that happen to be in it --
        # silently, and only in the configuration that ships. Drop the whole
        # channel rather than rank against a half-built index.
        missing = [name for name in self.documents if name not in self.vectors]
        if self.vectors and missing:
            logger.warning(
                f"Specialty vector cache is stale: {len(missing)} of "
                f"{len(self.documents)} documents have no vector "
                f"({', '.join(missing[:4])}...). Falling back to lexical search. "
                f"Run `make index` to rebuild it."
            )
            self.vectors = {}

    # ---- construction ------------------------------------------------------
    def _build_document(self, specialty: str) -> SpecialtyDocument:
        type_ids = self.catalog.types_by_specialty.get(specialty, [])
        type_names = [self.catalog.appointment_types[t].name for t in type_ids]
        aliases = tuple(self.aliases.get(specialty, []))
        text = " ; ".join([specialty, *type_names, *aliases])
        return SpecialtyDocument(
            name=specialty,
            type_count=len(type_ids),
            sample_types=tuple(type_names[:4]),
            aliases=aliases,
            text=text,
            tokens=frozenset(tokenise(text)),
        )

    def _compute_idf(self) -> dict:
        total = len(self.documents) or 1
        frequency = {}
        for doc in self.documents.values():
            for token in doc.tokens:
                frequency[token] = frequency.get(token, 0) + 1
        # A token appearing in one specialty ("mamografia") is far more telling
        # than one appearing in fifteen ("consultation").
        return {t: math.log(1 + total / count) for t, count in frequency.items()}

    def document_texts(self) -> dict:
        return {name: doc.text for name, doc in self.documents.items()}

    # ---- channels ----------------------------------------------------------
    def _lexical_ranking(self, query: str) -> list:
        query_tokens = tokenise(query)
        normalised_query = normalise(query)
        scores = {}
        hits = {}

        for name, doc in self.documents.items():
            score = 0.0
            matched = []
            for token in query_tokens:
                if token in doc.tokens:
                    score += self._idf.get(token, 0.0)
                    matched.append(token)
            # Multi-word aliases ("dolor de espalda", "chest pain") are far more
            # specific than their individual tokens, so a phrase hit is worth
            # more than the sum of its parts.
            for alias in doc.aliases:
                normalised_alias = normalise(alias).strip()
                if " " in normalised_alias and normalised_alias in normalised_query:
                    score += 2.0 * len(normalised_alias.split())
                    matched.append(alias)
            if score > 0:
                scores[name] = score
                hits[name] = matched

        ranked = sorted(scores, key=lambda n: (-scores[n], n))
        return [(name, hits[name]) for name in ranked]

    def _vector_ranking(self, query: str) -> list:
        if not self.vectors or self.embedder is None:
            return []
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception:  # noqa: BLE001 - retrieval must degrade, not crash a call
            return []

        scored = []
        for name in self.documents:
            vector = self.vectors.get(name)
            if vector:
                scored.append((_cosine(query_vector, vector), name))
        scored.sort(reverse=True)
        return [name for _, name in scored]

    # ---- public API --------------------------------------------------------
    def search(self, query: str, *, limit: int = 10) -> list:
        lexical = self._lexical_ranking(query)
        vector = self._vector_ranking(query)

        # Reciprocal Rank Fusion: merges the two rankings, drops scores and
        # uses ranks
        fused = {}
        matches = {}
        for rank, (name, matched) in enumerate(lexical):
            fused[name] = fused.get(name, 0.0) + 1.0 / (RRF_K + rank + 1)
            matches[name] = matched
        for rank, name in enumerate(vector):
            fused[name] = fused.get(name, 0.0) + 1.0 / (RRF_K + rank + 1)
            matches.setdefault(name, [])

        ordered = sorted(fused, key=lambda n: (-fused[n], n))[:limit]
        return [
            ScoredSpecialty(
                name=name,
                score=round(fused[name], 6),
                type_count=self._bookable_type_count(name),
                sample_types=self.documents[name].sample_types,
                matched=tuple(matches.get(name, [])),
            )
            for name in ordered
        ]

    def _bookable_type_count(self, specialty: str) -> int:
        combos = find_bookable(self.catalog, specialty=specialty)
        return len({c.appointment_type.id for c in combos})


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


# ---- loading ---------------------------------------------------------------


def load_aliases(path: Union[str, Path] = ALIASES_PATH) -> dict:
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_vectors(path: Union[str, Path] = VECTORS_PATH) -> dict:
    """Precomputed specialty embeddings, or {} when they haven't been built.

    Absent vectors are not an error: the index runs lexical-only, which needs no
    network and no API key.
    """
    file = Path(path)
    if not file.exists():
        return {}
    payload = json.loads(file.read_text())
    return payload.get("vectors", {})


def build_index(
    catalog: Catalog, *, with_vectors: bool = True, include_unstaffed: bool = True
) -> InMemorySpecialtyIndex:
    """The index the tool layer uses. Vectors are optional by design.

    Unstaffed specialties are INDEXED, not hidden. Leaving them out looks safer
    -- why route a caller somewhere nothing is bookable? -- but it is worse in
    practice: the query then matches whatever is next-closest, and the agent
    confidently books the wrong thing. Retrieval should find them and the tool
    should say they are unavailable.
    """
    vectors = load_vectors() if with_vectors else {}
    return InMemorySpecialtyIndex(
        catalog,
        aliases=load_aliases(),
        vectors=vectors,
        embedder=OpenAIEmbedder() if vectors else None,
        include_unstaffed=include_unstaffed,
    )
