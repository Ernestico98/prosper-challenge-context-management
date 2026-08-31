#
# Precompute specialty embeddings.
#
# One batched call to text-embedding-3-small over ~18 short documents. Run it
# after editing the catalog or the alias file:
#
#     make index
#
# The result is cached to backend/data/specialty_vectors.json and committed, so
# nothing embeds at runtime except the caller's query. Skipping this step is
# supported: the index falls back to its lexical channel, which needs no network.
#

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv                                     # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from catalog import Catalog                                        # noqa: E402
from catalog.index import (                                        # noqa: E402
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    VECTORS_PATH,
    InMemorySpecialtyIndex,
    OpenAIEmbedder,
    load_aliases,
)


def main() -> int:
    catalog = Catalog.load()
    index = InMemorySpecialtyIndex(catalog, aliases=load_aliases())
    documents = index.document_texts()

    names = sorted(documents)
    texts = [documents[name] for name in names]
    print(f"Embedding {len(names)} specialty documents with {EMBEDDING_MODEL} "
          f"({EMBEDDING_DIMENSIONS} dims)...")

    vectors = OpenAIEmbedder().embed(texts)

    payload = {
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        # Lets a stale cache be detected after the catalog or aliases change.
        "documents_digest": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "vectors": dict(zip(names, vectors)),
    }
    Path(VECTORS_PATH).write_text(json.dumps(payload))
    print(f"Wrote {VECTORS_PATH} ({len(names)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
