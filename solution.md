# Solution — voice scheduling agent with a context-managed catalog

**The catalog never enters the prompt.** It sits behind deterministic tools, the booking
policies are enforced in code rather than in the prompt, and the conversation graph exposes
only the tools each step needs — so the model chooses between three or four options instead
of eighty-two.

Measured on the provided catalog (8 locations, 50 providers, 82 appointment types):

| | Result |
| --- | --- |
| Input tokens per booking call | **40× fewer** than putting the catalog in the prompt |
| Largest catalog excerpt the model ever reads | **353 tokens**, against 82 appointment types |
| Illegal bookings across 7 eval runs (~100 scenario executions) | **0** |
| Deterministic tests | **154** Python + **17** JavaScript, no API keys, no network |

```bash
make install && make ui     # Python venv, then build the builder UI
make run                    # http://localhost:7860/builder
make test                   # 154 unit tests, offline, ~0.1s
make test-ui                # 17 graph-operation tests
make benchmark              # reproduces the token figures below
make evals ARGS=--dry-run   # lists the 16 accuracy scenarios; drop ARGS to run them
```

---

## 1. What was built

| Requirement from the brief | Where it lives |
| --- | --- |
| A UI to create an agent and place a test call | `frontend/` — create, wire and delete agents; edit nodes, edges, prompts and slots; choose which agent the call runs |
| Creating appointments efficiently | `find_appointments` → `book_appointment` (`backend/tools/scheduling.py`) |
| Offering available appointment slots | Offers carry provider, site and time together, ranked by what the caller asked for |
| Matching request → type, location, provider | Retrieval (`catalog/index.py`) narrows; the constraint solver (`catalog/query.py`) decides what is legal |
| Disambiguating similar options | Duplicate provider names return every candidate with an id to pick from; near-duplicate appointment types are separated by a question, never guessed |
| Honouring preferences | Provider, location and soonest-available are carried as slots that survive a context reset |
| Answering questions about locations, doctors, types | A read-only Q&A node with `find_providers`, `get_location_info`, `describe_appointment_type` |

Beyond the brief: a live context-cost panel that reads real token usage off the pipeline, and
an offline eval harness that drives the compiled graph over text.

---

## 2. The problem, measured

Flattening the catalog into a system prompt costs **5,449 tokens**, re-sent on every turn:
**65,388 tokens for a twelve-turn booking**.

Cost is the lesser of the two problems, and prompt caching recovers much of it in money terms.
Two things caching does not fix:

- **Accuracy.** The catalog contains `New Patient Consultation` *and* `New Patient Visit`;
  `Skin Cancer Screening` *and* `Full Body Skin Exam`; three duplicated provider names; and six
  cross-cutting booking policies that must hold simultaneously. Narrowing the choice to a
  handful of rows before the model decides is the central bet of this design — a bet argued
  from the structure of the data rather than from a head-to-head measurement, which §7 sets out
  honestly. In a phone call the failure mode is an appointment the clinic cannot honour,
  discovered when the patient arrives.
- **Latency, at scale.** A model cannot emit its first token until it has read the entire
  prompt, and that silence falls between the caller finishing their sentence and the agent
  starting to speak. The effect is not visible at this catalog size: across 17 measured
  turns spanning 42–1,434 prompt tokens, time-to-first-token correlated with prompt length at
  only +0.46, dominated by network and queueing rather than by prompt size. It becomes a real
  cost at the catalog sizes in the scaling table (§7), where the naive prompt reaches tens of
  thousands of tokens. Treating it as a present-day argument at 5,449 tokens would overstate
  the case.

The design therefore optimises for **how much the model must read at the moment it decides**,
and treats cost as a consequence rather than the target.

---

## 3. Architecture

```
                    catalog.json  (data)
                          │
   ┌──────────────────────┴───────────────────────┐
   │  catalog/            no LLM. deterministic.  │
   │    store.py      records + inverted indices  │
   │    policies.py   the booking rules           │
   │    query.py      find_bookable() — ONE solver│
   │    index.py      hybrid retrieval            │
   └──────────────────────┬───────────────────────┘
                          │
   ┌──────────────────────┴───────────────────────┐
   │  tools/     budget context · speak · recover │
   └──────────────────────┬───────────────────────┘
                          │
   ┌──────────────────────┴───────────────────────┐
   │  agent_builder/   JSON → Pipecat Flows graph │
   └──────────────────────┬───────────────────────┘
                          │
              bot.py — the voice pipeline
```

The division of labour is the organising idea:

| | Responsibility | Rationale |
| --- | --- | --- |
| **LLM** | Language and world knowledge: mapping "my knee has been hurting for weeks" to Orthopedics, asking a question when two options are genuinely ambiguous | The model already has this; approximating it in code would be worse |
| **Code** | Exactness: filtering, applying the policies, cross-checking provider × location × type | No hallucination surface in a `for` loop |

A consequence worth stating: the correctness-critical half contains no LLM, so it is covered by
ordinary unit tests that run in 100 milliseconds. Voice agents are otherwise difficult to test,
and this moves most of the system into territory where testing is cheap.

---

## 4. Design decisions

### 4.1 One constraint solver; every tool is a projection of it

The booking policies are cross-cutting: the same rule constrains searching for options, offering
slots, and confirming a booking. Implemented per-tool, they would drift, and the drift would
surface as an appointment the clinic cannot honour.

There is therefore exactly one query path:

```python
find_bookable(catalog, *, specialty=None, appointment_type_id=None,
              provider_id=None, location_id=None, patient=Patient()) -> list[BookableCombo]
```

`None` means "not asked yet", which matches how a conversation accumulates constraints —
incrementally and out of order ("with Dr. Garcia" … "near Mission" … "and it needs to be an
MRI"). `list_appointment_types`, `find_appointments` and `book_appointment` are all projections
of this one call, so no path can bypass a policy — including the booking step, which re-runs the
solver to validate whatever the model selected.

**Alternative rejected:** a catalog handler with one method per use case. It degenerates into a
combinatorial explosion of methods (`get_providers_for_type_and_location_and_patient`) that each
re-derive the same rules, which is precisely the drift the single path prevents.

### 4.2 Retrieval narrows; it never decides

Entering the hierarchy costs **zero prompt tokens**: `find_specialties(complaint)` takes the
caller's own words rather than a specialty name. This also removes hallucinated specialties at
this step, because the model relays text instead of selecting from a vocabulary it was never
shown.

**Why a vector database is not the decision-maker.** On this dataset it structurally cannot be.
`New Patient Consultation` and `New Patient Visit` are semantically identical; an embedding
places them at the same point. What separates them is a business rule, not a meaning.
**Retrieval's job is to produce a short candidate list; disambiguation is a conversational act
or a deterministic rule.** Any design that expects the vector store to select the right row is
wrong for this data.

Three implementation choices carry most of the accuracy:

1. **Enriched documents, not labels.** `"knee pain"` against the bare string `"Orthopedics"` is
   a weak match, because the link is clinical inference rather than similarity. Each specialty's
   document is its name, its appointment type names, and **452 curated bilingual symptom
   aliases** (`backend/data/specialty_aliases.json`). This is the cheapest and highest-leverage
   part of the retrieval design, and in production the alias vocabulary is something the clinic
   owns and curates.
2. **Hybrid, not pure vector.** A lexical channel is fused with the vector channel by reciprocal
   rank fusion, which requires no score calibration between them. Embeddings handle rare exact
   tokens poorly — `MRI`, `DEXA`, and the Spanish `mamografía` — while the lexical channel
   matches them exactly. The bilingual example is deliberate: the alias vocabulary covers both
   languages, and cross-lingual embedding similarity is exactly where a vector-only design is
   weakest.
3. **The LLM is the reranker.** The shortlist returns to the model with appointment-type counts
   and it selects. No cross-encoder: the model is already in the loop, already paid for, and
   already holds the clinical knowledge a general-purpose embedding model only approximates.

The lexical channel requires no network, so the whole system runs with no embeddings at all —
`make test` and `make benchmark` both force that path. `make index` precomputes the vectors with
one batched `text-embedding-3-small` call at 256 dimensions.

**A trap worth documenting: partial vector coverage is worse than none.** Rank fusion rewards
appearing in both channels, so a specialty missing from a stale cache ranks below worse matches
that happen to be present — silently, and only in the configuration that ships. The index now
refuses a cache that does not cover every document and falls back to lexical with a warning
(`test_partial_vector_coverage_disables_the_channel`).

#### Both strategies ship, and switching between them is a configuration change

Retrieval is not the only option, and it is not always the right one. Injecting the specialty
list costs tokens but no round trip; retrieving costs a round trip but no tokens. Which wins
depends on how long the list is, so **both are built and either can be selected per node from
the builder UI, without touching code**:

| | How it is configured | Prompt cost | Round trips |
| --- | --- | --- | --- |
| **Retrieval** (shipped default) | Give the node the `find_specialties` tool | 0 tokens | +1 LLM call |
| **Injection** | Remove that tool; write `{specialties}` in the node prompt | 57 tokens | none |

`{specialties}` renders the bookable specialty list at node-build time from the catalog itself,
so it cannot drift from the data the way a hand-maintained list would. Both are two clicks apart
in the node inspector: uncheck the tool, edit the prompt.

**The trade-off.** Measured time-to-first-token in this system ranges from 0.5 s to 4 s per
model call, so retrieval's extra round trip is a real pause on a phone call rather than a
rounding error. Injection avoids it entirely, and at 57 tokens it is strictly cheaper than a
round trip while the list stays short. **The crossover is around 50 specialties**, beyond which
the injected list starts costing more than it saves and stops fitting comfortably in a prompt
the model must read every turn.

The shipped agent retrieves, because that is what survives a catalog ten times this size — but
on this catalog, at 18 bookable specialties, injection is the faster configuration and is a
legitimate choice. Making that switchable rather than deciding it once is the point: the right
answer depends on a customer's catalog, not on this one.

### 4.3 Data tools versus edges

The starter compiled every tool into a node transition. Pipecat Flows itself supports
non-transitioning tools — a handler returning `next_node=None` — so the node schema gained a
`tools` field alongside `edges`:

| | Moves the graph | Example |
| --- | --- | --- |
| **tool** | No — the result enters context and the step continues | `find_specialties`, `find_appointments` |
| **edge** | Yes — a decision has closed | `select_appointment_type`, `finish` |

This is what allows a node to search the catalog as many times as the conversation requires.
"Not those times" has to be another tool call inside the same node; if it required a graph
transition, the graph would need a node per possible outcome.

### 4.4 Context is recycled per node, not accumulated per call

Because transitions now mark closed decisions, they are exactly where a context reset is safe.
The `find_appointment` node declares `context_strategy: "reset"`: every search result the
previous step produced is dropped, and the facts that must survive return through templated
`task_messages` (`{appointment_type_name}`, `{full_name}`) for a couple of dozen tokens.

**This accounts for 61% of the total token saving** (§7), and it is the only mechanism in the
design that stops context growing with call length.

**Its cost, discovered by running the evals rather than by reasoning about them.** The first
smoke run passed every policy check and then switched to English halfway through a Spanish call:
the reset had discarded the only evidence of what language the caller spoke. Later, a caller who
named her doctor in her opening sentence was booked with someone else, because the name was
stated before the reset and nothing carried it across.

Twice is a pattern rather than an accident: **anything that must survive a reset has to be named
as a slot, and whatever is forgotten disappears silently.** That is the real price of recycling
context, and it is worth paying at 61% of the saving, but it has to be paid deliberately.
`record_patient` captures `language`; `select_appointment_type` captures `provider_preference`
and `location_preference` — exactly the preferences the brief asks to honour — and reset nodes
restate all three. A static test enforces that every reset node interpolates *something*, which
catches the shape of the mistake but not a specific missing slot; the evals close that gap.

### 4.5 Offers are opaque ids, so an illegal booking cannot be composed

`find_appointments` returns fully-specified offers — provider **and** site **and** time, already
policy-valid — and `book_appointment(option_id)` takes a single opaque id from the list just
produced. The model selects; it never assembles. The hallucination surface at the booking step
is therefore zero, and the solver re-validates regardless.

This also collapses what would otherwise be two nodes. A caller who says "whatever is soonest, I
don't mind who" is stating a *constraint*, not skipping a step, so slot search runs across all
viable combinations rather than inside one already-chosen provider. Multi-location
disambiguation — the sixth catalog policy — then resolves implicitly, because every offer names
its own site.

### 4.6 Policies bind; preferences bend

A policy decides what is **legal**. A preference decides what is **desirable**. Conflating them
is how a booking system starts refusing appointments the clinic would happily make.

`catalog/policies.py` holds exactly the rules `catalog.json` states — five filtering rules plus
one conversational obligation — and `Patient` carries only the facts those rules read
(`is_new`, `has_referral`). Everything a caller might merely want lives in the tool layer:

| | Example | When it cannot be satisfied |
| --- | --- | --- |
| **Policy** | this site has no imaging | Refuse, and explain why |
| **Preference** | this doctor, this site, soonest | Widen the search, and say what was dropped |

Three behaviours follow, each of which fixed a failure the evals surfaced:

- **A language wish ranks; it never filters.** Nothing in the catalog makes it illegal to book a
  doctor who does not speak the caller's language, and the brief lists the preferences to honour
  as *provider, location, soonest*. An earlier version enforced language as a policy, and it
  cost a real booking: a caller speaking Spanish had that read as a demand for a
  Spanish-speaking doctor, and a bookable echocardiogram returned "no availability" because the
  cardiologist speaks only English.
- **Unresolvable is not ambiguous.** Two doctors sharing a name is a question only the caller can
  settle, so that stops and asks, returning each candidate with a `provider_id` to select. "My
  usual family doctor" is not a name and no amount of asking will make it one, so it is dropped,
  the search runs without it, and the result declares what was ignored.
- **Relaxing cannot produce an illegal result**, by construction rather than by care. Dropping a
  filter enlarges the candidate set; every combination returned is still checked against every
  policy, and `book_appointment` re-validates besides. Two unit tests pin this: that the policy
  list is exactly the catalog's, and that a widened search is still wholly legal.

### 4.7 Considered and rejected: a specialty hierarchy

Real medical taxonomies — SNOMED, the NUCC provider taxonomy — are DAGs rather than trees:
"Sports Medicine" belongs under both Orthopedics and Primary Care, "Pediatric Cardiology" under
both parents. A tree forces a wrong choice and then loses recall through the branch not taken.

With a flat enriched index, hierarchy also buys no recall at all: it would add a traversal hop
and another opportunity to be wrong. It would earn its place for human browsing in the UI and
for sharding the index at real scale, and if it is ever needed it should be **many-to-many tags,
not a tree**, which dissolves the multi-parent problem by construction. At 21 specialties,
building it would be over-engineering.

---

## 5. The scheduling agent

```
greeting ──▶ intake ──▶ identify_need ──▶ find_appointment ──▶ farewell (ends the call)
   │                    ↑ tool loop        ↑ tool loop
   └──(question)──▶ qa
```

Four booking nodes rather than five. The two middle nodes are **refinement loops**; a transition
fires only when a decision has actually closed, which is exactly when a context reset is safe.

| Node | Does | Notes |
| --- | --- | --- |
| `greeting` | Routes to booking or Q&A | |
| `intake` | Name, new-or-returning, language | Collected before any catalog lookup, because new-patient status prunes the search space |
| `identify_need` | `find_specialties` → `list_appointment_types` → optionally `describe_appointment_type` | Loops until the type is settled; asks about a referral only if the chosen type requires one |
| `find_appointment` | `find_appointments`, re-called freely as the caller refines | Declares `context_strategy: reset` |
| `farewell` | Reads the booking back and ends the call | |
| `qa` | Read-only catalog tools | Reachable from the greeting |

Seven tools are exposed in total: `find_specialties`, `list_appointment_types`,
`describe_appointment_type`, `find_appointments`, `book_appointment`, `find_providers`,
`get_location_info`.

**Self-correction.** A tool must never hand the agent an empty result it cannot act on. A
hallucinated specialty returns `did_you_mean` resolved through the retrieval index rather than by
edit distance — character similarity places "Traumatology" next to *Dermatology*, whereas the lay
vocabulary places it next to *Orthopedics*, which is the useful answer. An impossible request
returns the reason and the nearest alternatives: asking for an X-ray at a site without imaging
produces "needs a site with imaging" plus the sites that have it.

---

## 6. Phase 1 — the builder

The UI creates and deletes agents, adds, renames and deletes nodes, marks a node terminal or as
the entry point, draws edges by dragging between nodes, and edits the slots an edge collects. It
also browses the catalog and places a test call.

Decisions worth noting:

- **Node names are identity.** Edges target them by name and `initial_node` names one, so
  renaming rewrites every edge that pointed at the old name and deleting takes inbound edges with
  it — in a single state update, because the API compiles before it writes and would reject a
  half-updated graph. These operations live in `frontend/src/graphOps.js` as pure functions and
  are tested directly (17 tests); clicking through a browser proves a control is wired, whereas
  these prove the graph survives the edit.
- **Two classes of invalid, treated differently.** Duplicate node names, edges to missing nodes
  and unknown tools cannot compile, so the API rejects them with 422 and nothing reaches disk.
  Dead ends, unreachable nodes and a graph that never hangs up are shown as warnings and never
  block saving, because a graph is built a piece at a time and an editor that refuses to save
  half-built work is unusable.
- **Slots are not plumbing.** The properties an edge collects are the arguments the model fills
  in, and they land in `flow_manager.state` — which makes them the only things that survive a
  context reset. `provider_preference` exists because a preference stated before a reset was
  otherwise lost (§4.4).
- **Layout is derived, not stored.** A generated agent should not have to invent coordinates, so
  an unarranged graph opens readable via breadth-first placement. Once a node is dragged, that
  position is authoritative and saved with the agent.
- **A live context-cost panel.** Phase 2's work is otherwise invisible — a cheap, accurate agent
  and an expensive, confused one sound identical. The panel reads real token usage from the
  pipeline's `MetricsFrame`, so nothing in it is estimated, and traces every tool call and node
  transition as they happen.

Agents are files in `backend/data/agents/`. Saving is a write and deploying is a reconnect,
because the runner re-reads the agent JSON on every connection.

---

## 7. Evidence

### Cost

`make benchmark`, exact counts via tiktoken:

| Strategy | Tokens for a 12-turn booking |
| --- | --- |
| Naive — whole catalog in the system prompt | **65,388** (5,449 re-sent every turn) |
| This design, `append` — tool results linger | 4,185 |
| This design, `reset` — dropped once the type is settled | **1,653** |

**40× cheaper, and 61% of that saving comes from the context reset alone.**

These figures count re-sends. A tool result added on turn 4 is part of the prompt on every
subsequent turn, so "paid once" would be inaccurate; the reset is what makes it nearly true.

The figure that matters for accuracy is different: **353 tokens** is the largest single tool
result — the most the model ever reads at once, against 82 appointment types in the naive
prompt. Prompt caching does nothing for that.

### Scaling

| Catalog size | Naive prompt | What the model reads |
| --- | --- | --- |
| 1× | 5,449 | 353 |
| 10× | 54,490 | 353 |
| 50× | 272,450 | 353 |

The naive prompt scales with the catalog. The slice the model reads does not: a clinic ten times
this size has more specialties, not ten times more MRI variants.

### Accuracy

- **154 Python unit tests** (`make test`) covering every policy, the solver, retrieval, the
  builder API and static checks on the shipped agent graph — no API keys, no network, ~100 ms.
- **17 JavaScript tests** (`make test-ui`) covering the graph edits where a mistake is silent:
  rename and delete cascades, name uniqueness, and the warning rules.
- **16 scripted scenarios** (`make evals`) that drive the *same compiled graph* the voice
  pipeline runs, over text, asserting on the outcome — what was booked and whether it was legal
  — rather than on wording. They cover the traps: a knee MRI needing both a referral and an
  imaging site; a provider practising at four sites who offers a gated service at two; the
  `Dr. Maria Garcia` ambiguity; new-patient restrictions; near-duplicate annual visits; a caller
  who rejects the first times offered; and a specialty the clinic advertises but cannot staff.

**Results.** Seven full runs, in order: **11, 7, 12, 11, 13, 16, 16 of 16.** Each step followed
from fixing a defect the previous run exposed; the 7 was a regression introduced and then caught
(the stale-vector trap in §4.2). The last two runs are the same code twice.

Two clean runs are two samples rather than a stable accuracy figure, and scenarios do flip
between runs even at `temperature=0`, since the provider offers no determinism guarantee without
a seed. The trend is meaningful because every step traces to a specific fix; the final digit is
not.

**The figure that does hold: across all seven runs — roughly a hundred scenario executions —
zero illegal bookings.** No provider booked outside their sites, no capability-gated service at
a site lacking it, no new patient on a restricted type, no referral-gated type without one. That
is the claim the architecture makes; it is re-verified against the catalog independently of the
agent in `_policy_failures`, and it held on every run, including the run that scored 7.
Completion accuracy moved substantially; legality never moved at all. That separation is the
design behaving as intended.

**What the evals found that 154 unit tests could not.** This is the argument for building the
harness, so it is worth being concrete:

1. *A misroute worse than a dead end.* Unstaffed specialties were excluded from the retrieval
   index, which looked safer since nothing there is bookable. In practice the query then matched
   the next-closest specialty: a caller asking for an eye exam was booked a general follow-up,
   and would have discovered it in the waiting room. They are now indexed and flagged
   `not_currently_offered`.
2. *Preferences dying at a context reset.* Twice — the conversation language, then a named
   doctor. Both are now slots (§4.4).
3. *A tool advertising a name it would not accept.* Ambiguous-provider candidates were shown as
   "Dr. Maria Garcia, MD" while the resolver matched only "Dr. Maria Garcia", so the model could
   not disambiguate using the words it had just been handed. Candidates now carry `provider_id`
   — the same pick-from-a-list-we-produced mechanism that makes booking hallucination-free.
4. *A dead end the agent circled.* "My usual family doctor" is not a name; returning
   `unknown_provider` had the agent ask for it three times in succession. Unresolvable
   preferences are now dropped and declared (§4.6).
5. *An invented requirement, and the regression it caused.* Language enforced as a policy (§4.6).

None of these are visible without running the graph end to end. Unit tests cover what the domain
must do; only a conversation reveals what the agent does with it.

### What is not measured

Cost is measured for both strategies; accuracy is measured only for this one. The claim that the
naive approach is *less accurate* — that a model choosing from 82 near-identical rows makes
mistakes a model choosing from four does not — is reasoned, not demonstrated. Demonstrating it
would require a second arm over the same scenarios and the same assertions, with the whole
catalog in the prompt and a `book(type_id, provider_id, location_id)` tool the model composes
itself. That last detail is what would make the comparison fair: giving the naive arm the same
opaque-offer booking tool would hand it the very property under test. The scenarios and the
legality assertions already support such an arm; it is simply not built.

---

## 8. What the catalog turned out to contain

Two findings changed the implementation:

- **Eight appointment types have no provider at all** — the whole of Ophthalmology, Physical
  Therapy and Urology. The clinic advertises services nobody is staffed for. Routing a caller
  there is a dead end discovered at the end of the call, so the booking funnel is built from
  `bookable_specialties()` (18 of 21) while retrieval still finds them and marks them
  unavailable. The catalog browser surfaces them rather than hiding them, because it is a data
  problem someone should fix.
- **Providers offer appointment types outside their own specialty** — internists book `General`
  and `Family Medicine` types. Specialty filtering must therefore be on the **appointment type's**
  specialty, never the provider's; the other way loses most of primary care, which is the most
  common case.

---

## 9. Scope

**Built:** the domain layer and its tests, hybrid retrieval, mocked availability, the schema
extension (data tools, context strategy, prompt templating), seven tools, the scheduling agent,
the eval harness and cost benchmark, the builder UI, and the live cost panel.

**Mocked:** availability. Deterministic, seeded from `(provider, location, date)` so demos
replay identically. It models one thing deliberately: a provider practising at several sites is
at only one of them per day, which is what makes "book with Dr. Chen" genuinely require a
location rather than as a formality.

**Deliberately not built:**

| Omitted | Reasoning |
| --- | --- |
| Reschedule and cancel | No new context-management surface, considerable state. The interesting problem is navigating the catalog |
| A specialty hierarchy | Reasoned about and rejected (§4.7) |
| Nested JSON-schema editing for edge slots | A flat name/type/description/required table covers every slot the shipped agent uses; the JSON stays hand-editable for anything exotic |
| Auth, multi-tenancy, a database | Not what the challenge is testing |

---

## 10. Limits, and what changes at scale

- **Specialties stop fitting in one retrieval pass** somewhere in the low thousands. The fix sits
  behind the `SpecialtyIndex` Protocol: replace `InMemorySpecialtyIndex` with a pgvector or
  Qdrant implementation, fetch top-N from the database, rerank in code. **No tool handler
  changes** — that substitution is the purpose of the seam.
- **The alias vocabulary is hand-curated.** It is the highest-leverage data in the system and it
  does not scale by hand past a few hundred specialties. A production system would generate a
  first pass from historical call transcripts and have the clinic curate it.
- **`find_appointments` probes availability for up to 25 combinations.** Against a real calendar
  API that is 25 network calls; it would need a batch availability endpoint or a cached free/busy
  index.
- **No concurrency control.** Two callers can be offered the same slot, because nothing is held.
  A real system needs a soft hold at offer time.
- **The embedded WebRTC client does not recover from a session the server ended.** That is a
  dependency rather than code in this repo; the builder works around it by remounting the client
  when a call finishes.

---

## 11. Demo walkthrough

```bash
make install && make ui && make run     # then open http://localhost:7860/builder
```

1. **Build an agent from nothing.** New agent → rename a node → add a second → connect them by
   dragging → give the edge a slot → mark the second node terminal → save → *Set live*. Invalid
   graphs are refused at save time; incomplete ones are only warned about.
2. **Browse the catalog** to show the scale and the messiness the agent navigates — including the
   eight advertised-but-unbookable types the browser flags.
3. **Place a call** against the scheduling agent: *"my knee has been hurting for weeks and my
   orthopedist sent me for an MRI."* It narrows to Orthopedics and Radiology, asks about the
   referral, offers only the three imaging-capable sites, and books.
4. **Repeat the opening line in Spanish** to show the same path working end to end in another
   language, and the conversation language surviving the context reset as a slot (§4.4).
5. **Watch the context panel** while it happens: prompt tokens per model call, every tool call,
   and the node transition where the context is reset — next to the conversation producing it.
6. **Switch to the graph mid-call** — the call keeps running — and show that the largest thing
   the model ever read was a few hundred tokens.
7. **Swap retrieval for injection live** (§4.2): uncheck `find_specialties` on `identify_need`,
   put `{specialties}` in its prompt, save, reconnect. The same agent now trades 57 prompt
   tokens for one fewer round trip, and the panel shows the difference on the next call.
