# Solution — voice scheduling agent with a context-managed catalog

**The catalog never enters the prompt.** It sits behind deterministic tools, the booking
policies are enforced in code, and the conversation graph exposes only the tools each step
needs — so the model chooses between three or four options instead of eighty-two.

Measured on the provided catalog (8 locations, 50 providers, 82 appointment types):

| | |
| --- | --- |
| Input tokens for one booking | **7× fewer** — 12,454 vs 88,740, measured end to end |
| Largest catalog excerpt the model ever reads | **353 tokens**, against 82 appointment types |
| Illegal bookings across 7 eval runs (~100 scenarios) | **0** |
| Deterministic tests | **154** Python + **17** JavaScript, offline |

```bash
make install && make ui     # venv, then build the builder UI
make run                    # http://localhost:7860/builder
make test                   # 154 unit tests, no API keys, ~0.1s
make benchmark              # reproduces the token figures
make evals ARGS=--dry-run   # 16 accuracy scenarios; drop ARGS to run them
```

---

## 1. Requirements coverage

| Asked for | Where |
| --- | --- |
| UI to create an agent and place a test call | `frontend/` — create/wire/delete agents, edit nodes, edges, prompts, slots |
| Create appointments efficiently | `find_appointments` → `book_appointment` |
| Offer available slots | Offers carry provider + site + time together |
| Match request → type, location, provider | Retrieval narrows (`catalog/index.py`); the solver decides legality (`catalog/query.py`) |
| Disambiguate similar options | Duplicate names return every candidate with an id; near-duplicate types are separated by a question, never guessed |
| Honour preferences | Provider, location and soonest-available carried as slots that survive a context reset |
| Answer questions | Read-only Q&A node: `find_providers`, `get_location_info`, `describe_appointment_type` |

Beyond the brief: a live context-cost panel reading real token usage off the pipeline, and an
offline eval harness that drives the compiled graph over text.

---

## 2. The problem

Flattening the catalog into a system prompt costs **5,449 tokens, re-sent every turn** —
65,388 for a twelve-turn booking. Prompt caching recovers much of that in money terms. It does
not fix the two things that matter:

- **Accuracy.** The catalog holds `New Patient Consultation` *and* `New Patient Visit`;
  `Skin Cancer Screening` *and* `Full Body Skin Exam`; three duplicated provider names; six
  cross-cutting policies that must hold at once. In a phone call the failure mode is an
  appointment the clinic cannot honour, discovered when the patient arrives.
- **Latency.** A model cannot emit its first token until it has read the whole prompt, so the
  longer the prompt, the longer the silence before the agent speaks. Minor at this catalog size,
  decisive at the sizes in the scaling table (§6).

The design optimises for **how much the model must read at the moment it decides**, and treats
cost as a consequence.

---

## 3. Architecture

```
catalog.json
    │
    ├─ catalog/     records, indices, policies, solver, specialty retrieval   ← no LLM
    ├─ tools/       budget context · shape for speech · recover
    ├─ agent_builder/   JSON → Pipecat Flows graph
    └─ bot.py       the voice pipeline
```

| | Responsibility | Why there |
| --- | --- | --- |
| **LLM** | Language and world knowledge: symptom → specialty, asking when genuinely ambiguous | The model already has this |
| **Code** | Exactness: filtering, the policies, provider × location × type | No hallucination surface in a `for` loop |

The correctness-critical half contains no LLM, so it is covered by unit tests that run in 100 ms.
Voice agents are otherwise hard to test; this moves most of the system somewhere testing is cheap.

---

## 4. Design decisions

### 4.1 One constraint solver — every tool is a projection of it

```python
find_bookable(catalog, *, specialty=None, appointment_type_id=None,
              provider_id=None, location_id=None, patient=Patient()) -> list[BookableCombo]
```

- **Policies are cross-cutting** — the same rule constrains searching, offering slots and
  confirming. Implemented per-tool they would drift, and drift becomes an unhonourable booking.
- **`None` means "not asked yet"**, which is how conversations accumulate constraints: out of
  order, a bit at a time.
- **No path bypasses a policy**, including the booking step, which re-runs the solver to validate
  what the model selected.
- *Rejected:* a handler with one method per use case — it explodes into
  `get_providers_for_type_and_location_and_patient` variants that each re-derive the same rules.

### 4.2 Narrow by specialty before appointment type

The funnel goes complaint → specialty → appointment type, rather than complaint → appointment
type directly. **Specialty is the level where matching works.**

Measured over ten complaints, asking whether the right specialty appears in the top three:

| Searching | Hit rate |
| --- | --- |
| Specialties | **8/10** |
| Appointment types directly | **4/10** |

- **Specialties are ~20 well-separated concepts** — Cardiology does not resemble Dermatology.
  Appointment types are 82 overlapping names *inside* those specialties: `New Patient
  Consultation` vs `New Patient Visit`, the MRI Brain/Spine/Knee family. Searching there means
  searching at the level where nothing is separable.
- **Most direct searches return nothing at all.** Type names are clinical terminology —
  `Well-Child Visit`, `Mammogram` — and nobody phones asking for those.
- **The one that does match is worse than nothing.** *"My knee has been hurting for weeks"*
  retrieves `MRI - Knee` on the literal word "knee", when the caller probably needs an
  orthopedic consultation. The only literal match is the most specific and most expensive option.
- **Cost.** All 82 types is 3,292 tokens; one specialty is 161 median, 436 at its widest, and the
  model chooses among three to eight options rather than eighty-two.
- **A wrong turn is cheap.** A mistaken specialty costs ~50 tokens to discover and retry. A
  mistaken appointment type is discovered when the patient arrives.

So retrieval runs where it is reliable, and what similarity cannot separate is settled by asking.

### 4.3 Specialty retrieval narrows; it never decides

`find_specialties(complaint)` takes the caller's own words, not a specialty name. **No catalog
content sits in the prompt** — the cost is the tool's 86-token schema, which does not grow as the
catalog does — and the model cannot hallucinate a specialty it was never shown.

**Why it returns three candidates rather than one.** Retrieval knows the clinic's vocabulary and
lacks clinical judgement; the model is the exact reverse. Asked to name the specialty for ten
complaints with no list and no tools, the model was conceptually right every time but produced a
string the catalog actually contains only **6/10** — `Ortopedia` and `Psiquiatría` when the caller
spoke Spanish, `Pathology` where this clinic says `Lab`, `Dentistry` where it says `Dental`.
Retrieval fails the other way: a valid catalog string **9/10**, but the conceptually right one
also only 6/10, since lexical matching depends on the words being in the alias file.

Neither is sufficient alone, so the design uses both: retrieval returns three candidates — **the
right one is among them 8/10** — and the model picks. Retrieval supplies the vocabulary; the model
supplies the judgement. That is what "narrows, never decides" means in practice.

Three choices carry most of the accuracy:

- **Enriched documents, not labels.** `"knee pain"` against the bare string `"Orthopedics"` is a
  weak match — the link is clinical inference. Each specialty's document is its name, its
  appointment type names, and **452 curated bilingual symptom aliases**. Cheapest, highest-leverage
  part of the design; in production the clinic owns that vocabulary.
- **Hybrid, not pure vector.** A lexical channel is fused with the vector channel by reciprocal
  rank fusion. Embeddings handle rare exact tokens poorly — `MRI`, `DEXA`, `mamografía` — and
  lexical matches them exactly. The lexical channel needs no network, so the system runs with no
  embeddings at all.
- **The LLM is the reranker.** The shortlist returns to the model, which selects. No cross-encoder:
  the model is already in the loop and already holds the clinical knowledge.

**A product argument that holds at any scale:** with retrieval, the mapping from lay language to
the clinic's terms is a versioned file the customer edits. When a clinic says *"callers here ask
for the sugar doctor — send them to Endocrinology"*, that is a line in `specialty_aliases.json`.
Left to the model, it is a line in a prompt.

### 4.4 Both narrowing strategies ship, selectable per node

| | Configured by | Prompt cost | Scales with catalog? | Round trips |
| --- | --- | --- | --- | --- |
| **Retrieval** (default) | Give the node the `find_specialties` tool | 86 tokens of tool schema, plus 69 for its result once called | **No** — constant | +1 LLM call |
| **Injection** | Remove the tool; write `{specialties}` in the prompt | 57 tokens today | **Yes** — ~3.2 tokens per specialty | none |

- Both are **two clicks apart in the node inspector** — no code change.
- `{specialties}` renders from the catalog at node-build time, so it cannot drift from the data.
- Measured time-to-first-token is 0.5–4 s per model call, so retrieval's extra round trip is a
  real pause, not a rounding error.
- **The crossover is ~49 specialties**, where the injected list grows past retrieval's constant
  155. Below it injection is cheaper *and* avoids the round trip, so on this catalog (18 bookable)
  it wins on both axes. Above it the list keeps growing and retrieval does not. The shipped agent
  retrieves because that is what survives a 10× catalog.
- Making it switchable rather than deciding once is the point: the right answer depends on the
  customer's catalog, not on this one.

### 4.5 Data tools vs edges

| | Moves the graph | Example |
| --- | --- | --- |
| **tool** | No — result enters context, the step continues | `find_specialties`, `find_appointments` |
| **edge** | Yes — a decision has closed | `select_appointment_type`, `finish` |

The starter compiled every tool into a transition. Pipecat Flows supports non-transitioning
tools, so nodes gained a `tools` field. This is what lets a node search repeatedly: *"not those
times"* must be another tool call in the same node, or the graph would need a node per outcome.

### 4.6 Context is recycled per node, not accumulated per call

Because transitions now mark closed decisions, they are exactly where a reset is safe.
`find_appointment` declares `context_strategy: "reset"`.

- **Worth 61% of the catalog-token saving** (§6) — the only mechanism that stops context growing
  with call length.
- Facts that must survive return through templated `task_messages` for a couple of dozen tokens.
- **The cost: anything that must survive a reset has to be named as a slot.** `record_patient`
  captures `language`; `select_appointment_type` captures `provider_preference` and
  `location_preference` — exactly the preferences the brief asks to honour. A static test
  enforces that every reset node interpolates something.

### 4.7 Offers are opaque ids, so an illegal booking cannot be composed

`find_appointments` returns fully-specified offers — provider **and** site **and** time, already
policy-valid — and `book_appointment(option_id)` takes one opaque id from that list.

- The model selects; it never assembles. Hallucination surface at booking: zero.
- Slot search runs across *all* viable combinations, so "whatever is soonest, I don't mind who"
  is a constraint rather than a skipped step.
- Multi-location disambiguation resolves implicitly, because every offer names its own site.

### 4.8 Policies bind; preferences bend

| | Example | When unsatisfiable |
| --- | --- | --- |
| **Policy** | this site has no imaging | Refuse, and explain why |
| **Preference** | this doctor, this site, soonest | Widen the search, and say what was dropped |

`catalog/policies.py` holds exactly the rules `catalog.json` states — five filtering rules plus
one conversational obligation — and `Patient` carries only the facts those rules read. Everything
a caller merely *wants* lives in the tool layer.

- **A language wish ranks; it never filters.** Nothing makes it illegal to book a doctor who does
  not speak the caller's language, and the brief lists the preferences to honour as *provider,
  location, soonest*.
- **Unresolvable is not ambiguous.** Two doctors sharing a name is a question only the caller can
  settle, so that stops and asks with a `provider_id` per candidate. *"My usual family doctor"* is
  not a name and never will be, so it is dropped, the search runs, and the result says so.
- **Relaxing cannot produce an illegal result, by construction.** Dropping a filter enlarges the
  candidate set; every combination is still checked against every policy. Two tests pin this.

### 4.9 Rejected: grouping specialties into categories

The funnel in §4.2 is two levels — specialty, then appointment type. A third level above it,
grouping the 21 specialties into categories, was considered and rejected.

Real medical taxonomies (SNOMED, NUCC) are DAGs, not trees — "Sports Medicine" belongs under both
Orthopedics and Primary Care. A tree forces a wrong choice and loses recall through the branch not
taken. With a flat enriched index it also buys no recall, only a hop and another chance to be
wrong. If ever needed it should be **many-to-many tags, not a tree**.

---

## 5. The agent

```
greeting ──▶ intake ──▶ identify_need ──▶ find_appointment ──▶ farewell (ends call)
   │                     ↑ tool loop        ↑ tool loop, resets context
   └──(question)──▶ qa ──┘
```

Four booking nodes. The two middle ones are **refinement loops** — they call tools repeatedly
without moving — and a transition fires only when a decision has closed.

| Node | Tools available | Collects | Notes |
| --- | --- | --- | --- |
| `greeting` | — | — | Routes to booking or Q&A |
| `intake` | — | `full_name`, `is_new_patient`, `language` | Before any catalog lookup: new-patient status prunes the search |
| `identify_need` | `find_specialties`, `list_appointment_types`, `describe_appointment_type` | `appointment_type_id`, `has_referral`, `provider_preference`, `location_preference` | Loops until the type is settled; asks about a referral only if the type needs one |
| `find_appointment` | `find_appointments`, `find_providers`, `get_location_info`, `book_appointment` | — | **Resets context on entry**; re-called freely as the caller refines |
| `farewell` | — | — | Reads the booking back, ends the call |
| `qa` | the five read-only tools | — | Reachable from the greeting |

### What each tool does

| Tool | Returns | Schema cost |
| --- | --- | --- |
| `find_specialties(complaint)` | Ranked specialties for the caller's own words, with how many types each has. Flags any not currently offered | 86 |
| `list_appointment_types(specialty)` | Bookable types in that specialty, each with duration, referral and new-patient flags. Unknown specialty returns `did_you_mean` | 57 |
| `describe_appointment_type(id)` | Duration, requirements, how many sites and providers can do it | 65 |
| `find_appointments(type, …prefs)` | Fully-specified offers — provider **and** site **and** time, already policy-valid — ranked by stated preference, with anything unmatchable declared | **325** |
| `book_appointment(option_id)` | Confirms one offer by opaque id, re-validated through the solver | 80 |
| `find_providers(name/specialty/site/language)` | Doctors, their sites, languages, new-patient status. Duplicate names return every candidate with a `provider_id` | 129 |
| `get_location_info(name?)` | Address, phone, hours, on-site services | 62 |
| | **all seven** | **804** |

Every tool that can fail returns something the agent can act on: suggestions for an unknown
specialty, candidates for an ambiguous name, the reason and the nearest alternatives for an
impossible request. Never an empty list.

---

## 6. Evidence

### Cost

One booking, measured end to end: **14 model calls, 12,454 input tokens.** That is the figure
the live panel shows during a demo. With the catalog in the prompt, the same conversation would
have sent **88,740**, because the catalog is re-sent on every one of those 14 calls.

| One booking | Total input tokens |
| --- | --- |
| Catalog in the prompt | 88,740 |
| Catalog behind tools | **12,454** |
| | **7.1× fewer** |

What remains is mostly not catalog: the persona, the node's prompt, the conversation, and the
tool schemas — which the function-calling protocol re-sends on every call and which are the
largest single component (688 tokens on `find_appointment`).

**Isolating the variable the design changes** — catalog-derived tokens alone — `make benchmark`
reports **65,388 → 1,653 for a 12-turn booking, 40× fewer, 61% of it from the context reset.**
That is the mechanism; the 7.1× above is what the mechanism is worth on a real call.

The figure that matters for accuracy is different again: **353 tokens** is the largest single
tool result — the most catalog the model ever reads at once, against 82 appointment types.
Caching does nothing for that.

### Scaling

| Catalog | Naive prompt | What the model reads |
| --- | --- | --- |
| 1× | 5,449 | 353 |
| 10× | 54,490 | 353 |
| 50× | 272,450 | 353 |

The naive prompt scales with the catalog; the slice the model reads does not. A clinic ten times
this size has more specialties, not ten times more MRI variants.

### Correctness

- **154 Python tests** — every policy, the solver, retrieval, the builder API, static checks on
  the shipped graph. No API keys, no network, ~100 ms.
- **17 JavaScript tests** — the graph edits where a mistake is silent: rename and delete cascades,
  name uniqueness, warning rules.
- **16 scripted scenarios** driving the *same compiled graph* the voice pipeline runs, over text,
  asserting on the outcome — what was booked and whether it was legal — not on wording. They cover
  the traps: a knee MRI needing both a referral and an imaging site; a provider at four sites
  offering a gated service at two; duplicate provider names; new-patient restrictions;
  near-duplicate annual visits; a caller who rejects the first times offered; a specialty the
  clinic advertises but cannot staff.

**Across 7 full runs — roughly a hundred scenario executions — zero illegal bookings.** No provider
booked outside their sites, no capability-gated service where the capability is missing, no new
patient on a restricted type, no referral-gated type without one. That is the claim the
architecture makes, re-verified against the catalog independently of the agent.

*Not measured:* that the naive approach is **less accurate**. That claim is argued from the
structure of the data, not demonstrated. Doing so would need a second arm over the same scenarios
with the catalog in the prompt and a `book(type_id, provider_id, location_id)` tool the model
composes itself — the scenarios and assertions already support it; the arm is not built.

---

## 7. Two things the catalog turned out to contain

- **Eight appointment types have no provider** — all of Ophthalmology, Physical Therapy and
  Urology. Routing a caller there is a dead end found at the end of the call, so the booking funnel
  is built from bookable specialties (18 of 21) while retrieval still finds them and marks them
  unavailable. Hiding them from retrieval is worse: the query then matches the next-closest
  specialty and books something wrong.
- **Providers offer types outside their own specialty** — internists book `General` and
  `Family Medicine` types. Specialty filtering must therefore be on the **appointment type's**
  specialty, never the provider's, or most of primary care disappears.

---

## 8. Scope

**Mocked:** availability. Deterministic, seeded from `(provider, location, date)` so demos replay
identically. It models one thing deliberately — a provider practising at several sites is at only
one per day, which makes "book with Dr. Chen" genuinely require a location.

**Deliberately not built:**

| Omitted | Why |
| --- | --- |
| Reschedule and cancel | No new context-management surface, considerable state |
| A specialty hierarchy | Reasoned about and rejected (§4.9) |
| Nested JSON-schema editing for slots | A flat table covers every slot the agent uses; JSON stays hand-editable |
| Auth, multi-tenancy, a database | Not what the challenge tests |

---

## 9. Where the remaining tokens are, and how to cut them

None of the following is implemented. They are listed because the measurements above point at
them, and because the cheapest wins are no longer in the catalog — they are in what surrounds it.

A booking sends 12,454 input tokens across 14 model calls. Only ~1,653 of those are catalog. The
rest is persona, node prompts, conversation, and **tool schemas — 804 tokens for all seven, of
which `find_appointments` alone is 325**, re-sent on every call in its node.

**1. Trim the widest schema.** `find_appointments` costs 325 tokens because it carries eight
optional preference parameters (provider, location, weekdays, time of day, language, ranking).
Splitting the rare ones into a second tool, or accepting a single free-text `preferences` string
the tool parses, would cut most of it. Cheapest change on this list, and it only touches one
tool definition.

**2. Retrieve tools instead of declaring them all.** The node graph already scopes tools per
step, which is the coarse version of this. The finer version is to expose a small core set plus
a `load_tools(intent)` call that returns the schemas actually needed — schema retrieval, the same
idea as catalog retrieval applied one level up. Worth stating precisely because it sounds more
abstract than it is: the measurement says schemas are the largest single component of a prompt,
so this is where the next order of magnitude lives.
*Cost:* an extra round trip, exactly the trade-off weighed in §4.4, and a model that cannot see
a tool cannot call it — so the routing has to be right or the agent loses a capability silently.

**3. Reset context in more nodes.** Only `find_appointment` resets today. Entering
`identify_need` could also reset, dropping the greeting and intake exchange — worth roughly 600
tokens on every subsequent call in that node, from the measured 1,439-token prompt against 835
of fixed content.
*Cost:* everything the later nodes need must become an explicit slot, and §4.6 records what
happens when one is forgotten — the failure is silent. Each new reset point is a new set of
slots to get right, so this trades tokens for a class of bug that unit tests do not catch.

**4. Summarise instead of carrying the transcript.** Pipecat supports on-demand summarisation.
For long calls the conversation eventually outweighs everything else; a summary at each node
boundary bounds it.
*Cost:* an extra LLM call per boundary, and summaries lose detail the agent may need — which is
the same slot problem as (3), with less control over what survives.

**5. Order the prompt for prefix caching.** Persona and tool schemas are identical across every
call within a node. Keeping them strictly first, ahead of anything that varies, maximises what a
provider's prefix cache can reuse. This does not reduce tokens sent; it reduces what they cost.

**6. Use a smaller model for narrow steps.** `intake` collects a name and a yes/no with no tools
and a 54-token prompt. A cheaper model would serve it, with the frontier model reserved for
`identify_need` and `find_appointment`, where the judgement actually is.
*Cost:* two models to evaluate rather than one, and a quality cliff that only shows up in
conversation.

**What is deliberately not on this list:** compressing the tool *results*. They are already
capped at three options and hold 353 tokens at their widest — the accuracy argument in §6 depends
on the model seeing those options clearly, and squeezing them trades the thing this design is
for against a rounding error.

---

## 10. Limits at scale

- **Specialties stop fitting one retrieval pass** in the low thousands. The fix sits behind the
  `SpecialtyIndex` Protocol: swap `InMemorySpecialtyIndex` for pgvector or Qdrant, fetch top-N from
  the database, rerank in code. **No tool handler changes** — that is the purpose of the seam.
- **The alias vocabulary is hand-curated** and does not scale past a few hundred specialties by
  hand. Production would generate a first pass from call transcripts and have the clinic curate it.
- **`find_appointments` probes up to 25 combinations** — against a real calendar that is 25 network
  calls, needing a batch availability endpoint or a cached free/busy index.
- **No concurrency control.** Two callers can be offered the same slot; a real system needs a soft
  hold at offer time.

---

## 11. Demo

```bash
make install && make ui && make run     # http://localhost:7860/builder
```

1. **Build an agent from nothing** — new agent → rename a node → add a second → connect by
   dragging → give the edge a slot → mark it terminal → save → *Set live*. Invalid graphs are
   refused at save time; incomplete ones only warned about.
2. **Browse the catalog** — the scale and the messiness, including the eight advertised-but-
   unbookable types.
3. **Place a call:** *"my knee has been hurting for weeks and my orthopedist sent me for an MRI."*
   It narrows to Orthopedics and Radiology, asks about the referral, offers only the three
   imaging-capable sites, and books.
4. **Repeat the opening in Spanish** — same path, and the conversation language survives the
   context reset as a slot.
5. **Watch the context panel** — prompt tokens per model call, every tool call, and the node
   transition where context is reset, next to the conversation producing it.
6. **Switch to the graph mid-call** — the call keeps running, and the largest thing the model read
   was a few hundred tokens.
7. **Swap retrieval for injection live** (§4.4) — uncheck the tool, add `{specialties}`, save,
   reconnect. The same agent now trades 57 prompt tokens for one fewer round trip.
