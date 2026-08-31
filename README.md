# Prosper Challenge — Context Management

Voice AI for healthcare scheduling. An agent is a **graph of nodes** (Pipecat Flows), defined
declaratively as JSON and compiled into a runnable voice pipeline.

- **Phase 1** — a UI to edit the node graph and place a test call.
- **Phase 2** — a context-management approach so a scheduling agent can reliably and
  cost-effectively navigate a large catalog of locations, doctors and appointment types.

**The design and its trade-offs are in [`solution.md`](solution.md).** In one line: the
catalog never enters the prompt, the booking policies are enforced in code, and each node
exposes only the tools its step needs — 40× fewer input tokens per call, and the model never
reads more than ~350 tokens of catalog at once.

```
browser mic  ->  ElevenLabs STT  ->  OpenAI LLM  ->  ElevenLabs TTS  ->  browser
                                        ^
                                        └── tools over catalog/, never the prompt
```

## Quickstart

Requires **Python 3.11+** and **Node 18+**. Run from the repo root:

```bash
make install     # Python venv + dependencies
make ui          # build the builder UI (needs npm)
make run         # then open http://localhost:7860/builder
```

Remember to fill in `backend/.env` (see `.env.example`).

The builder, the catalog browser, the test call and the API are all served from
`http://localhost:7860` — one process. `Ctrl+C` to stop, `make help` for every target.

| Target | What it does |
| --- | --- |
| `make test` | 127 unit tests. No API keys, no network, ~0.05s. |
| `make benchmark` | Token cost against the naive full-catalog prompt. |
| `make evals ARGS=--dry-run` | List the 16 accuracy scenarios; drop `ARGS` to run them (makes OpenAI calls). |
| `make index` | Precompute specialty embeddings (one batched embedding call). Optional — retrieval falls back to its lexical channel. |

## Layout

| Path | Responsibility |
| --- | --- |
| `backend/catalog/` | The domain. `store.py` records + indices, `policies.py` the 6 booking rules, `query.py` the single `find_bookable` solver, `index.py` hybrid retrieval. **No LLM anywhere** — deterministic and unit-tested. |
| `backend/tools/` | Thin adapters between the LLM and the domain: budget context, shape for speech, recover from bad input. `registry.py` maps tool names in JSON to handlers. |
| `backend/agent_builder/` | `schema.py` the declarative `AgentConfig`/`Node`/`Edge` contract, extended with node-level `tools` and `context_strategy`; `builder.py` compiles it into a Pipecat Flows graph. |
| `backend/scheduler_flow.json` | The scheduling agent, as data. `example_flow.json` remains as the minimal format example. |
| `backend/bot.py` | The voice pipeline. Loads an agent JSON and runs it; no graph logic. |
| `backend/api.py` | Builder API + live metrics websocket, mounted on the runner's own FastAPI app. |
| `backend/evals/` | The headless flow harness, 16 accuracy scenarios, and the context benchmark. |
| `backend/data/` | `catalog.json` (the provided clinic catalog) and `specialty_aliases.json` (curated bilingual symptom vocabulary). |
| `frontend/` | React + Vite + React Flow: node graph editor, catalog browser, test call with a live context-cost panel. |

To run a different agent, point `AGENT_FLOW` in `bot.py` at another JSON file. The runner
re-reads it on every connection, so saving in the UI and reconnecting is enough — no restart.
