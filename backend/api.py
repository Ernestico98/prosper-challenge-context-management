#
# Builder API — the Phase 1 UI's backend.
#
# Pipecat's dev runner exposes a module-level FastAPI `app`, so these routes ride
# on the same server (:7860) that serves the WebRTC client. One process, no CORS,
# no second port to explain in the demo.
#
# Agents are files on disk. That is the point: an agent is data, so "save" is a
# write and "deploy" is a reconnect — the runner calls bot() per connection and
# re-reads the JSON, so an edit is live on the next call with no restart.
#
# The catalog routes exist because the resource browser needs exactly the queries
# the tools need. Both read the same domain layer; neither reimplements it.
#

import json
import re
from pathlib import Path

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.runner.run import app

from agent_builder import AgentBuilder
from agent_builder.schema import DEFAULT_MODEL, DEFAULT_VOICE_ID
from catalog import (
    Catalog,
    bookable_specialties,
    find_bookable,
    unstaffed_appointment_types,
)
from observability import BROADCASTER
from tools import describe_all

BACKEND_DIR = Path(__file__).parent
# Agents are data, so they live with the rest of the data. The UI names files
# after the agent, so this directory is discovered wholesale rather than by a
# filename pattern that only ever matched the two that shipped.
AGENTS_DIR = BACKEND_DIR / "data" / "agents"
_VALID_AGENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

CATALOG = Catalog.load()

DEFAULT_AGENT = "scheduler_flow"
# Which agent a test call runs. Held in memory rather than in a state file: the
# runner re-reads the JSON on every connection, so switching takes effect on the
# next call, and a restart sensibly returns to the shipped agent.
_live_agent = DEFAULT_AGENT


def live_agent_path() -> Path:
    """The agent bot.py should run. Falls back if the live one was deleted."""
    path = AGENTS_DIR / f"{_live_agent}.json"
    return path if path.exists() else AGENTS_DIR / f"{DEFAULT_AGENT}.json"


# ---- agents ----------------------------------------------------------------


def _agent_path(name: str) -> Path:
    """Resolve an agent id to its file, refusing anything that is not one.

    Two checks rather than one. The shape check is what users hit: the builder
    lets people name agents, and a name with a slash or a space is a mistake
    worth catching with a clear message. The resolve check is defence in depth,
    because reasoning about name shapes is easy to get subtly wrong -- an
    earlier version tested `Path(name).name != name`, which lets a bare ".."
    through untouched.
    """
    if not _VALID_AGENT_NAME.match(name or ""):
        raise HTTPException(
            400,
            "Agent names must start with a letter or digit and contain only "
            "letters, digits, dots, dashes and underscores.",
        )
    candidate = (AGENTS_DIR / f"{name}.json").resolve()
    if candidate.parent != AGENTS_DIR.resolve():
        raise HTTPException(400, "Invalid agent name.")
    return candidate


@app.get("/api/agents")
async def list_agents():
    agents = []
    for path in sorted(AGENTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        agents.append(
            {
                "id": path.stem,
                "name": data.get("name", path.stem),
                "nodes": len(data.get("nodes", [])),
                "active": path.stem == _live_agent,
            }
        )
    return {"agents": agents}


@app.get("/api/agents/template")
async def agent_template():
    """A minimal agent that already runs.

    Two wired nodes rather than one, so a freshly created agent greets, hangs up
    and can be called immediately — a lone node would compile but trap the
    caller in a step with no way out.
    """
    return {
        "name": "New agent",
        "voice_id": DEFAULT_VOICE_ID,
        "model": DEFAULT_MODEL,
        "persona": "You are a warm, concise assistant. Your words are spoken aloud, so "
        "avoid emojis, lists or anything unreadable. Keep replies to one or two short "
        "sentences.",
        "initial_node": "greeting",
        "nodes": [
            {
                "name": "greeting",
                "task_messages": [
                    {
                        "role": "developer",
                        "content": "Greet the caller and ask how you can help. When they "
                        "are done, call finish.",
                    }
                ],
                "edges": [
                    {
                        "function": "finish",
                        "description": "The caller has nothing further.",
                        "target": "farewell",
                    }
                ],
            },
            {
                "name": "farewell",
                "task_messages": [
                    {"role": "developer", "content": "Thank the caller and say goodbye."}
                ],
                "end": True,
            },
        ],
    }


@app.get("/api/agents/{name}")
async def get_agent(name: str):
    path = _agent_path(name)
    if not path.exists():
        raise HTTPException(404, f"No agent named {name!r}.")
    return json.loads(path.read_text())


@app.put("/api/agents/{name}")
async def save_agent(name: str, agent: dict):
    """Compile before writing.

    A graph that references a missing tool or an unreachable node is rejected
    here, at save time, rather than failing in front of a caller.
    """
    try:
        AgentBuilder.from_dict(agent)
    except (ValueError, KeyError) as error:
        return JSONResponse({"error": str(error)}, status_code=422)

    path = _agent_path(name)
    path.write_text(json.dumps(agent, ensure_ascii=False, indent=2) + "\n")
    logger.info(f"Saved agent '{name}' ({len(agent.get('nodes', []))} nodes)")
    return {"status": "saved", "id": name}


@app.post("/api/agents/{name}")
async def create_agent(name: str, agent: dict):
    """Create a new agent. Distinct from PUT so an existing one is never
    overwritten by a mistyped name."""
    path = _agent_path(name)
    if path.exists():
        return JSONResponse({"error": f"An agent named {name!r} already exists."}, status_code=409)
    return await save_agent(name, agent)


@app.delete("/api/agents/{name}")
async def delete_agent(name: str):
    path = _agent_path(name)
    if not path.exists():
        raise HTTPException(404, f"No agent named {name!r}.")
    if name == _live_agent:
        return JSONResponse(
            {"error": "That agent is live. Make another one live before deleting it."},
            status_code=409,
        )
    path.unlink()
    logger.info(f"Deleted agent '{name}'")
    return {"status": "deleted", "id": name}


@app.post("/api/agents/{name}/activate")
async def activate_agent(name: str):
    """Choose which agent the test call runs. Live on the next connection."""
    global _live_agent
    if not _agent_path(name).exists():
        raise HTTPException(404, f"No agent named {name!r}.")
    _live_agent = name
    logger.info(f"Agent '{name}' is now live; reconnect the test call to run it")
    return {"status": "live", "id": name}


@app.get("/api/tools")
async def list_tools():
    """What the node editor offers in its tool picker: the tools that exist."""
    return {"tools": describe_all()}


# ---- catalog ---------------------------------------------------------------


@app.get("/api/catalog/summary")
async def catalog_summary():
    unstaffed = unstaffed_appointment_types(CATALOG)
    return {
        "locations": len(CATALOG.locations),
        "providers": len(CATALOG.providers),
        "appointment_types": len(CATALOG.appointment_types),
        "specialties": len(CATALOG.specialties),
        "bookable_specialties": len(bookable_specialties(CATALOG)),
        # Surfaced rather than hidden: services the clinic advertises but has
        # nobody to staff are a data problem the UI should make visible.
        "unstaffed_appointment_types": [a.name for a in unstaffed],
        "policies": CATALOG.policies,
    }


@app.get("/api/catalog/locations")
async def catalog_locations():
    return {
        "locations": [
            {
                "id": l.id,
                "name": l.name,
                "address": l.address,
                "city": l.city,
                "phone": l.phone,
                "hours": l.hours,
                "capabilities": sorted(l.capabilities),
                "providers": len(CATALOG.providers_by_location.get(l.id, [])),
            }
            for l in sorted(CATALOG.locations.values(), key=lambda x: x.name)
        ]
    }


@app.get("/api/catalog/providers")
async def catalog_providers():
    return {
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "title": p.title,
                "specialty": p.specialty,
                "locations": sorted(CATALOG.locations[l].name for l in p.location_ids),
                "accepting_new_patients": p.accepting_new_patients,
                "languages": sorted(p.languages),
                "appointment_types": len(p.appointment_type_ids),
                # The catalog's own trap, made visible in the UI.
                "duplicate_name": sum(
                    1 for other in CATALOG.providers.values() if other.name == p.name
                ) > 1,
            }
            for p in sorted(CATALOG.providers.values(), key=lambda x: x.name)
        ]
    }


@app.get("/api/catalog/appointment-types")
async def catalog_appointment_types():
    return {
        "appointment_types": [
            {
                "id": a.id,
                "name": a.name,
                "specialty": a.specialty,
                "duration_min": a.duration_min,
                "requires_referral": a.requires_referral,
                "new_patients_allowed": a.new_patients_allowed,
                "required_capability": a.required_capability,
                "bookable_combinations": len(
                    find_bookable(CATALOG, appointment_type_id=a.id)
                ),
            }
            for a in sorted(CATALOG.appointment_types.values(), key=lambda x: x.name)
        ]
    }


# ---- live metrics ----------------------------------------------------------


@app.websocket("/api/metrics")
async def metrics(websocket: WebSocket):
    """Streams token usage, tool calls and node transitions to the UI panel."""
    await BROADCASTER.connect(websocket)
    try:
        while True:
            # Nothing is expected from the client; this keeps the socket open
            # until the panel closes it.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        BROADCASTER.disconnect(websocket)


# ---- the built UI ----------------------------------------------------------

if FRONTEND_DIST.exists():
    app.mount("/builder/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/builder")
    @app.get("/builder/{path:path}")
    async def builder(path: str = ""):
        """Serve the app shell, and never let a browser cache it.

        Asset filenames are content-hashed, so index.html is the only thing that
        says which build to load. Without an explicit header the browser caches
        it heuristically and keeps loading the previous bundle: you rebuild the
        UI, reload, and silently get the old app — which is a miserable thing to
        debug, because everything looks correctly deployed from the server side.
        """
        return FileResponse(
            FRONTEND_DIST / "index.html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

else:  # pragma: no cover - developer convenience

    @app.get("/builder")
    async def builder_not_built():
        return JSONResponse(
            {"error": "The builder UI has not been built yet. Run `make ui` first."},
            status_code=503,
        )
