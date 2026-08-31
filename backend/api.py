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
from pathlib import Path

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.runner.run import app

from agent_builder import AgentBuilder
from catalog import (
    Catalog,
    bookable_specialties,
    find_bookable,
    unstaffed_appointment_types,
)
from observability import BROADCASTER
from tools import describe_all

BACKEND_DIR = Path(__file__).parent
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

CATALOG = Catalog.load()


# ---- agents ----------------------------------------------------------------


def _agent_path(name: str) -> Path:
    # Anything that is not a bare filename is a path traversal attempt.
    if Path(name).name != name:
        raise HTTPException(400, "Invalid agent name.")
    return BACKEND_DIR / f"{name}.json"


@app.get("/api/agents")
async def list_agents():
    agents = []
    for path in sorted(BACKEND_DIR.glob("*_flow.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        agents.append(
            {
                "id": path.stem,
                "name": data.get("name", path.stem),
                "nodes": len(data.get("nodes", [])),
                "active": path.name == "scheduler_flow.json",
            }
        )
    return {"agents": agents}


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
        return FileResponse(FRONTEND_DIST / "index.html")

else:  # pragma: no cover - developer convenience

    @app.get("/builder")
    async def builder_not_built():
        return JSONResponse(
            {"error": "The builder UI has not been built yet. Run `make ui` first."},
            status_code=503,
        )
