#
# Voice pipeline — Prosper AI Software Engineer Challenge
#
# The runnable voice agent: WebRTC transport + ElevenLabs STT/TTS + OpenAI LLM,
# driven by a Pipecat Flows node graph. This file is generic — it loads an agent
# definition (JSON) via AgentBuilder and runs it. Swapping the agent is a data
# change (edit/replace the JSON), not a code change.
#
#   scheduler_flow.json  ->  AgentBuilder  ->  Pipecat Flows graph  ->  FlowManager
#
# The catalog never enters the prompt. It is reachable only through the tools in
# tools/, which are bound to the graph here via a per-call SchedulingContext.
#
# Run:  python bot.py   then open http://localhost:7860/client
#

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager

import api  # noqa: F401  (registers the builder UI routes on the runner's app)
from agent_builder import AgentBuilder
from catalog import Catalog, bookable_specialties, build_index
from observability import MetricsTap
from tools.context import SchedulingContext

# Load .env next to this file, so the bot runs the same from the repo root or backend/.
load_dotenv(Path(__file__).parent / ".env", override=True)


# The agent this bot runs. Point this at any agent JSON (the Phase 2 Copilot
# would generate one and drop it here).
AGENT_FLOW = Path(__file__).parent / "scheduler_flow.json"

# Catalog and retrieval index are process-wide: loading them per call would add
# startup latency to every conversation for data that never changes mid-session.
CATALOG = Catalog.load()
SPECIALTY_INDEX = build_index(CATALOG)

# Values an agent's task_messages may interpolate. `{specialties}` is the
# small-catalog escape hatch: injecting the list costs ~65 tokens but no round
# trip, which beats retrieval while the list stays short. The shipped scheduler
# does not use it — it retrieves instead, which is what survives a catalog ten
# times this size. See solution.md.
TEMPLATE_VALUES = {
    "specialties": ", ".join(bookable_specialties(CATALOG)),
    "clinic_name": "Prosper Clinic",
    # Default for a slot the conversation may never fill. Without it the prompt
    # would show the caller-facing model a literal "{provider_language}".
    "provider_preference": "nobody in particular",
    "location_preference": "no site in particular",
}


transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, builder: AgentBuilder
) -> None:
    config = builder.config
    logger.info(f"Starting '{config.name}' with {len(config.nodes)} nodes")

    stt = ElevenLabsRealtimeSTTService(api_key=os.environ["ELEVENLABS_API_KEY"])
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice=config.voice_id),
    )
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model=config.model)

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            # Observes token usage on its way past; never alters or delays it.
            MetricsTap(),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    flow_manager = FlowManager(
        llm=llm,
        context_aggregator=context_aggregator,
        worker=worker,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting flow at initial node")
        await flow_manager.initialize(builder.build_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point invoked by the Pipecat dev runner (and Pipecat Cloud).

    Called once per connection, so the agent JSON is re-read on every call:
    editing the graph in the builder UI and reconnecting is all it takes to run
    the new version. Bookings are per-call; the catalog is shared.
    """
    transport = await create_transport(runner_args, transport_params)
    tool_context = SchedulingContext(catalog=CATALOG, index=SPECIALTY_INDEX)
    builder = AgentBuilder.from_json(
        AGENT_FLOW, tool_context=tool_context, template_values=TEMPLATE_VALUES
    )
    await run_bot(transport, runner_args, builder)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
