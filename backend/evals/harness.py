#
# A headless flow runner.
#
# Voice agents are miserable to test: the interesting failures are semantic, and
# reproducing them through a microphone is slow and unreliable. This drives the
# SAME compiled graph the voice pipeline runs -- same nodes, same tools, same
# context strategy -- over text, so accuracy becomes something you can measure
# in a loop rather than something you claim after a good demo.
#
# The caller's lines are scripted. When the agent asks something, the next
# scripted line is fed in. No second model plays the patient: it would double
# the cost and add a source of variance to the thing being measured.
#
# Once the script runs out the caller simply agrees a few times, because a real
# caller does not fall silent mid-booking and the scenarios are testing routing
# and legality, not how many questions the agent asks to get there.
#

import json
from dataclasses import dataclass, field


@dataclass
class Turn:
    speaker: str          # "caller" | "agent" | "tool"
    text: str


@dataclass
class Transcript:
    turns: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)   # (name, args, result)
    nodes_visited: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def tools_used(self) -> set:
        return {name for name, _, _ in self.tool_calls}

    def render(self) -> str:
        lines = []
        for turn in self.turns:
            prefix = {"caller": "CALLER", "agent": "AGENT ", "tool": "  tool"}[turn.speaker]
            lines.append(f"{prefix} | {turn.text}")
        return "\n".join(lines)


class HeadlessFlow:
    """Runs a compiled agent graph against a scripted caller."""

    FILLER = "Si, la primera me va bien, gracias."
    MAX_FILLER = 4

    def __init__(self, builder, ctx, client, *, model="gpt-4o", temperature=0.0, max_steps=40):
        self.builder = builder
        self.ctx = ctx
        self.client = client
        self.model = model
        # Pinned low on purpose. Real calls run at the provider default, but an
        # eval that varies run to run measures the sampler rather than the
        # system: the first version of this suite passed a scenario on one roll
        # and failed it on the next, with no code change in between. Measuring
        # production variance would mean N runs per scenario, which is a
        # different (and much more expensive) experiment.
        self.temperature = temperature
        self.max_steps = max_steps
        self.state = {}
        self.node = None
        self.finished = False
        self.messages = []
        self.transcript = Transcript()

    # ---- pipecat-flows compatibility --------------------------------------
    # The compiled handlers expect an object with `.state`; that is the whole
    # surface they touch, so this class can stand in for the FlowManager.

    def _enter(self, node_config, *, reset: bool) -> None:
        self.node = node_config
        # A terminal node ends the call in the real pipeline; here it ends the
        # loop, so we stop paying for turns after the goodbye.
        self.finished = any(
            action.get("type") == "end_conversation"
            for action in node_config.get("post_actions", [])
        )
        self.transcript.nodes_visited.append(node_config.get("name", "?"))
        if reset:
            # Mirrors ContextStrategy.RESET: the conversation so far is dropped
            # and the node starts from its own (templated) messages.
            self.messages = []
        # Mirror how Flows actually assembles context, because getting this wrong
        # makes the harness measure a system nobody ships:
        #
        #   role_message  -> the system instruction, persisting across nodes
        #                    (LLMUpdateSettingsFrame)
        #   task_messages -> APPENDED to the end of the conversation
        #                    (manager.py:833, messages.extend(task_messages))
        #
        # The node's task has to be the most recent thing the model sees. Folded
        # into the system prompt at position 0 it sits under the whole
        # conversation, and the model carries on with what it was doing instead
        # of doing what the node asks.
        persona = node_config.get("role_message", "")
        body = [m for m in self.messages if m["role"] != "system"]
        self.messages = ([{"role": "system", "content": persona}] if persona else []) + body
        for message in node_config.get("task_messages", []):
            # "developer" is the role Flows uses; chat.completions wants
            # "system", and the two mean the same thing here.
            role = message.get("role", "system")
            self.messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": message.get("content", ""),
                }
            )

    def _tool_schemas(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": fn.name,
                    "description": fn.description,
                    "parameters": {
                        "type": "object",
                        "properties": fn.properties,
                        "required": list(fn.required),
                    },
                },
            }
            for fn in self.node["functions"]
        ]

    # ---- the loop ----------------------------------------------------------
    async def run(self, script: list) -> Transcript:
        self._enter(self.builder.build_initial_node(), reset=False)
        pending = list(script)
        filler_used = 0

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                tools=self._tool_schemas() or None,
            )
            if response.usage:
                self.transcript.prompt_tokens += response.usage.prompt_tokens
                self.transcript.completion_tokens += response.usage.completion_tokens

            choice = response.choices[0].message
            self.messages.append(choice.model_dump(exclude_none=True))

            if choice.tool_calls:
                if await self._run_tool_calls(choice.tool_calls):
                    continue
                # A transition happened: the node (and possibly the context)
                # changed, so go straight back to the model.
                continue

            if choice.content:
                self.transcript.turns.append(Turn("agent", choice.content))
            if self.finished:
                break
            if pending:
                line = pending.pop(0)
            elif filler_used < self.MAX_FILLER:
                line = self.FILLER
                filler_used += 1
            else:
                break
            self.transcript.turns.append(Turn("caller", line))
            self.messages.append({"role": "user", "content": line})

        return self.transcript

    async def _run_tool_calls(self, tool_calls) -> bool:
        """Returns True if we stayed in the same node.

        The model can emit several tool calls in one message, and every
        tool_call_id must come back with a response or the provider rejects the
        next request outright. So a transition cannot break out of this loop:
        it is remembered and applied once every call has been answered.
        """
        transition = None

        for call in tool_calls:
            if transition is not None:
                # Belongs to the node we are leaving. Running it now would apply
                # the previous step's tools to the next step, so acknowledge it
                # and drop it.
                self._append_tool_result(
                    call.id,
                    {"status": "skipped", "reason": "the conversation moved to another step"},
                )
                continue

            fn = next((f for f in self.node["functions"] if f.name == call.function.name), None)
            if fn is None:
                self._append_tool_result(call.id, {"error": "unknown_function"})
                continue

            args = json.loads(call.function.arguments or "{}")
            result, next_node = await fn.handler(args, self)
            self.transcript.tool_calls.append((call.function.name, args, result))
            self.transcript.turns.append(
                Turn("tool", f"{call.function.name}({json.dumps(args, ensure_ascii=False)}) -> "
                             f"{json.dumps(result, ensure_ascii=False)[:200]}")
            )
            self._append_tool_result(call.id, result)

            if next_node is not None:
                transition = next_node

        if transition is None:
            return True

        strategy = transition.get("context_strategy")
        reset = bool(strategy) and strategy.strategy.value == "reset"
        self._enter(transition, reset=reset)
        return False

    def _append_tool_result(self, call_id: str, result) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
