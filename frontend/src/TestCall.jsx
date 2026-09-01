import React, { useEffect, useRef, useState } from "react";

/**
 * The test call, plus the panel that makes Phase 2 visible.
 *
 * The context work is invisible from the outside: a good agent and an expensive
 * one sound identical. This shows what it actually costs — prompt tokens per
 * model call, which tools ran, and where the context was reset — live, next to
 * the conversation producing it.
 */
export default function TestCall({ agent, live, onCallChange }) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  // Remounting the iframe hands the next call a fresh WebRTC client. The
  // prebuilt one does not survive a session the server ended: Connect posts
  // /start, gets a new session id, and then never sends an offer. The server is
  // healthy throughout, and the bug is in a dependency we ship rather than own,
  // so this reloads it instead of leaving people to press F5.
  const [clientKey, setClientKey] = useState(0);
  const socket = useRef(null);

  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/metrics`;
    const ws = new WebSocket(url);
    socket.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data);
        if (event.type === "call_started") {
          // Clear on start, not on end: clearing when the agent hangs up would
          // wipe the trace at the exact moment someone wants to read it.
          setEvents([event]);
          onCallChange?.(true);
          return;
        }
        if (event.type === "call_ended") {
          setClientKey((key) => key + 1);
          onCallChange?.(false);
        }
        setEvents((previous) => [...previous.slice(-200), event]);
      } catch {
        /* ignore malformed frames */
      }
    };
    return () => {
      onCallChange?.(false);
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const totals = events.reduce(
    (accumulator, event) => {
      if (event.type === "tokens") {
        accumulator.prompt += event.prompt_tokens || 0;
        accumulator.completion += event.completion_tokens || 0;
        accumulator.cached += event.cache_read_input_tokens || 0;
        accumulator.calls += 1;
        accumulator.peak = Math.max(accumulator.peak, event.prompt_tokens || 0);
      }
      if (event.type === "tool") accumulator.tools += 1;
      return accumulator;
    },
    { prompt: 0, completion: 0, cached: 0, calls: 0, tools: 0, peak: 0 }
  );

  // The comparison that makes the number mean something.
  // Tokens the catalog would add to *every* model call if it lived in the
  // prompt (measured by evals/benchmark_context.py). The naive agent sends
  // everything this one sends and the catalog on top, so the comparison is
  // additive: dividing the catalog alone by our total would compare two
  // different things and understate the difference.
  const CATALOG_IN_PROMPT = 5449;
  const catalogWouldAdd = CATALOG_IN_PROMPT * totals.calls;
  const naiveTotal = totals.prompt + catalogWouldAdd;

  return (
    <div className="call-layout">
      <div className="call-frame">
        <iframe key={clientKey} title="Pipecat test call" src="/client" allow="microphone" />
      </div>

      <aside className="metrics">
        <h2>
          Context cost
          <span className={connected ? "dot live" : "dot"} title={connected ? "live" : "offline"} />
          <div className="spacer" />
          <button
            className="ghost tiny"
            onClick={() => setClientKey((key) => key + 1)}
            title="Reload the embedded WebRTC client"
          >
            Reload client
          </button>
        </h2>
        <p className="hint">
          Running <strong>{live?.name || "—"}</strong>. Every model call is measured;
          nothing here is estimated.
        </p>
        {live && agent && live.name !== agent.name && (
          // Easy trap now that agents can be switched: you edit one and ring
          // another, then wonder why your change did nothing.
          <p className="hint warn">
            ⚠ You have <strong>{agent.name}</strong> open in the editor, but this call runs{" "}
            <strong>{live.name}</strong>. Set it live to hear your edits.
          </p>
        )}

        <div className="metric-grid">
          <Metric value={totals.prompt.toLocaleString()} label="input tokens, all calls" />
          <Metric value={totals.peak.toLocaleString()} label="largest single prompt" />
          <Metric value={totals.calls} label="model calls" />
          <Metric value={totals.tools} label="tool calls" />
        </div>

        {totals.calls > 0 && (
          <div className="comparison">
            <div>
              With the catalog in the prompt, this same conversation would have sent{" "}
              <strong>{naiveTotal.toLocaleString()}</strong> tokens: it adds{" "}
              {CATALOG_IN_PROMPT.toLocaleString()} to each of the {totals.calls} model call
              {totals.calls === 1 ? "" : "s"}.
            </div>
            <div className="ratio">
              {(naiveTotal / Math.max(totals.prompt, 1)).toFixed(1)}× cheaper
            </div>
          </div>
        )}

        <h3>Trace</h3>
        <ol className="trace">
          {events.length === 0 && (
            <li className="muted">Connect and start talking — events appear here.</li>
          )}
          {events
            .slice()
            .reverse()
            .map((event, index) => (
              <li key={index} className={event.type}>
                {event.type === "call_started" && (
                  <>
                    <code>call</code> started — {event.agent}
                  </>
                )}
                {event.type === "call_ended" && (
                  <>
                    <code>call</code> ended — client reloaded for the next one
                  </>
                )}
                {event.type === "tokens" && (
                  <>
                    <code>llm</code> {event.prompt_tokens?.toLocaleString()} in /{" "}
                    {event.completion_tokens?.toLocaleString()} out
                    {event.cache_read_input_tokens ? (
                      <span className="muted"> ({event.cache_read_input_tokens.toLocaleString()} cached)</span>
                    ) : null}
                  </>
                )}
                {event.type === "tool" && (
                  <>
                    <code>tool</code> {event.name}
                    <span className="muted"> {event.summary}</span>
                  </>
                )}
                {event.type === "node" && (
                  <>
                    <code className={event.reset ? "reset" : ""}>node</code> {event.name}
                    {event.reset && <span className="badge reset">context reset</span>}
                  </>
                )}
              </li>
            ))}
        </ol>
      </aside>
    </div>
  );
}

function Metric({ value, label }) {
  return (
    <div className="metric">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
