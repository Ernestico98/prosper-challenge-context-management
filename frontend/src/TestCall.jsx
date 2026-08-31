import React, { useEffect, useRef, useState } from "react";

/**
 * The test call, plus the panel that makes Phase 2 visible.
 *
 * The context work is invisible from the outside: a good agent and an expensive
 * one sound identical. This shows what it actually costs — prompt tokens per
 * model call, which tools ran, and where the context was reset — live, next to
 * the conversation producing it.
 */
export default function TestCall({ agent }) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const socket = useRef(null);

  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/metrics`;
    const ws = new WebSocket(url);
    socket.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (message) => {
      try {
        setEvents((previous) => [...previous.slice(-200), JSON.parse(message.data)]);
      } catch {
        /* ignore malformed frames */
      }
    };
    return () => ws.close();
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
  const naive = 5449;
  const naiveEquivalent = naive * Math.max(totals.calls, 1);

  return (
    <div className="call-layout">
      <div className="call-frame">
        <iframe title="Pipecat test call" src="/client" allow="microphone" />
      </div>

      <aside className="metrics">
        <h2>
          Context cost
          <span className={connected ? "dot live" : "dot"} title={connected ? "live" : "offline"} />
        </h2>
        <p className="hint">
          Running <strong>{agent?.name || "—"}</strong>. Every model call on this call is
          measured; nothing here is estimated.
        </p>

        <div className="metric-grid">
          <Metric value={totals.prompt.toLocaleString()} label="prompt tokens" />
          <Metric value={totals.peak.toLocaleString()} label="largest single prompt" />
          <Metric value={totals.calls} label="model calls" />
          <Metric value={totals.tools} label="tool calls" />
        </div>

        {totals.calls > 0 && (
          <div className="comparison">
            <div>
              Naive equivalent (whole catalog re-sent every call):{" "}
              <strong>{naiveEquivalent.toLocaleString()}</strong> tokens
            </div>
            <div className="ratio">
              {(naiveEquivalent / Math.max(totals.prompt, 1)).toFixed(1)}× cheaper
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
